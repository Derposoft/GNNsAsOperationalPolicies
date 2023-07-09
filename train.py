"""
all outputted metrics can be found and visualized in tensorboard at ~/ray_results (on unix-based machines).

run `python train.py --help` for more information on how to start training a model.
"""

# general
import warnings

import argparse
import os
import pickle
import torch
import ray
import tempfile
import numpy as np
import random

# our code
from sigma_graph.envs.figure8.action_lookup import MOVE_LOOKUP, TURN_90_LOOKUP
from sigma_graph.envs.figure8.default_setup import OBS_TOKEN
from sigma_graph.envs.figure8.figure8_squad_rllib import Figure8SquadRLLib
from graph_scout.envs.base import ScoutMissionStdRLLib
import sigma_graph.envs.figure8.default_setup as default_setup
import model  # THIS NEEDS TO BE HERE IN ORDER TO RUN __init__.py!
import model.utils as utils

# algorithms to test
from ray.rllib.algorithms import ppo, dqn, pg, a3c, impala
from ray.rllib.models.catalog import MODEL_DEFAULTS
from ray.tune.logger import pretty_print, UnifiedLogger

warnings.filterwarnings("ignore", module="dgl")
log = True
ray.init(num_gpus=1, num_cpus=3, log_to_driver=log)  # test 1 cpu and 30 cpus
SEED = 0


# create env configuration
def create_env_config(config):
    # n_episodes = config.n_episode
    # init_red and init_blue should have number of agents dictionary elements if you want to specify it
    # [!!] remember to update this dict if adding new args in parser
    env_config = {
        # FIG8 PARAMETERS
        "env_path": config.env_path,
        # "max_step": config.max_step,
        "act_masked": config.act_masked,
        "n_red": config.n_red,
        "n_blue": config.n_blue,
        "init_red": config.init_red,
        "init_blue": config.init_blue,
        "init_health_red": config.init_health,
        "init_health_blue": config.init_health,
        "obs_embed": config.obs_embed,
        "obs_dir": config.obs_dir,
        "obs_team": config.obs_team,
        "obs_sight": config.obs_sight,
        "log_on": config.log_on,
        "log_path": config.log_path,
        # "reward_step_on": False, "reward_episode_on": True, "episode_decay_soft": True,
        # "health_lookup": {"type": "table", "reward": [8, 4, 2, 0], "damage": [0, 1, 2, 100]},
        # "faster_lookup": {"type": "none"},
        "fixed_start": config.fixed_start,
        # "aggregation_fn": config.aggregation_fn,
        # "hidden_size": config.hidden_size,
        # "is_hybrid": config.is_hybrid,
        # "conv_type": config.conv_type,
        # SCOUT PARAMETERS
        "num_red": config.n_red,
        "num_blue": config.n_blue,
    }
    # i.e. init_red "pos": tuple(x, z) or "L"/"R" region of the map
    # "init_red": [{"pos": (11, 1), "dir": 1}, {"pos": None}, {"pos": "L", "dir": None}]
    if hasattr(config, "penalty_stay"):
        env_config["penalty_stay"] = config.penalty_stay
    if hasattr(config, "threshold_blue"):
        env_config["threshold_damage_2_blue"] = config.threshold_blue
    if hasattr(config, "threshold_red"):
        env_config["threshold_damage_2_red"] = config.threshold_red
    return env_config  # , n_episodes


# store tb logs in custom named dir
def custom_log_creator(log_name, custom_dir="~/ray_results"):
    # https://stackoverflow.com/questions/62241261/change-logdir-of-ray-rllib-training-instead-of-ray-results
    custom_path = os.path.expanduser(custom_dir)
    log_name += "_"

    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=log_name, dir=custom_path)
        return UnifiedLogger(config, logdir)

    return logger_creator


# create trainer configuration
def create_trainer(config, trainer_type=None, custom_model=""):
    # check params
    trainer_types = [dqn, pg, a3c, ppo]
    assert trainer_type != None, f"trainer_type must be one of {trainer_types}"

    # initialize env and required config settings
    env = ScoutMissionStdRLLib if "scout" in custom_model else Figure8SquadRLLib
    env_config = create_env_config(config)
    setup_env = env(env_config)
    obs_space = setup_env.observation_space
    act_space = setup_env.action_space
    # policies = {}
    # for agent_id in setup_env.learning_agent:
    #    policies[str(agent_id)] = (None, obs_space, act_space, {})
    # policy mapping function not currently used.
    # def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    #    return str(agent_id)

    # create graph obs
    GRAPH_OBS_TOKEN = {
        "embed_dir": config.embed_dir,
        # different types of OPT subproblems to load
        "embed_opt": config.embed_opt,
        # does positioning on this node consistute "flanking" the enemy?
        "flanking": config.opt_flanking,
        "scout_high_ground": config.opt_scout_high_ground,
        "scout_high_ground_relevance": config.opt_scout_high_ground_relevance,
        "verbose": config.log_on,
    }

    # set model defaults
    CUSTOM_DEFAULTS = {
        "custom_model": custom_model,
        # Extra kwargs to be passed to your model"s c"tor.
        "custom_model_config": {
            "map": setup_env.map,
            "nred": env_config["n_red"],
            "nblue": env_config["n_blue"],
            "aggregation_fn": config.aggregation_fn,
            "hidden_size": config.hidden_size,
            "is_hybrid": config.is_hybrid,
            "conv_type": config.conv_type,
            "layernorm": config.layernorm,
            "graph_obs_token": GRAPH_OBS_TOKEN,
        },
    }

    model_config = CUSTOM_DEFAULTS if custom_model != "" else MODEL_DEFAULTS
    batch_size = config.batch_size
    is_scout = "scout" in custom_model
    return (
        ppo.PPOConfig()
        .environment(
            env=env,
            env_config=env_config,
            action_space=act_space,
            observation_space=obs_space,
        )
        .framework("torch")
        .resources(num_gpus=1, num_cpus_per_worker=1)
        .rollouts(
            rollout_fragment_length="auto",  # if not is_scout else 50,
            num_rollout_workers=1,
            # batch_mode="truncate_episodes",
        )  # 200)
        .evaluation(
            evaluation_interval=1,
            evaluation_duration_unit="episodes",
            evaluation_num_workers=1,
        )
        .debugging(
            logger_creator=lambda x: custom_log_creator(config.name)(x),
            log_level="INFO" if config.log_on else "ERROR",
        )
        .training(
            sgd_minibatch_size=batch_size,
            lr=config.lr,
            model=model_config,
            train_batch_size=batch_size,
        )
        .build()
    )


def train(trainer, model_name, train_time=200, checkpoint_models=True, config=None):
    assert model_name != "", "you must name your model. please use --name"
    if checkpoint_models:
        assert config != None, "configs must not be none if models are being saved."
    for _ in range(train_time):
        result = trainer.train()
        print(pretty_print(result))
    if checkpoint_models:
        # model_dir = "checkpoints/"+model_name+str(time.time()) + "/"
        model_dir = "checkpoints/" + model_name + "/"
        checkpoint_path = trainer.save(checkpoint_dir=model_dir + "model")
        with open(model_dir + "config.pkl", "wb") as f:
            pickle.dump(config, f)
        with open(model_dir + "checkpoint_path.txt", "w") as f:
            f.write(checkpoint_path)


# run baseline tests with a few different algorithms
def run_baselines(
    config,
    run_default_baseline_metrics=False,
    train_time=200,
    checkpoint_models=True,
    custom_model="graph_transformer_policy",
):
    """
    runs a set of baseline algorithms on the red v blue gym environment using rllib. the
    chosen algorithms are from the following list of algorithms:
    https://docs.ray.io/en/latest/rllib-algorithms.html#available-algorithms-overview

    the only requirements for an algorithm to function with the environment are:
    (a) Continuous Actions - Yes. (because MultiDiscrete counts as continuous :c...
        perhaps we can get rid of this requirement by "flattening" our action space into
        a more simple Discrete action space in the future)
    (b) Multi-Agent - Yes. Because the red v blue is a multi-agent environment.

    experimentally, ppo was the only one that performed/worked well with the gat model. therefore,
    the experiments are all focused around its use.
    """
    # get env config/setting seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    trainer = create_trainer(config, trainer_type=ppo, custom_model=custom_model)
    train(trainer, config.name, train_time, checkpoint_models, None)


# parse arguments
def parse_arguments():
    """
    Parses cli arguments. These arguments are used in create_env_config to create env arguments, and again in
    create_trainer to create the trainer and model, so if any arguments here should be passed to the model/trainer/environment
    artifacts, those functions must also be updated.
    """
    parser = argparse.ArgumentParser()
    # configs for sigma_graph env
    parser.add_argument(
        "--env_path", type=str, default=".", help="path of the project root"
    )
    parser.add_argument("--n_red", type=int, default=1, help="numbers of red agent")
    parser.add_argument("--n_blue", type=int, default=1, help="numbers of blue agent")
    parser.add_argument(
        "--n_episode",
        type=int,
        default=128,
        help="numbers of episodes per training cycle",
    )
    parser.add_argument(
        "--max_step", type=int, default=None, help="max step for each episode"
    )
    parser.add_argument(
        "--init_health", type=int, default=20, help="initial HP for all agents"
    )
    # advanced configs
    parser.add_argument(
        "--obs_embed_on",
        dest="obs_embed",
        action="store_true",
        default=OBS_TOKEN["obs_embed"],
        help="encoded embedding rather than raw one-hot POS",
    )
    parser.add_argument(
        "--obs_dir_off",
        dest="obs_dir",
        action="store_false",
        default=OBS_TOKEN["obs_dir"],
        help="observation self 4 dir",
    )
    parser.add_argument(
        "--obs_team_off",
        dest="obs_team",
        action="store_false",
        default=OBS_TOKEN["obs_team"],
        help="observation teammates",
    )
    parser.add_argument(
        "--obs_sight_off",
        dest="obs_sight",
        action="store_false",
        default=OBS_TOKEN["obs_sight"],
        help="observation in sight indicators",
    )
    parser.add_argument(
        "--act_masked_off",
        dest="act_masked",
        action="store_false",
        default=True,
        help="invalid action masking",
    )
    parser.add_argument(
        "--init_red",
        type=list,
        default=None,
        help="set init 'pos' and 'dir' for team red",
    )
    parser.add_argument(
        "--init_blue",
        type=list,
        default=None,
        help="set init 'route' and 'idx' for team blue",
    )
    parser.add_argument(
        "--log_on",
        dest="log_on",
        action="store_true",
        default=False,
        help="generate verbose logs",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="logs/temp/",
        help="relative path to the project root",
    )
    parser.add_argument(
        "--penalty_stay",
        type=int,
        default=0,
        help="penalty for take stay action [0: 'NOOP']",
    )
    parser.add_argument("--threshold_blue", default=2)
    parser.add_argument("--threshold_red", default=5)

    # model/training config
    parser.add_argument("--name", default="", help="name this model")
    parser.add_argument(
        "--model",
        default="graph_transformer",
        choices=[
            "graph_transformer",
            "hybrid",
            "fc",
            "gnn",
            "gt",
            "hybrid_scout",
            "fc_scout",
            "gnn_scout",
            "gt_scout",
        ],
    )
    parser.add_argument(
        "--is_hybrid",
        type=bool,
        default=True,
        help="choose between hybrid/not hybrid for gnn",
    )
    parser.add_argument("--conv_type", default="gcn", choices=["gcn", "gat"])
    parser.add_argument(
        "--layernorm",
        type=bool,
        default=False,
        help="add layer norm in between each layer of graph network",
    )
    parser.add_argument(
        "--aggregation_fn",
        type=str,
        default="agent_node",
        help="which output fn to use after gat",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=10, help="size of the hidden layer to use"
    )  # 169
    parser.add_argument(
        "--train_time", type=int, default=200, help="how long to train the model"
    )
    parser.add_argument(
        "--fixed_start",
        type=int,
        default=-1,
        help="where to fix the agent init points when training",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed to use for reproducibility purposes"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=800, help="batch size")

    # graph obs config
    parser.add_argument(
        "--embed_opt",
        action="store_true",
        default=False,
        help="embed graph optimization",
    )
    parser.add_argument(
        "--embed_dir",
        action="store_true",
        default=False,
        help="embed agent dirs",
    )
    parser.add_argument(
        "--opt_flanking",
        action="store_true",
        default=False,
        help="embed flanking subproblem",
    )
    parser.add_argument(
        "--opt_scout_high_ground",
        action="store_true",
        default=False,
        help="embed high ground subproblem",
    )
    parser.add_argument(
        "--opt_scout_high_ground_relevance",
        action="store_true",
        default=False,
        help="embed high ground relevance subproblem",
    )

    # testing config
    parser.add_argument(
        "--policy_file",
        type=str,
        default="",
        help="use hardcoded policy from provided policy file",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # parse args
    config = parse_arguments()
    SEED = config.seed

    # run models
    run_baselines(
        config,
        custom_model=config.model + "_policy",
        train_time=config.train_time,
        checkpoint_models=False,
    )
