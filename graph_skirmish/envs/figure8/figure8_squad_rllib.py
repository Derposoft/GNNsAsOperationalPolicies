import gymnasium.spaces as spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models.catalog import MODEL_DEFAULTS
import numpy as np

from graph_skirmish.envs.figure8.figure8_squad import Figure8Squad
from . import default_setup as env_setup

local_action_move = env_setup.act.MOVE_LOOKUP
local_action_turn = env_setup.act.TURN_90_LOOKUP


# a variant of figure8_squad that can be used by rllib multiagent setups
# reference: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py
class Figure8SquadRLLib(Figure8Squad, MultiAgentEnv):
    def __init__(self, config=None):
        config = config or {}
        super().__init__(**config)

        # extra values to make graph embedding viable
        num_extra_graph_obs = 0  # 5 if self.obs_token["obs_graph"] else 0
        # self.action_space = spaces.MultiDiscrete([len(local_action_move), len(local_action_turn)])
        # "flatten" the above action space into the below discrete action space
        self.action_space = spaces.Discrete(
            len(local_action_move) * len(local_action_turn)
        )
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.state_shape + num_extra_graph_obs,),
            dtype=np.int8,
        )
        self.done = set()

    # return an arbitrary encoding from the "flat" action space to the normal action space 0-indexed
    def convert_discrete_action_to_multidiscrete(self, action):
        return [action % len(local_action_move), action // len(local_action_move)]

    def convert_multidiscrete_action_to_discrete(move_action, turn_action):
        return turn_action * len(local_action_move) + move_action

    def reset(self, *, seed: int = 0, options=None):
        _resets = super().reset()
        resets = {}
        for idx in range(len(_resets)):
            resets[str(self.learning_agent[idx])] = _resets[idx]
        self.done = set()
        return resets, {}

    def step(self, _n_actions: dict):
        # reference: https://docs.ray.io/en/latest/rllib-env.html#pettingzoo-multi-agent-environments
        # undictify the actions to interface rllib -> env input
        n_actions = []
        for a in self.learning_agent:
            if str(a) in _n_actions:
                n_actions.append(
                    self.convert_discrete_action_to_multidiscrete(
                        _n_actions.get(str(a))
                    )
                )
            else:
                n_actions.append(self.convert_discrete_action_to_multidiscrete(0))
        _obs, _rew, _done, _ = super().step(n_actions)

        # dictify the observations to interface env output -> rllib
        obs, rew, done = {}, {}, {}
        all_done = True
        for a_id in self.learning_agent:
            if a_id in self.done:
                continue
            obs[str(a_id)] = _obs[a_id]
            rew[str(a_id)] = _rew[a_id]
            done[str(a_id)] = _done[a_id]
            if _done[a_id]:
                self.done.add(a_id)
            # for some reason in rllib MARL __all__ must be included in 'done' dict
            all_done = all_done and _done[a_id]
        done["__all__"] = all_done

        return obs, rew, done, {}
