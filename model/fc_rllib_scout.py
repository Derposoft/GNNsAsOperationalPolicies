"""
modified variant of https://github.com/ray-project/ray/blob/master/rllib/models/torch/fcnet.py

most of this code is the same as the code on the linked github repo above; there was no reason to
rebuild one from scratch when one existed. 
"""

import sys
import numpy as np
import gymnasium as gym

import ray.rllib.models.torch.torch_modelv2 as TMv2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

# RL/AI imports
# from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
import ray.rllib.models.torch.torch_modelv2 as TMv2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
import torch.nn as nn
import torch

# our code imports
# from sigma_graph.data.graph.skirmish_graph import MapInfo
from graph_scout.envs.data.terrain_graph import MapInfo
import model.utils as utils

# other imports
import numpy as np


class FCScoutPolicy(TMv2.TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        map: MapInfo,
        **kwargs
    ):
        TMv2.TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        utils.set_obs_token(kwargs["graph_obs_token"])
        self.embed_opt = kwargs["graph_obs_token"]["embed_opt"]
        self.map = map
        self.hidden_size = kwargs["hidden_size"]
        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
            model_config.get("post_fcnet_hiddens", [])
        )
        # hiddens = [170, 170] # ensures that this model has ~90k params
        # hiddens = [177, 177] # ensures that this model has ~96k params
        # hiddens = [200, 200] # ~160k params
        # hiddens = [100, 50] # ~100k params
        # hiddens = [50, 25] # ~85k params
        # hiddens = [20, 10] # ~76k
        # hiddens = [10, 5] # ~2.5k
        hiddens = [self.hidden_size, self.hidden_size // 2]

        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers")
        self.free_log_std = model_config.get("free_log_std")
        num_inputs = int(np.product(obs_space.shape))
        if self.embed_opt:
            map_size = self.map.get_graph_size()
            num_inputs += (
                (4 if utils.GRAPH_OBS_TOKEN["flanking"] else 0)
                + (map_size if utils.GRAPH_OBS_TOKEN["scout_high_ground"] else 0)
                + (
                    map_size
                    if utils.GRAPH_OBS_TOKEN["scout_high_ground_relevance"]
                    else 0
                )
            )

        # policy
        self._logits = None
        self._hidden_layers = None
        self._hidden_layers, self._logits = utils.create_policy_fc(
            hiddens=hiddens,
            activation=activation,
            num_outputs=num_outputs,
            no_final_linear=no_final_linear,
            num_inputs=num_inputs,
        )

        # value
        self._value_branch, self._value_branch_separate = utils.create_value_branch(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            vf_share_layers=self.vf_share_layers,
            activation=activation,
            hiddens=utils.VALUE_HIDDENS,
        )
        # hold previous inputs
        self._features = None
        self._last_flat_in = None

        utils.count_model_params(self, True)

    @override(TMv2.TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ):
        utils.timeit("start")
        current_device = next(self.parameters()).device
        obs = input_dict["obs_flat"].float()
        x = input_dict["obs_flat"].float()
        utils.timeit("get obs flat")

        if self.embed_opt:
            pos_obs_size = self.map.get_graph_size()
            if utils.GRAPH_OBS_TOKEN["flanking"]:
                opts = []
                for x in obs:
                    blue_positions = set([])
                    for j in range(pos_obs_size):
                        if x[pos_obs_size : 2 * pos_obs_size][j]:
                            blue_positions.add(j)
                    opt = utils.flank_optimization_scout(
                        self.map,
                        utils.get_loc(x, pos_obs_size),
                        blue_positions,
                    )
                    opt_vector = [0] * 4
                    if opt != 0:
                        opt_vector[opt - 1] = 1
                    opts.append(opt_vector)
                opt_flanking = torch.Tensor(opts)
                x = torch.cat([x, opts], dim=-1)
                utils.timeit("embed flanking subproblem")
            if utils.GRAPH_OBS_TOKEN["scout_high_ground"]:
                opt_high_ground = utils.scout_get_high_ground_embeddings(
                    len(obs), pos_obs_size, current_device
                ).reshape([len(obs), pos_obs_size])
                x = torch.cat([x, opt_high_ground], dim=-1)
                utils.timeit("embed HighGround subproblem")
            if utils.GRAPH_OBS_TOKEN["scout_high_ground_relevance"]:
                opt_high_ground_relevance = (
                    utils.scout_get_high_ground_embeddings_relevance(
                        obs, self.map, current_device
                    ).reshape([len(obs), pos_obs_size])
                )
                utils.timeit("embed HighGroundRelevance subproblem")
                x = torch.cat([x, opt_high_ground_relevance], dim=-1)

        obs = x
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features) if self._logits else self._features
        utils.timeit("hiddens and logits")

        return logits, state

    @override(TMv2.TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if not self._value_branch:
            current_device = next(self.parameters()).device
            return torch.Tensor([0] * len(self._features)).to(current_device)
        if self._value_branch_separate:
            return self._value_branch(
                self._value_branch_separate(self._last_flat_in)
            ).squeeze(1)
        else:
            return self._value_branch(self._features).squeeze(1)
