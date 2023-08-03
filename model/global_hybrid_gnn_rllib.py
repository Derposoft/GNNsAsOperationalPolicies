"""
base class from https://github.com/ray-project/ray/blob/master/rllib/models/torch/fcnet.py
"""
import torch.nn as nn
import torch
import ray.rllib.models.torch.torch_modelv2 as TMv2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
import gym

import dgl
import torch_geometric
from torch_geometric.nn.conv import GATv2Conv, GCNConv
from torch_geometric.nn.norm import BatchNorm
import networkx as nx
import numpy as np

from sigma_graph.data.graph.skirmish_graph import MapInfo
from sigma_graph.envs.figure8 import default_setup as env_setup
from sigma_graph.envs.figure8.action_lookup import MOVE_LOOKUP
from sigma_graph.envs.figure8.figure8_squad_rllib import Figure8SquadRLLib
import model.utils as utils

from ray.rllib.models.torch.misc import SlimFC, normc_initializer


class GlobalHybridGNNPolicy(TMv2.TorchModelV2, nn.Module):
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
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )
        nn.Module.__init__(self)

        """
        values that we need to instantiate and use GNNs
        """
        utils.set_obs_token(kwargs["graph_obs_token"])
        (
            hiddens,
            activation,
            no_final_linear,
            self.vf_share_layers,
            self.free_log_std,
        ) = utils.parse_config(model_config)
        self.map = map
        self.num_red = kwargs["nred"]
        self.num_blue = kwargs["nblue"]
        self.aggregation_fn = kwargs["aggregation_fn"]
        self.hidden_size = kwargs["hidden_size"]
        self.is_hybrid = kwargs["is_hybrid"]  # is this a hybrid model or a gat-only model?
        self.conv_type = kwargs["conv_type"]
        self.layernorm = kwargs["layernorm"]
        self_shape, blue_shape, red_shape = env_setup.get_state_shapes(
            self.map.get_graph_size(),
            self.num_red,
            self.num_blue,
            env_setup.OBS_TOKEN,
        )
        self.obs_shapes = [
            self_shape,
            blue_shape,
            red_shape,
            self.num_red,
            self.num_blue,
        ]
        self.adjacency = []
        for n in map.g_acs.adj:
            ms = map.g_acs.adj[n]
            for m in ms:
                self.adjacency.append([n-1, m-1])
        self.adjacency = torch.LongTensor(self.adjacency).t().contiguous()
        self._features = None  # current "base" output before logits
        self._last_flat_in = None  # last input

        """
        instantiate policy and value networks
        """
        self.GAT_LAYERS = 4
        self.N_HEADS = 1 if self.conv_type == "gcn" else 4
        self.HIDDEN_DIM = 4
        self.hiddens = [self.hidden_size, self.hidden_size] # TODO TEMP removed //2
        #self.hiddens = [169, 169]
        gat = GATv2Conv if self.conv_type == "gat" else GCNConv
        self.gats = nn.ModuleList([
            gat(
                in_channels=utils.NODE_EMBED_SIZE if i == 0 else self.HIDDEN_DIM*self.N_HEADS,
                out_channels=self.HIDDEN_DIM,
                heads=self.N_HEADS,
            )
            for i in range(self.GAT_LAYERS)
        ])
        if self.layernorm:
            self.norms = nn.ModuleList([
                BatchNorm(len(list(self.map.g_acs.adj.keys())))
                for _ in range(self.GAT_LAYERS)
            ])
        else:
            self.norms = [None]*self.GAT_LAYERS
        self.aggregator = utils.GeneralGNNPooling(
            aggregator_name=self.aggregation_fn,
            input_dim=self.HIDDEN_DIM*self.N_HEADS,
            output_dim=int(action_space.n)
        )
        self._hiddens, self._logits = utils.create_policy_fc(
            hiddens=self.hiddens,
            activation=activation,
            num_outputs=num_outputs,
            no_final_linear=no_final_linear,
            num_inputs=int(np.product(obs_space.shape))+num_outputs,
        )
        self._value_branch, self._value_branch_separate = utils.create_value_branch(
            num_inputs=int(np.product(obs_space.shape)),
            num_outputs=num_outputs,
            vf_share_layers=self.vf_share_layers,
            activation=activation,
            hiddens=utils.VALUE_HIDDENS,
        )

        self.agg = torch_geometric.nn.MeanAggregation()

        self.global_classifier_1 = SlimFC(in_size=self.N_HEADS*self.HIDDEN_DIM, out_size=self.N_HEADS*self.HIDDEN_DIM*2, initializer=normc_initializer(1.0),
                                      activation_fn=activation)
        self.global_classifier_2 = SlimFC(in_size=self.N_HEADS*self.HIDDEN_DIM*2, out_size=self.N_HEADS*self.HIDDEN_DIM*4, initializer=normc_initializer(1.0),
                                      activation_fn=activation)

        self.edge_classifier_1 = SlimFC(in_size=self.N_HEADS*self.HIDDEN_DIM*2+3+(4*self.N_HEADS*self.HIDDEN_DIM), out_size=self.N_HEADS*self.HIDDEN_DIM, initializer=normc_initializer(1.0),
                                      activation_fn=activation)
        self.edge_classifier_2 = SlimFC(in_size=self.N_HEADS*self.HIDDEN_DIM, out_size=self.N_HEADS*self.HIDDEN_DIM//2, initializer=normc_initializer(1.0),
                                        activation_fn=activation)
        self.edge_classifier_3 = SlimFC(in_size=self.N_HEADS*self.HIDDEN_DIM//2, out_size=1, initializer=normc_initializer(1.0))

        """
        produce debug output and ensure that model is on right device
        """
        utils.count_model_params(self, print_model=True)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to("cpu")

    @override(TMv2.TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ):
        # transform obs to graph (for pyG, also do list[data]->torch_geometric.Batch)
        obs = input_dict["obs_flat"].float()
        current_device = next(self.parameters()).device
        x = utils.efficient_embed_obs_in_map(obs, self.map, self.obs_shapes).to(current_device)
        agent_nodes = [utils.get_loc(gx, self.map.get_graph_size()) for gx in obs]
        
        self.adjacency = self.adjacency.to(current_device)
        # inference
        for conv, norm in zip(self.gats, self.norms):
            x = torch.stack([conv(_x, self.adjacency) for _x in x], dim=0)
            if self.layernorm: x = norm(x)
        # x.size() is [-, 27, 4]
        # torch.flatten(x, start_dim=1).size() is [-, 108]
        self._features = x # self._features = self.aggregator(x, self.adjacency, agent_nodes=agent_nodes)
        # if self.is_hybrid:
        #     self._features = self._hiddens(torch.cat([self._features, obs], dim=1))
        logits = torch.zeros((self._features.shape[0], 15)).to(current_device) # logits = self._logits(self._features)

        matching_tensor_indices = [(self.adjacency[0] == i).nonzero() for i in agent_nodes] # Indicies of possible edges from agent locations in self.adjacency
        edge_pairs_for_masking = [(torch.index_select(self.adjacency, 1, i.squeeze())) for i in matching_tensor_indices] # Edges for agent nodes in the form [agent][[source0, source1, ...], [target0, target1, ...]]

        #print(f"agent_nodes: {agent_nodes}")
        #print(f"matching_tensor_indices: {matching_tensor_indices}")
        #print(f"edge_pairs_for_masking: {edge_pairs_for_masking}")

        total_edges = []
        for edge_pairs in edge_pairs_for_masking:
            edges = []
            for id in range(len(edge_pairs[0])):
                edges += [[edge_pairs.tolist()[0][id], edge_pairs.tolist()[1][id]]]
            total_edges += [edges]
        total_actions = []
        for batch_edge in total_edges:
            collect_actions = []
            for edge in batch_edge:

                if edge[0] == edge[1]:
                    collect_actions += [0]
                else:
                    collect_actions += [self.map.g_acs[edge[0] + 1][edge[1] + 1]['action']]
            total_actions += [collect_actions]

        def convert_multidiscrete_action_to_discrete(move_action, turn_action):
            return turn_action * len([0, 1, 2, 3, 4]) + move_action

        for batch_id in range(len(total_edges)):
            agg = self.agg(self._features[batch_id])
            agg = self.global_classifier_1(agg)
            agg = self.global_classifier_2(agg)[0]
            for edge_id in range(len(total_edges[batch_id])):

                source = self._features[batch_id][total_edges[batch_id][edge_id][0]]
                target = self._features[batch_id][total_edges[batch_id][edge_id][1]]

                # [1, 0, 0] Case
                act_tensor = torch.tensor([1, 0, 0], dtype = torch.float).to(current_device)
                feats = torch.cat((source, act_tensor, target, agg)).to(current_device)

                feats = self.edge_classifier_1(feats)
                feats = self.edge_classifier_2(feats)
                feats = self.edge_classifier_3(feats)

                logits[batch_id][convert_multidiscrete_action_to_discrete(total_actions[batch_id][edge_id], 0)] = feats

                # [0, 1, 0] Case
                act_tensor = torch.tensor([0, 1, 0], dtype = torch.float).to(current_device)
                feats = torch.cat((source, act_tensor, target, agg)).to(current_device)

                feats = self.edge_classifier_1(feats)
                feats = self.edge_classifier_2(feats)
                feats = self.edge_classifier_3(feats)

                logits[batch_id][convert_multidiscrete_action_to_discrete(total_actions[batch_id][edge_id], 1)] = feats

                # [0, 0, 1] Case
                act_tensor = torch.tensor([0, 0, 1], dtype = torch.float).to(current_device)
                feats = torch.cat((source, act_tensor, target, agg)).to(current_device)

                feats = self.edge_classifier_1(feats)
                feats = self.edge_classifier_2(feats)
                feats = self.edge_classifier_3(feats)

                logits[batch_id][convert_multidiscrete_action_to_discrete(total_actions[batch_id][edge_id], 2)] = feats


        # return
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        return logits, state

    @override(TMv2.TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        if not self._value_branch:
            current_device = next(self.parameters()).device
            return torch.Tensor([0]*len(self._features)).to(current_device)
        if self._value_branch_separate:
            return self._value_branch(
                self._value_branch_separate(self._last_flat_in)
            ).squeeze(1)
        else:
            return self._value_branch(self._features).squeeze(1)
