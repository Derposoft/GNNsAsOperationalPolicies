"""
base class from https://github.com/ray-project/ray/blob/master/rllib/models/torch/fcnet.py
"""
import torch.nn as nn
import torch
import ray.rllib.models.torch.torch_modelv2 as TMv2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
import gymnasium as gym
import torch_geometric
from torch_geometric.nn.conv import GATv2Conv, GCNConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.data import Batch, Data as GraphData
import numpy as np
import time

from graph_scout.envs.data.terrain_graph import MapInfo
from graph_scout.envs.utils.config import default_configs as env_setup
import model.utils as utils

from ray.rllib.models.torch.misc import SlimFC, normc_initializer

class TwoGNNDirectScoutPolicy(TMv2.TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        map: MapInfo,
        **kwargs,
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
        self.is_hybrid = kwargs[
            "is_hybrid"
        ]  # is this a hybrid model or a gat-only model?
        self.conv_type = kwargs["conv_type"]
        self.layernorm = kwargs["layernorm"]
        self.adjacency = []
        for n in map.g_move.adj:
            ms = map.g_move.adj[n]
            for m in ms:
                self.adjacency.append([n - 1, m - 1])
        self.adjacency = torch.LongTensor(self.adjacency).t().contiguous()
        self._features = None  # current "base" output before logits
        self._last_flat_in = None  # last input
        self.action_space_output_dim = np.sum(action_space.nvec)

        """
        instantiate policy and value networks
        """
        # self.GAT_LAYERS = 4
        # self.N_HEADS = 1 if self.conv_type == "gcn" else 4
        # self.HIDDEN_DIM = 4
        self.GAT_LAYERS = 2  # 1_6:1804, 3_50: ~15k
        self.N_HEADS = 1 if self.conv_type == "gcn" else 4
        self.HIDDEN_DIM = 2

        self.hiddens = [self.hidden_size, self.hidden_size // 2]  # TODO withut //2
        gnn = GATv2Conv if self.conv_type == "gat" else GCNConv
        self.move_gats = nn.ModuleList(
            [
                gnn(
                    in_channels=utils.SCOUT_NODE_EMBED_SIZE
                    if i == 0
                    else self.HIDDEN_DIM * self.N_HEADS,
                    out_channels=self.HIDDEN_DIM,
                    heads=self.N_HEADS,
                )
                for i in range(self.GAT_LAYERS)
            ]
        )
        self.look_gats = nn.ModuleList(
            [
                gnn(
                    in_channels=utils.SCOUT_NODE_EMBED_SIZE
                    if i == 0
                    else self.HIDDEN_DIM * self.N_HEADS,
                    out_channels=self.HIDDEN_DIM,
                    heads=self.N_HEADS,
                )
                for i in range(self.GAT_LAYERS)
            ]
        )
        self.move_norms = nn.ModuleList(
            [
                BatchNorm(len(list(self.map.g_move.adj.keys())))
                for _ in range(self.GAT_LAYERS)
            ]
        )
        self.look_norms = nn.ModuleList(
            [
                BatchNorm(len(list(self.map.g_move.adj.keys())))
                for _ in range(self.GAT_LAYERS)
            ]
        )
        self.aggregator = utils.GeneralGNNPooling(
            aggregator_name=self.aggregation_fn,
            input_dim=self.HIDDEN_DIM * self.N_HEADS,
            output_dim=self.action_space_output_dim,
        )
        if self.is_hybrid:
            self._hiddens, self._logits = utils.create_policy_fc(
                hiddens=self.hiddens,
                activation=activation,
                num_outputs=num_outputs,
                no_final_linear=no_final_linear,
                num_inputs=int(np.product(obs_space.shape)) + num_outputs,
            )
        else:
            self._hiddens, self._logits = (
                None,
                utils.create_policy_fc(
                    hiddens=self.hiddens,
                    activation=activation,
                    num_outputs=num_outputs,
                    no_final_linear=no_final_linear,
                    num_inputs=int(np.product(obs_space.shape)) + num_outputs,
                )[1],
            )

        self._value_branch, self._value_branch_separate = utils.create_value_branch(
            num_inputs=int(np.product(obs_space.shape)),
            num_outputs=num_outputs,
            vf_share_layers=self.vf_share_layers,
            activation=activation,
            hiddens=utils.VALUE_HIDDENS,
        )

        self.aggregator = torch_geometric.nn.MeanAggregation()

        self.move_global_classifier_1 = SlimFC(in_size=self.N_HEADS*self.HIDDEN_DIM, out_size=self.N_HEADS*self.HIDDEN_DIM*2, initializer=normc_initializer(1.0),
                                      activation_fn=activation)
        self.move_global_classifier_2 = SlimFC(in_size=self.N_HEADS*self.HIDDEN_DIM*2, out_size=self.N_HEADS*self.HIDDEN_DIM*4, initializer=normc_initializer(1.0),
                                      activation_fn=activation)
        
        self.look_global_classifier_1 = SlimFC(in_size=self.N_HEADS*self.HIDDEN_DIM, out_size=self.N_HEADS*self.HIDDEN_DIM*2, initializer=normc_initializer(1.0),
                                      activation_fn=activation)
        self.look_global_classifier_2 = SlimFC(in_size=self.N_HEADS*self.HIDDEN_DIM*2, out_size=self.N_HEADS*self.HIDDEN_DIM*4, initializer=normc_initializer(1.0),
                                      activation_fn=activation)
        
        self.move_edge_classifier_1 = SlimFC(in_size=self.N_HEADS*self.HIDDEN_DIM*2+(4*self.N_HEADS*self.HIDDEN_DIM), out_size=self.N_HEADS*self.HIDDEN_DIM, initializer=normc_initializer(1.0),
                                      activation_fn=activation)
        self.move_edge_classifier_2 = SlimFC(in_size=self.N_HEADS*self.HIDDEN_DIM, out_size=self.N_HEADS*self.HIDDEN_DIM//2, initializer=normc_initializer(1.0),
                                        activation_fn=activation)
        self.move_edge_classifier_3 = SlimFC(in_size=self.N_HEADS*self.HIDDEN_DIM//2, out_size=1, initializer=normc_initializer(1.0))

        self.look_edge_classifier_1 = SlimFC(in_size=self.N_HEADS*self.HIDDEN_DIM*2+(4*self.N_HEADS*self.HIDDEN_DIM), out_size=self.N_HEADS*self.HIDDEN_DIM, initializer=normc_initializer(1.0),
                                      activation_fn=activation)
        self.look_edge_classifier_2 = SlimFC(in_size=self.N_HEADS*self.HIDDEN_DIM, out_size=self.N_HEADS*self.HIDDEN_DIM//2, initializer=normc_initializer(1.0),
                                        activation_fn=activation)
        self.look_edge_classifier_3 = SlimFC(in_size=self.N_HEADS*self.HIDDEN_DIM//2, out_size=1, initializer=normc_initializer(1.0))

        """
        produce debug output and ensure that model is on right device
        """
        utils.count_model_params(self, print_model=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    @override(TMv2.TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ):
        utils.timeit("start")

        # transform obs to graph (for pyG, also do list[data]->torch_geometric.Batch)
        obs = input_dict["obs_flat"].float()
        utils.timeit("obs to float")

        current_device = next(self.parameters()).device
        x = utils.scout_embed_obs_in_map(obs, self.map, current_device)
        batch_size = x.shape[0]
        graph_size = x.shape[1]
        utils.timeit("scout embed obs")

        agent_nodes = [utils.get_loc(gx, self.map.get_graph_size()) for gx in obs]
        utils.timeit("get agent nodes")

        s_gnn = time.time()

        self.adjacency = self.adjacency.to(current_device)
        # inference
        if self.conv_type == "gat":
            graph = Batch.from_data_list([GraphData(_x, self.adjacency) for _x in x])
            x, batch_adjacency = graph.x, graph.edge_index
            utils.timeit("batching")

        # Move
        x_move = x.clone()
        for conv, norm in zip(self.move_gats, self.move_norms):
            if self.conv_type == "gcn":
                x_move = conv(x_move, self.adjacency)
            else:
                x_move = conv(x_move, batch_adjacency)  # batching=8.9ms, convs=26.3ms
                # x = torch.stack([conv(_x, self.adjacency) for _x in x], dim=0) # 180-190ms
            if self.layernorm:
                x_move = norm(x_move)
        utils.timeit("gnn convs")

        x_move = x_move.reshape([batch_size, graph_size, -1])
        self._move_features = x_move # self._features = self.aggregator(x, self.adjacency, agent_nodes=agent_nodes)

        # Look
        x_look = x.clone()
        for conv, norm in zip(self.look_gats, self.look_norms):
            if self.conv_type == "gcn":
                x_look = conv(x_look, self.adjacency)
            else:
                x_look = conv(x_look, batch_adjacency)  # batching=8.9ms, convs=26.3ms
                # x = torch.stack([conv(_x, self.adjacency) for _x in x], dim=0) # 180-190ms
            if self.layernorm:
                x_look = norm(x_look)
        utils.timeit("gnn convs")

        x_look = x_look.reshape([batch_size, graph_size, -1])
        self._look_features = x_look # self._features = self.aggregator(x, self.adjacency, agent_nodes=agent_nodes)

        self._features = self._move_features.clone()

        e_gnn = time.time()

        logits = torch.zeros((self._move_features.shape[0], 11)).to(current_device)

        s_edge = time.time()

        matching_tensor_indices = [(self.adjacency[0] == i).nonzero() for i in agent_nodes] # Indicies of possible edges from agent locations in self.adjacency
        edge_pairs_for_masking = [(torch.index_select(self.adjacency, 1, i.squeeze())) for i in matching_tensor_indices] # Edges for agent nodes in the form [agent][[source0, source1, ...], [target0, target1, ...]]

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
                    collect_actions += [self.map.g_move[edge[0] + 1][edge[1] + 1]['action']]
            total_actions += [collect_actions]

        e_edge = time.time()

        s_network = time.time()

        # move_agg = self.aggregator(self._move_features)
        # move_agg = self.move_global_classifier_1(move_agg)
        # move_agg = self.move_global_classifier_2(move_agg)

        # look_agg = self.aggregator(self._look_features)
        # look_agg = self.look_global_classifier_1(look_agg)
        # look_agg = self.look_global_classifier_2(look_agg)

        for batch_id in range(len(total_edges)):
            # Set crouch to 1
            logits[batch_id][-1] = 1

            move_agg = self.aggregator(self._move_features[batch_id])
            move_agg = self.move_global_classifier_1(move_agg)
            move_agg = self.move_global_classifier_2(move_agg)[0]

            look_agg = self.aggregator(self._look_features[batch_id])
            look_agg = self.look_global_classifier_1(look_agg)
            look_agg = self.look_global_classifier_2(look_agg)[0]

            for edge_id in range(len(total_edges[batch_id])):
                
                move_source = self._move_features[batch_id][total_edges[batch_id][edge_id][0]]
                move_target = self._move_features[batch_id][total_edges[batch_id][edge_id][1]]

                move_feats = torch.cat((move_source, move_target, move_agg))
                move_feats = self.move_edge_classifier_1(move_feats)
                move_feats = self.move_edge_classifier_2(move_feats)
                move_feats = self.move_edge_classifier_3(move_feats)

                logits[batch_id][total_actions[batch_id][edge_id]] = move_feats

                if not(total_actions[batch_id][edge_id] == 0):

                    look_source = self._look_features[batch_id][total_edges[batch_id][edge_id][0]]
                    look_target = self._look_features[batch_id][total_edges[batch_id][edge_id][1]]

                    look_feats = torch.cat((look_source, look_target, look_agg))
                    look_feats = self.look_edge_classifier_1(look_feats)
                    look_feats = self.look_edge_classifier_2(look_feats)
                    look_feats = self.look_edge_classifier_3(look_feats)

                    logits[batch_id][4+total_actions[batch_id][edge_id]] = look_feats

        e_network = time.time()

        # utils.timeit("convs")

        # if self.is_hybrid:
        #     self._features = self._hiddens(torch.cat([self._features, obs], dim=1))
        #     utils.timeit("hybrid section")

        # logits = self._logits(self._features)
        # utils.timeit("get logsits")
        # print(f"logits: {logits[batch_id]}")
        # return
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        utils.timeit("reshape")
        #print(f"gnn: {e_gnn - s_gnn}, edge: {e_edge - s_edge}, network: {e_network - s_network}")
        return logits, state

    @override(TMv2.TorchModelV2)
    def value_function(self):
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
