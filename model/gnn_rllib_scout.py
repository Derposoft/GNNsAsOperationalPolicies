"""
base class from https://github.com/ray-project/ray/blob/master/rllib/models/torch/fcnet.py
"""
import torch.nn as nn
import torch
import ray.rllib.models.torch.torch_modelv2 as TMv2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
import gymnasium as gym
from torch_geometric.nn.conv import GATv2Conv, GCNConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.data import Batch, Data as GraphData
import numpy as np
import time

from graph_scout.envs.data.terrain_graph import MapInfo
from graph_scout.envs.utils.config import default_configs as env_setup
import model.utils as utils


class GNNScoutPolicy(TMv2.TorchModelV2, nn.Module):
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
        self.is_hybrid = kwargs["is_hybrid"]  # is this a hybrid model or gat-only?
        self.conv_type = kwargs["conv_type"]
        self.layernorm = kwargs["layernorm"]
        self.vis_enabled = kwargs.get("vis_enabled", False)
        self._features = None  # current "base" output before logits
        self._last_flat_in = None  # last input
        self.action_space_output_dim = np.sum(action_space.nvec)

        ###### HYPERPARAMETERS
        gcn_num_layers = 2
        gcn_hidden_dim = 5

        gat_num_layers = 2
        gat_hidden_dim = 5
        gat_num_heads = 4
        #########################
        print(f"USING CONV TYPE: {self.conv_type}")
        self.GAT_LAYERS = (
            gcn_num_layers if self.conv_type == "gcn" else gat_num_layers
        )  # 1_6:1804, 3_50: ~15k

        self.N_HEADS = 1 if self.conv_type == "gcn" else gat_num_heads
        self.HIDDEN_DIM = gcn_hidden_dim if self.conv_type == "gcn" else gat_hidden_dim
        self.hiddens = [self.hidden_size, self.hidden_size // 2]
        gnn = GATv2Conv if self.conv_type == "gat" else GCNConv

        # move graph instantiation
        self.adjacency = []
        for n in map.g_move.adj:
            ms = map.g_move.adj[n]
            for m in ms:
                self.adjacency.append([n - 1, m - 1])
        self.adjacency = torch.LongTensor(self.adjacency).t().contiguous()

        # vis graph instantiation TODO
        if self.vis_enabled:
            self.vis_adjacency = []
            for n in map.g_view.adj:
                ms = map.g_view.adj[n]
                for m in ms:
                    self.vis_adjacency.append([n - 1, m - 1])
            self.vis_adjacency = torch.LongTensor(self.vis_adjacency).t().contiguous()

        """
        instantiate policy and value networks
        """
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
        # move graph gnns
        self.gnns = nn.ModuleList(
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

        # vis graph gnns TODO
        if self.vis_enabled:
            self.vis_gnns = nn.ModuleList(
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

        self.norms = nn.ModuleList(
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

        """
        produce debug output and ensure that model is on right device
        """
        utils.count_model_params(self, print_model=True)

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
        utils.timeit(f"obs to float -- {len(obs)}")
        utils.check_device(self, "POLICY_MODEL")
        utils.check_device(obs, "obs")
        current_device = next(self.parameters()).device
        x = utils.scout_embed_obs_in_map(obs, self.map, current_device)
        batch_size = x.shape[0]
        graph_size = x.shape[1]
        utils.timeit("scout embed obs")

        # agent_nodes = [utils.get_loc(gx, self.map.get_graph_size()) for gx in obs]
        # utils.timeit("get agent nodes")

        agent_nodes_new = (obs[:, : self.map.get_graph_size()] == 2).nonzero()
        assert len(obs) == len(agent_nodes_new) or len(agent_nodes_new) == 0
        utils.timeit("get agent nodes new")

        # inference
        self.adjacency = self.adjacency.to(current_device)
        if self.conv_type == "gat":
            graph = Batch.from_data_list([GraphData(_x, self.adjacency) for _x in x])
            x, batch_adjacency = graph.x, graph.edge_index
            utils.timeit("batching")

        for conv, norm in zip(self.gnns, self.norms):
            utils.check_device(conv, "convs")
            utils.check_device(x, "x_for_convs")
            if self.conv_type == "gcn":
                x = conv(x, self.adjacency)
            else:
                utils.check_device(batch_adjacency, "adj_for_convs")
                x = conv(x, batch_adjacency)  # batching=8.9ms, convs=26.3ms
                # x = torch.stack([conv(_x, self.adjacency) for _x in x], dim=0) # 180-190ms

            if self.layernorm:
                x = norm(x)
        utils.timeit("gnn convs")

        x = x.reshape([batch_size, graph_size, -1])
        self._features = self.aggregator(x, self.adjacency, agent_nodes=agent_nodes_new)
        utils.timeit("convs")

        utils.check_device(self._features, "features")
        utils.check_device(self._hiddens, "hiddens")
        if self.is_hybrid:
            self._features = self._hiddens(torch.cat([self._features, obs], dim=1))
            utils.timeit("hybrid section")

        logits = self._logits(self._features)
        utils.timeit("get logits")
        utils.check_device(logits, "logits")
        utils.check_device(self._logits, "_logits_layer")

        # return
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        utils.timeit("reshape")
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
