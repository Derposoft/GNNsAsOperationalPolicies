from collections import deque, defaultdict
import sys
import numpy as np
import heapq
from torchinfo import summary
import ray
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
import time
import torch.nn as nn
import torch
from torch_geometric.nn.aggr import Aggregation
import torch_geometric.nn.aggr as aggr
import torch_geometric.nn.pool as pool
from typing import Any, List
from functools import lru_cache
import os
from typing import Union

from train import create_env_config
from graph_scout.envs.data.terrain_graph import MapInfo as ScoutMapInfo
from graph_scout.envs.base import ScoutMissionStdRLLib
from graph_scout.envs.utils.config import default_configs as scout_config
import sigma_graph.envs.figure8.default_setup as env_setup
from sigma_graph.envs.figure8 import action_lookup
from sigma_graph.data.graph.skirmish_graph import MapInfo as Fig8MapInfo

# constants/helper functions
VALUE_HIDDENS = [175, 175]
NETWORK_SETTINGS = {
    "has_final_layer": True,
    "use_altr_model": False,
    # "use_s2v": False,
}
SUPPRESS_WARNINGS = {
    "embed": False,
    "embed_noshapes": False,
    "decode": False,
    "optimization_none": False,
}
GRAPH_OBS_TOKEN = {
    "embedding_size": 5,  # 7, #10
    "embedding_size_scout": 2,  # 7, #10
    "obs_embed": True,
    "embed_pos": False,
    "embed_dir": True,
    # different types of OPT subproblems to load
    "embed_opt": False,
    "flanking": False,  # does positioning on this node consistute "flanking" the enemy?
    "scout_high_ground": True,
    "scout_high_ground_relevance": True,
}
NODE_EMBED_SIZE = (
    GRAPH_OBS_TOKEN["embedding_size"]
    + (2 if GRAPH_OBS_TOKEN["embed_pos"] else 0)
    + (1 if GRAPH_OBS_TOKEN["embed_dir"] else 0)
    + (4 if GRAPH_OBS_TOKEN["embed_opt"] else 0)
)
SCOUT_NODE_EMBED_SIZE = (
    2
    + (4 if GRAPH_OBS_TOKEN["flanking"] else 0)
    + (1 if GRAPH_OBS_TOKEN["scout_high_ground"] else 0)
    + (1 if GRAPH_OBS_TOKEN["scout_high_ground_relevance"] else 0)
)


def ERROR_MSG(e):
    return f"ERROR READING OBS: {e}"


VIEW_DEGS = {
    "view_1deg_away": None,
    "view_2deg_away": None,
    "view_3deg_away": None,
}
MOVE_DEGS = {
    "move_1deg_away": None,
    "move_2deg_away": None,
    "move_3deg_away": None,
}
scout_map_info = None  # store it out here for lru_cache hashability reasons
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_obs_token(OBS_TOKEN):
    """
    update obs token and update node embedding size accordingly. only run BEFORE
    training (not during or after), otherwise everything breaks since tensor sizes will change.
    """
    global NODE_EMBED_SIZE, SCOUT_NODE_EMBED_SIZE
    GRAPH_OBS_TOKEN.update(OBS_TOKEN)
    NODE_EMBED_SIZE = (
        GRAPH_OBS_TOKEN["embedding_size"]
        + (2 if GRAPH_OBS_TOKEN["embed_pos"] else 0)
        + (1 if GRAPH_OBS_TOKEN["embed_dir"] else 0)
        + (4 if GRAPH_OBS_TOKEN["embed_opt"] else 0)
    )
    SCOUT_NODE_EMBED_SIZE = GRAPH_OBS_TOKEN["embedding_size_scout"] + (
        (
            (1 if GRAPH_OBS_TOKEN["scout_high_ground"] else 0)
            + (1 if GRAPH_OBS_TOKEN["scout_high_ground_relevance"] else 0)
            + (4 if GRAPH_OBS_TOKEN["flanking"] else 0)
        )
        if GRAPH_OBS_TOKEN["embed_opt"]
        else 0
    )
    global verbose
    verbose = OBS_TOKEN["verbose"]
    # print("running set") # sanity check to make sure that these settings are being set


@lru_cache(maxsize=None)
def scout_get_advantage_points(
    threshold: float = 0.75,
    min_expected_advantage_points: int = 5,
):
    """Calculate the advantage that each point has relative to other points. 0.75 threshold
    should lead to 10 advantage points."""
    scout_map = scout_map_info
    vis_map = scout_map.g_view
    advantage_points = set()
    for u in vis_map.adj:
        for v in vis_map.adj[u]:
            edge_uv = vis_map.adj[u][v]
            for position in edge_uv:
                stats_uv = edge_uv[position]
                prob_uv = stats_uv["prob"]
                prob_vu = 0
                if v in vis_map.adj and u in vis_map.adj[v]:
                    edge_vu = vis_map[v][u]
                    for position_vu in edge_vu:
                        stats_vu = edge_vu[position_vu]
                        prob_vu = max(prob_vu, stats_vu["prob"])
                advantage = prob_uv - prob_vu
                if advantage > threshold:
                    advantage_points.add((u, v))
    assert len(advantage_points) > min_expected_advantage_points
    return advantage_points


@lru_cache(maxsize=None)
def scout_get_advantage_embeddings_NEW(
    batch_size: int,
    pos_obs_size: int,
    device=None,
):
    """
    finds the advantageous points by filtering point pairs with a shooting probability
    value greater than the given threshold.
    """

    # TODO take into account v as well as u...?
    advantage_points = scout_get_advantage_points()
    extra_node_embeddings = torch.zeros([batch_size, pos_obs_size, 1])
    for u, v in advantage_points:
        extra_node_embeddings[:, u, -1] = 1
    if device:
        extra_node_embeddings = extra_node_embeddings.to(device)
    return extra_node_embeddings


hgr_embeddings_base_new = None


@lru_cache(maxsize=None)
def scout_get_advantage_relevance_embeddings_NEW(batch_size: int, pos_obs_size: int):
    """
    Calculates P_u(hit)_t = sum_v P(enemy on v | enemy on v' at t-a) P(hit from u to v)
    for all u, given the positions of blue agents.
    """
    # create extra node embeddings to add to
    global hgr_embeddings_base_new
    if (
        hgr_embeddings_base_new == None
        or hgr_embeddings_base_new.shape[0] != batch_size
    ):
        hg_relevance_node_embeddings = torch.zeros([batch_size, pos_obs_size, 1])
        hgr_embeddings_base_new = hg_relevance_node_embeddings
    else:
        hg_relevance_node_embeddings = hgr_embeddings_base_new
    # scout_map_info.g_view


@lru_cache(maxsize=None)
def scout_get_high_ground_embeddings(batch_size: int, pos_obs_size: int, device=None):
    """
    :param batch_size: batch size of input to create embeddings for
    :param pos_obs_size: observation size of positional one-hot encodings (n nodes in scout graph)
    """
    # list of (u, v, [directions]) for advantaged locations from u to v
    high_ground_points = scout_config.init_setup["LOCAL_CONTENT"]["imbalance_pairs"]

    # create extra node embeddings to add to
    extra_node_embeddings = torch.zeros([batch_size, pos_obs_size, 1])
    for u, v, dirs in high_ground_points:
        extra_node_embeddings[:, u, -1] = 1
    if device:
        extra_node_embeddings = extra_node_embeddings.to(device)
    return extra_node_embeddings


# @lru_cache(maxsize=None)
scout_compute_relevance_heuristic_for_waypoint_cache = {}
# scout_compute_relevance_heuristic_for_waypoint_cache = None  # ProcessSafeDict()
cache_miss, cache_calls = 0, 0


def scout_compute_relevance_heuristic_for_waypoint(blue_positions: torch.Tensor):
    """
    computes a simple relevance heuristic for certain waypoints based on how far away the nearest blue agent is.
    :param blue_positions: frozenset of integers corresponding to the locations where the blue agents are
    :returns: map from waypoint to relevance score
    """
    # global cache_calls, cache_miss
    # cache_calls += 1
    if not blue_positions:
        return {}
    global scout_compute_relevance_heuristic_for_waypoint_cache
    # if not scout_compute_relevance_heuristic_for_waypoint_cache:
    #    scout_compute_relevance_heuristic_for_waypoint_cache = ProcessSafeDict()

    blue_pos_string = str(blue_positions)
    if blue_pos_string in scout_compute_relevance_heuristic_for_waypoint_cache:
        return scout_compute_relevance_heuristic_for_waypoint_cache[blue_pos_string]
    # cache_miss += 1

    def relevance_score_from_dist(dist):
        return min(1 / (dist + 1e-5), 1)  # clipping

    relevances = {}
    # list of (u, v, [directions]) for advantaged locations from u to v
    # high_ground_points = scout_config.init_setup["LOCAL_CONTENT"]["imbalance_pairs"]
    high_ground_points = scout_get_advantage_points()
    blue_locations = list(blue_positions.cpu().numpy())
    for u, v in high_ground_points:
        # perform a search from high ground point's advantage point pair to the closest blue
        # print(v)
        # print(blue_locations)
        dist = flank_optimization_scout(
            scout_map_info, v, blue_locations, compute_dist_only=True
        )
        relevances[u] = max(relevances.get(u, 0), relevance_score_from_dist(dist))
    scout_compute_relevance_heuristic_for_waypoint_cache[blue_pos_string] = relevances
    return relevances


@lru_cache(maxsize=None)
def get_blue_positions(x: torch.Tensor):
    blue_positions = (x == 1).nonzero()
    if len(blue_positions) == 0:
        return None
    return blue_positions[0]


hgr_embeddings_base = None


# 1.6s with no cache, ~160-180ms with compute_relevance_heuristics cache, ~20ms with optimized get_blue_positions
# update: 2ms with a cache that actually works. note to self: don't use lru_cache for tensors.
scout_get_high_ground_embeddings_relevance_cache = {}


def scout_get_high_ground_embeddings_relevance(
    obs: torch.Tensor, model_map: ScoutMapInfo = None, device=None
) -> torch.Tensor:
    global scout_get_high_ground_embeddings_relevance_cache
    if str(obs) in scout_get_high_ground_embeddings_relevance_cache and len(
        scout_get_high_ground_embeddings_relevance_cache[str(obs)]
    ) == len(obs):
        if not device or scout_get_high_ground_embeddings_relevance_cache[
            str(obs)
        ].device == torch.device(device):
            check_device(
                scout_get_high_ground_embeddings_relevance_cache[str(obs)],
                "HGR_cache_output",
            )
            return scout_get_high_ground_embeddings_relevance_cache[str(obs)]
    global scout_map_info
    if not scout_map_info:
        scout_map_info = model_map
    batch_size = obs.shape[0]
    pos_obs_size = obs.shape[1] // 2

    # create extra node embeddings to add to
    global hgr_embeddings_base
    if hgr_embeddings_base == None or hgr_embeddings_base.shape[0] != batch_size:
        hg_relevance_node_embeddings = torch.zeros([batch_size, pos_obs_size, 1])
        hgr_embeddings_base = hg_relevance_node_embeddings
    else:
        hg_relevance_node_embeddings = hgr_embeddings_base
    # timeit("creating embeddings")

    for i in range(len(obs)):
        x = obs[i]
        blue_positions = get_blue_positions(x[pos_obs_size : 2 * pos_obs_size])
        relevance_scores = scout_compute_relevance_heuristic_for_waypoint(
            blue_positions  # frozenset(blue_positions) # note to self: don't do any weird transformations like this. too slow.
        )
        for u in relevance_scores:
            hg_relevance_node_embeddings[i, u, 0] = 1
    # timeit("updating embeddings")
    # print(f"{cache_miss} miss, {cache_calls} calls")
    if device:
        hg_relevance_node_embeddings = hg_relevance_node_embeddings.to(device)
    check_device(
        hg_relevance_node_embeddings,
        "HGR_UNcached_output",
    )
    scout_get_high_ground_embeddings_relevance_cache[
        str(obs)
    ] = hg_relevance_node_embeddings
    return hg_relevance_node_embeddings


def scout_embed_obs_in_map(obs: torch.Tensor, map: ScoutMapInfo, device=None):
    """
    :param obs: observation from combat_env gym
    :param map: MapInfo object from inside of combat_env gym (for graph connectivity info)
    :returns: node embeddings for each node of the move graph in map, using obs
    """
    global SUPPRESS_WARNINGS
    pos_obs_size = map.get_graph_size()
    batch_size = len(obs)
    node_embeddings = torch.stack(
        [obs[:, :pos_obs_size], obs[:, pos_obs_size:]], dim=-1
    )
    # assert obs[0, pos_obs_size:].equal(node_embeddings[0, :, 1])  # sanity check to make sure reshape is working as expected. uncomment this if things are exploding.

    # save some stuff globally for other functions
    global scout_map_info
    scout_map_info = map

    # embed graph subproblem-esque optimization into node embeddings
    if GRAPH_OBS_TOKEN["embed_opt"]:
        if GRAPH_OBS_TOKEN["scout_high_ground"]:
            hg_node_embeddings = scout_get_advantage_embeddings_NEW(
                batch_size, pos_obs_size, device
            )
            # hg_node_embeddings = scout_get_high_ground_embeddings(
            #    batch_size, pos_obs_size, device
            # )
            node_embeddings = torch.cat([node_embeddings, hg_node_embeddings], dim=-1)
            timeit("high ground embedding")
        if GRAPH_OBS_TOKEN["scout_high_ground_relevance"]:
            hg_relevance_node_embeddings = scout_get_high_ground_embeddings_relevance(
                obs, device=device
            )
            node_embeddings = torch.cat(
                [node_embeddings, hg_relevance_node_embeddings], dim=-1
            )
            timeit("high ground rel")
        if GRAPH_OBS_TOKEN["flanking"]:
            extra_node_embeddings_flanking = torch.zeros([batch_size, pos_obs_size, 4])
            for i in range(len(obs)):
                x = obs[i]
                red_position = get_loc(x, pos_obs_size)
                blue_positions = set([])
                for j in range(pos_obs_size):
                    if x[pos_obs_size : 2 * pos_obs_size][j]:
                        blue_positions.add(j)
                opt = flank_optimization_scout(
                    map,
                    red_position,
                    blue_positions,
                )
                if opt != 0:
                    extra_node_embeddings_flanking[i, red_position, opt - 1] = 1
            node_embeddings = torch.cat(
                [node_embeddings, extra_node_embeddings_flanking], dim=-1
            )
    assert node_embeddings.shape[-1] == SCOUT_NODE_EMBED_SIZE  # sanity check
    return node_embeddings


# TODO read obs using obs_token instead of hardcoding.
#      figure8_squad.py:_update():line ~250
def efficient_embed_obs_in_map(obs: torch.Tensor, map: Fig8MapInfo, obs_shapes=None):
    """
    :param obs: observation from combat_env gym
    :param map: MapInfo object from inside of combat_env gym (for graph connectivity info)
    :param obs_shapes: info used to partition obs into self/blue team/red team observations
    :return node embeddings for each node of the move graph in map, using obs.
        GRAPH_EMBEDDING=TRUE must be true in the default_env setup for the combat_env.
        the new graph embedding looks as follows:
        [[
            agent_x  (if GRAPH_OBS_TOKEN["embed_pos"] = True),
            agent_y  (if GRAPH_OBS_TOKEN["embed_pos"] = True),
            agent_is_here,
            num_red_here,
            num_blue_here,
            can_red_go_here_t,
            can_blue_see_here_t,
            external_optimization (if GRAPH_OBS_TOKEN["embed_opt"] = True)
        ],
        [...],
        ...]
    """
    # initialize node embeddings tensor
    global SUPPRESS_WARNINGS
    pos_obs_size = map.get_graph_size()
    batch_size = len(obs)
    node_embeddings = torch.zeros(
        batch_size, pos_obs_size, NODE_EMBED_SIZE
    )  # TODO changed +1 node for a dummy node that we'll use when needed
    move_map = create_move_map(map.g_acs)

    # embed x,y
    if GRAPH_OBS_TOKEN["embed_pos"]:
        pos_emb = torch.zeros(pos_obs_size + 1, 2)  # x,y coordinates
        for i in range(pos_obs_size):
            pos_emb[i, :] = torch.FloatTensor(map.n_info[i + 1])
        # normalize x,y
        min_x, min_y = torch.min(pos_emb[:, 0]), torch.min(pos_emb[:, 1])
        pos_emb[:, 0] -= min_x
        pos_emb[:, 1] -= min_y
        node_embeddings /= torch.max(pos_emb)
        node_embeddings[:, :, -3:-1] += pos_emb

    # embed rest of features
    for i in range(batch_size):
        # these default behaviors are required because rllib provides zerod test inputs
        if not obs_shapes:
            print("shapes not provided. returning")
            if not SUPPRESS_WARNINGS["embed"]:
                print(
                    ERROR_MSG(
                        "shapes not provided. skipping embed and suppressing this warning."
                    )
                )
                SUPPRESS_WARNINGS["embed_noshapes"] = True
        self_shape, blue_shape, red_shape, n_red, n_blue = obs_shapes
        if (
            self_shape < pos_obs_size
            or red_shape < pos_obs_size
            or blue_shape < pos_obs_size
        ):
            if not SUPPRESS_WARNINGS["embed"]:
                print(
                    ERROR_MSG(
                        "test batch detected while embedding. skipping embed and suppressing this warning."
                    )
                )
                SUPPRESS_WARNINGS["embed"] = True
            return

        # get obs shapes and parse obs
        self_obs = obs[i][:self_shape]
        blue_obs = obs[i][self_shape : (self_shape + blue_shape)]
        red_obs = obs[i][
            (self_shape + blue_shape) : (self_shape + blue_shape + red_shape)
        ]

        # agent_is_here
        red_node = get_loc(self_obs, pos_obs_size)
        if red_node == -1:
            print(ERROR_MSG("agent not found"))
        node_embeddings[i][red_node][0] = 1

        # num_red_here
        for j in range(pos_obs_size):
            if red_obs[j]:
                node_embeddings[i][j][1] += 1

        # num_blue_here
        blue_positions = set([])
        for j in range(pos_obs_size):
            if blue_obs[j]:
                node_embeddings[i][j][2] += 1
                blue_positions.add(j)

        ## EXTRA EMBEDDINGS TO PROMOTE LEARNING ##
        # can_red_go_here_t
        for possible_next in map.g_acs.adj[red_node + 1]:
            node_embeddings[i][possible_next - 1][3] = 1

        # can_blue_move_here_t
        if MOVE_DEGS["move_1deg_away"] == None:
            MOVE_DEGS["move_1deg_away"] = get_nodes_ndeg_away(map.g_acs.adj, 1)
        move_1deg_away = MOVE_DEGS["move_1deg_away"]
        for j in blue_positions:
            for possible_next in move_1deg_away[j + 1]:
                node_embeddings[i][possible_next - 1][4] = 1

        # direction of blue agent
        if GRAPH_OBS_TOKEN["embed_dir"]:
            blue_i = 0
            for blue_position in blue_positions:  # HERE
                start_idx_for_dir_i = len(blue_obs) - 1 - 4 * (blue_i + 1)
                end_idx_for_dir_i = len(blue_obs) - 1 - 4 * blue_i
                dir_i = blue_obs[start_idx_for_dir_i:end_idx_for_dir_i]
                blue_dir = (
                    get_loc(dir_i, 4) + 1
                )  # direction as defined by action_lookup.py
                blue_dir_behind = action_lookup.TURN_L[
                    action_lookup.TURN_L[blue_dir]
                ]  # 2 left turns = 180deg turn
                blue_dir_behind_node_idx = (
                    move_map[blue_position + 1][blue_dir_behind] - 1
                )
                if blue_dir_behind_node_idx >= 0:
                    node_embeddings[i][blue_dir_behind_node_idx][5] = 1
                blue_i += 1

        # add feature from some external "optimization", if desired
        if GRAPH_OBS_TOKEN["embed_opt"]:
            if GRAPH_OBS_TOKEN["flanking"]:  # use the "flanking" optimization
                node_embeddings[i][red_node][6:10] = flank_optimization(
                    map, red_node, blue_positions
                )
            else:  # no optimization given; defaulting to 0
                if not SUPPRESS_WARNINGS["optimization_none"]:
                    print(
                        ERROR_MSG(
                            "external optimization not provided. embedding \
                        will contain an extra unused 0 and decrease efficiency. \
                        did you mean to set GRAPH_OBS_TOKEN.embed_opt = False?"
                        )
                    )
                    SUPPRESS_WARNINGS["optimization_none"] = True

    return node_embeddings  # .to(device)


def get_loc(one_hot_graph: torch.Tensor, graph_size, default=0, get_all=False):
    """
    get location of an agent given one-hot positional encoding on graph (0-indexed)
    """
    global SUPPRESS_WARNINGS
    one_hot_locations = (one_hot_graph[:graph_size] == 1).nonzero()
    if len(one_hot_locations) > 1:
        if get_all:
            return one_hot_locations[0]
        return one_hot_locations[0][0]
    if not SUPPRESS_WARNINGS["decode"]:
        print(
            f"test batch detected while decoding. agent not found. returning default={default} and suppressing this warning."
        )
        SUPPRESS_WARNINGS["decode"] = True
    return default


def create_move_map(graph):  #: Fig8MapInfo):
    """
    turns map.g_acs into a dictionary in the form of:
    {
        start_node: {
            direction: next_node,
            ...
        },
        ...
    }
    """
    move_map = (
        {}
    )  # movement dictionary: d[node][direction] = newnode. newnode is -1 if direction is not possible from node
    for n in graph.adj:
        move_map[n] = {}
        ms = graph.adj[n]
        for m in ms:
            dir = ms[m]["action"]
            move_map[n][dir] = m
        for movement in action_lookup.MOVE_LOOKUP:
            if movement not in move_map[n]:
                move_map[n][movement] = -1
    return move_map


def get_nodes_ndeg_away(graph, n):
    """
    :param graph: networkx.adj adjacency dictionary
    :param n: number of degrees to search outwards
    :return dictionary mapping each node to a list of nodes n degrees away
    """
    result = {}
    for node in graph:
        result[node] = get_nodes_ndeg_from_s(graph, node, n)
    return result


def get_nodes_ndeg_from_s(graph, s, n):
    """
    collects the list of all nodes that are n degrees away from a source s on the given graph.
    :param graph: networkx.adj adjacency dictionary
    :param s: 1-indexed node starting location
    :param n: number of degrees to search outwards
    :return list of nodes that are n (or fewer) degrees from s
    """
    # run n iterations of bfs and collect node results
    visited = set([s])
    dq = deque([s])
    for i in range(n):
        if not dq:
            break
        node = dq.popleft()
        next_nodes = graph[node]
        for next_node in next_nodes:
            if next_node not in visited:
                visited.add(next_node)
                dq.append(next_node)
    return list(visited)


def load_edge_dictionary(map_edges):
    """
    :param map_edges: edges from a graph from MapInfo. input should be 1-indexed MapInfo map_edge dictionary.
    :return the 0-indexed edge dictionary for quick lookups.

    load edge dictionary from a map (0-indexed)
    """
    # create initial edge_array and TODO edge_to_action mappings
    edge_array = []
    for k, v in zip(map_edges.keys(), map_edges.values()):
        edge_array += [[k - 1, vi - 1] for vi in v.keys()]

    # create edge_dictionary
    edge_dictionary = {}
    for edge in edge_array:
        if edge[0] not in edge_dictionary:
            edge_dictionary[edge[0]] = set([])
        edge_dictionary[edge[0]].add(edge[1])

    return edge_dictionary


def get_cost_from_reward(reward):
    return 1 / (reward + 1e-3)  # takes care of div by 0


def get_probs_mask(agent_nodes, graph_size, edges_dict):
    node_exclude_list = np.array(list(range(graph_size)))
    mask = [
        np.delete(node_exclude_list, list(edges_dict[agent_node]) + [agent_node])
        for agent_node in agent_nodes
    ]
    return mask


def count_model_params(model, print_model=False):
    num_params = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    if print_model:
        summary(model)
    print(f"{type(model)} using {num_params} #params")


def parse_config(model_config):
    hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
        model_config.get("post_fcnet_hiddens", [])
    )
    activation = model_config.get("fcnet_activation")
    if not model_config.get("fcnet_hiddens", []):
        activation = model_config.get("post_fcnet_activation")
    no_final_linear = model_config.get("no_final_linear")
    vf_share_layers = model_config.get("vf_share_layers")  # this is usually 0
    free_log_std = model_config.get("free_log_std")  # skip worrying about log std
    return hiddens, activation, no_final_linear, vf_share_layers, free_log_std


def create_value_branch(
    num_inputs,
    num_outputs,
    *,
    vf_share_layers=False,
    activation="relu",
    hiddens=[],
    is_actor_critic=False,
):
    if not is_actor_critic:
        return None, None
    _value_branch_separate = None
    # create value network with equal number of hidden layers as policy net
    if not vf_share_layers:
        prev_vf_layer_size = num_inputs
        vf_layers = []
        for size in hiddens:
            vf_layers.append(
                SlimFC(
                    in_size=prev_vf_layer_size,
                    out_size=size,
                    activation_fn=activation,
                    initializer=normc_initializer(1.0),
                )
            )
            prev_vf_layer_size = size
        _value_branch_separate = torch.nn.Sequential(*vf_layers)
    # layer which outputs 1 value
    # prev_layer_size = hiddens[-1] if self._value_branch_separate else self.map.get_graph_size()
    prev_layer_size = hiddens[-1] if _value_branch_separate else num_outputs
    _value_branch = SlimFC(
        in_size=prev_layer_size,
        out_size=1,
        initializer=normc_initializer(0.01),
        activation_fn=None,
    )
    return _value_branch, _value_branch_separate


def create_policy_fc(
    hiddens,
    activation,
    num_outputs,
    no_final_linear,
    num_inputs,
):
    layers = []
    prev_layer_size = num_inputs
    _logits = None
    # Create layers 0 to second-last.
    for size in hiddens[:-1]:
        layers.append(
            SlimFC(
                in_size=prev_layer_size,
                out_size=size,
                initializer=normc_initializer(1.0),
                activation_fn=activation,
            )
        )
        prev_layer_size = size

    # The last layer is adjusted to be of size num_outputs, but it's a
    # layer with activation.
    if no_final_linear and num_outputs:
        layers.append(
            SlimFC(
                in_size=prev_layer_size,
                out_size=num_outputs,
                initializer=normc_initializer(1.0),
                activation_fn=activation,
            )
        )
        prev_layer_size = num_outputs
    # Finish the layers with the provided sizes (`hiddens`), plus -
    # iff num_outputs > 0 - a last linear layer of size num_outputs.
    else:
        if len(hiddens) > 0:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=hiddens[-1],
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_layer_size = hiddens[-1]
        if num_outputs:
            _logits = SlimFC(
                in_size=prev_layer_size,
                out_size=num_outputs,
                initializer=normc_initializer(0.01),
                activation_fn=None,
            )
    _hidden_layers = nn.Sequential(*layers)
    return _hidden_layers, _logits


def flank_optimization_scout(
    map: ScoutMapInfo, red_location: int, blue_locations: List[int], **kwargs
):
    # add these 2 to interface it with the old function
    map.g_acs = map.g_move
    map.g_vis = map.g_view
    return flank_optimization(map, red_location, blue_locations, **kwargs)


def flank_optimization(
    map: Fig8MapInfo,
    red_location: int,
    blue_locations: List[int],
    compute_dist_only=False,
    **kwargs,
):
    """
    :param map: MapInfo object with vis and move information
    :param red_location: 0-indexed location of red agent node
    :param blue_locations: 0-indexed locations of blue agent nodes
    :param compute_dist_only: whether or not to return a distance from red location to nearest blue location, ignoring any heuristic
    :return a direction that the agent has to go to get behind the enemy agent, 1-hot encoded
    """
    red_location += 1
    blue_locations = [x + 1 for x in blue_locations]

    """
    get information that is required for A* to run efficiently:
    movement map, visuals map, and nodes visible by some blue agent
    """
    # build move graph: node -> {next_node: dir_to_next_node, ...}
    move_map = {}
    for n in map.g_acs.adj:
        move_map[n] = {}
        ms = map.g_acs.adj[n]
        for m in ms:
            dir = ms[m]["action"]
            move_map[n][m] = dir

    # build vis graph: node -> {engage_set} = {x in nodes | dist(x, node) < engage_range}
    vis_map = {}
    for node in map.g_vis.adj:
        vis_map[node] = [
            x
            for x in map.g_vis.adj[node]
            if map.g_vis.adj[node][x][0]["dist"]
            < env_setup.INTERACT_LOOKUP["engage_range"]
        ]

    # set of nodes visible by some blue agent
    blue_visible_nodes = set()
    for blue_location in blue_locations:
        for visible_location in vis_map[blue_location]:
            blue_visible_nodes.add(visible_location)

    """
    run A* from each red agent to the closest node which can view blue agent while trying to minimize
    passing through nodes on which blue can engage. implement using priority queue sorting on H.
    PQ entries are of the form: pq(n) = (H(n), V(n), D(n), n, last_n)

    V == # nodes visible to enemy & within engagement distance
    D == degree away from start
    A* heuristic: H(n) = |G|*V(n) + D(n)
    """
    directions = {red_location: -1}  # node -> where we got to this node from
    nodes = [(0, 0, 0, red_location, -1)]
    visited = set([red_location])
    goal_node = -1
    while nodes:
        H, V, D, n, last_n = heapq.heappop(nodes)
        # take note of how we arrived here
        if n not in visited:
            directions[n] = last_n
            visited.add(n)
        elif n != red_location:
            continue
        # check to see if our search is done
        if compute_dist_only:
            if n in blue_locations:
                return D
        else:
            for blue_location in blue_locations:
                if blue_location in vis_map[n]:
                    goal_node = n
                    break
        # continue our search if not done
        next_nodes = move_map[n]
        for next_node in next_nodes:
            if next_node not in visited:
                is_visible_by_enemy = 1 if next_node in blue_visible_nodes else 0
                next_V = (V + is_visible_by_enemy) if not compute_dist_only else 0
                next_D = D + 1
                nodes.append(
                    (
                        map.g_acs.number_of_nodes() * next_V + next_D,
                        next_V,
                        next_D,
                        next_node,
                        n,
                    )
                )

    """
    backtrack back from goal_node->red_location using directions map
    """
    if goal_node == -1:
        return 0  # no direction if no nodes could be found from where we can see blue
    last_node = goal_node
    curr_node = goal_node
    while curr_node != red_location:
        last_node = curr_node
        curr_node = directions[curr_node]
    direction = move_map[red_location][last_node]
    return direction


class LocalPooling(nn.Module):
    def __init__(self):
        Aggregation.__init__(self)

    def forward(self, x, edge_index, agent_nodes=None):
        if agent_nodes == None:
            print("agent node not provided to local aggregation")
            sys.exit()
        if not len(agent_nodes):
            return x[:, 0, :]
        # print(agent_nodes)
        # print(x)
        # print(x[agent_nodes])
        # x = x[range(len(x)), agent_nodes, :]
        # print("AGENT NODES", agent_nodes)
        x = x[agent_nodes[:, 0], agent_nodes[:, 1]]
        # print(x)

        return x


class GeneralGNNPooling(nn.Module):
    def __init__(
        self,
        aggregator_name: str,
        input_dim: int,
        output_dim: int,
    ):
        Aggregation.__init__(self)
        self.aggregator = None
        self.aggregator_name = aggregator_name
        if self.aggregator_name == "attention":
            # self.aggregator = pool.SAGPooling(input_dim)
            self.aggregator = aggr.AttentionalAggregation(
                gate_nn=nn.Sequential(
                    SlimFC(input_dim, input_dim),
                    SlimFC(input_dim, 1),
                )
            )
            raise NotImplementedError("attention aggregation not yet implemented")
        elif self.aggregator_name == "global" or self.aggregator_name == "mean":
            self.aggregator = aggr.MeanAggregation()
        elif self.aggregator_name == "local" or self.aggregator_name == "agent_node":
            self.aggregator = LocalPooling()
        elif self.aggregator_name == "hybrid":
            self.aggregator1 = LocalPooling()
            self.aggregator2 = aggr.MeanAggregation()
            input_dim *= 2
        else:
            raise NotImplementedError(
                "aggregation_fn/aggregator_name is not one of the supported aggregations."
            )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reducer = nn.Sequential(
            SlimFC(input_dim, input_dim, activation_fn="relu"),
            SlimFC(input_dim, output_dim, activation_fn="relu"),
        )

    @override(nn.Module)
    def forward(self, x, edge_index, agent_nodes=None):
        if self.aggregator_name == "attention":
            print(x.shape, self.aggregator_name, "AGGREGATOR NAME AND INPUT SHAPE")
            x = torch.concat(
                [self.aggregator(_x, edge_index) for _x in x],
                dim=1,
            )
            print(x.shape, "AFTER AGGREGATION")
        elif self.aggregator_name == "global" or self.aggregator_name == "mean":
            x = self.aggregator(x).reshape([x.shape[0], -1])
        elif self.aggregator_name == "local" or self.aggregator_name == "agent_node":
            x = self.aggregator(x, edge_index, agent_nodes=agent_nodes)
        elif self.aggregator_name == "hybrid":
            x = torch.concat(
                [
                    self.aggregator1(x, edge_index, agent_nodes=agent_nodes),
                    self.aggregator2(x).reshape([x.shape[0], -1]),
                ],
                dim=1,
            )
        else:
            raise NotImplementedError(
                "aggregation_fn/aggregator_name is not one of the supported aggregations."
            )
        x = self.reducer(x)
        return x  # self.softmax(x)


prev_time = time.time()
verbose = os.environ.get("verbose", False)


def timeit(msg: str, n_digits: int = 4):
    if verbose:
        global prev_time
        print(f"{time.time()-prev_time:2.{n_digits}f} {msg}")
        prev_time = time.time()


def check_device(module_or_tensor: Union[nn.Module, torch.Tensor], name=""):
    if verbose:
        if isinstance(module_or_tensor, nn.Module):
            print(
                f"{type(module_or_tensor)}:{name} seems to be on {next(module_or_tensor.parameters()).device}"
            )
        else:
            print(f"{type(module_or_tensor)}:{name} is on {module_or_tensor.device}")
