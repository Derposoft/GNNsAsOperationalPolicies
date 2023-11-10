from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.multi_agent_env_compatibility import (
    MultiAgentEnvCompatibility,
)
import sys

# from ray.rllib.models.catalog import MODEL_DEFAULTS

# from ray.rllib.agents import dqn
# import numpy as np
# import os
# import time

# import sys
from .env_scout_mission_std import ScoutMissionStd

# from . import default_setup as env_setup
# local_action_move = env_setup.act.MOVE_LOOKUP
# local_action_turn = env_setup.act.TURN_90_LOOKUP
from . import action_lookup as env_setup

local_action_move = env_setup.MOVE_LOOKUP
local_action_turn = env_setup.TURN_3_LOOKUP


# a variant of figure8_squad that can be used by rllib multiagent setups
# reference: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py
class ScoutMissionStdRLLib(ScoutMissionStd, MultiAgentEnv):
    def __init__(self, config=None):
        config = config or {}
        super().__init__(**config)

        # extra values to make graph embedding viable
        if not hasattr(self, "_agent_ids"):
            self._agent_ids = self.states.dump_dict()[0].keys()
        if not hasattr(self, "_obs_space_in_preferred_format"):
            self._obs_space_in_preferred_format = None
        if not hasattr(self, "_action_space_in_preferred_format"):
            self._action_space_in_preferred_format = None
        self.action_space = self.action_space[0]
        self.observation_space = self.observation_space[0]
        self.done = set()
        super(MultiAgentEnv, self).__init__()

    # return an arbitrary encoding from the "flat" action space to the normal action space 0-indexed
    def convert_discrete_action_to_multidiscrete(self, action):
        return [action % len(local_action_move), action // len(local_action_move)]

    def convert_multidiscrete_action_to_discrete(move_action, turn_action):
        return turn_action * len(local_action_move) + move_action

    def reset(self, *, seed: int = 0, options=None):
        """
        :returns: dictionary of agent_id -> reset observation
        """
        super().reset()
        self.done = set()
        return self.states.dump_dict()[0], {}

    def step(self, _n_actions: dict):
        n_actions = []
        for _id in self.states.name_list:
            if _id in _n_actions:
                n_actions.append(_n_actions[_id])
            else:
                n_actions.append(self.action_space.sample())
        super().step(n_actions)

        obs, rew, done = self.states.dump_dict(
            step=self.step_counter, provide_totals=True
        )
        truncateds = {}
        all_done = True
        for k in done:
            if done[k]:
                self.done.add(k)
            all_done = all_done and done[k]
        print(done)
        done["__all__"] = all_done
        truncateds["__all__"] = all_done

        # alter the obs for each individual agent so there's a higher weight at its location
        for _id in range(self.states.num):
            _key = self.states.name_list[_id]
            # print("pre-transformation obs", obs[_key])
            _current_agent_node_idx = self.agents.gid[_id].at_node - 1
            # print("agent node:", _current_agent_node_idx)

            # /2 and +0.5 at the proper agent node index ensures all values stay within 0 and 1
            obs[_key][: self.map.get_graph_size()] /= 2
            obs[_key][_current_agent_node_idx] += 0.5
            # print("post-transformation obs", obs[_key])
        # sys.exit()

        # make sure to only report done ids once
        for id in self.done:
            done.pop(id)
        return obs, rew, done, truncateds, {}  # last 2 are "truncateds" and "infos"
