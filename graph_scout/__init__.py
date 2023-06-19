# from gymnasium.envs.registration import register
from ray.tune.registry import register_env
from .envs.base.env_scout_mission_rllib import ScoutMissionStdRLLib

"""
register_env(
    id="graphScoutMission-v0",
    entry_point="graph_scout.envs.base:ScoutMissionStd",
    max_episode_steps=100,
)
"""

register_env(
    name="graphScoutMission-v0", env_creator=lambda config: ScoutMissionStdRLLib(config)
)

# register(
#     id='graphScoutMission-v1',
#     entry_point='graph_scout.envs.base:ScoutMissionExt',
#     max_episode_steps=100,
# )
