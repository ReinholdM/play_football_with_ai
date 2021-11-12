from .grf_env import BaseGFootBall as base_env, ParameterSharingWrapper
from .encoders import encoder_basic, encoder_highpass, rewarder_basic

default_config = {
    # env building config
    "use_built_in_GK": True,
    "scenario_config": {
        "env_name": "5_vs_5",
        "number_of_left_players_agent_controls": 4,
        "number_of_right_players_agent_controls": 4,
        "representation": "raw",
        "logdir": "",
        "write_goal_dumps": False,
        "write_full_episode_dumps": False,
        "render": False,
        "stacked": False,
    },
}


def env(**kwargs):
    return ParameterSharingWrapper(base_env(**kwargs), lambda x: x[:6])
    # return base_env(**kwargs)
