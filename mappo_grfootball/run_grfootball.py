# -*- coding: utf-8 -*-
"""
run_grfootball.py

@Organization:  Institue of Automation, CAS
@Author: Linghui Meng
@Time: 4/22/21 5:28 PM
@Function:
"""

import argparse

import yaml
import os

from rollout_function import grf_simultaneous
from malib.runner import run
import copy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from malib.envs.gr_football import default_config, env as env_creator


def build_env_desc(env_desc):
    print(f">>>>>>>>>>>> Building {env_desc['id']} <<<<<<<<<<<<<")
    env_desc["config"] = env_desc.get("config", {})
    env_config = copy.deepcopy(default_config)
    env_config.update(env_desc["config"])
    env_desc["config"] = env_config

    print(env_desc["config"])
    # env_desc["config"]["logdir"] = "/tmp/grfootball"

    env_desc["creator"] = env_creator
    env = env_creator(**env_desc["config"])
    # env = env_creator(**env_desc["config"])

    possible_agents = env.possible_agents
    print("possible agents:", possible_agents)
    observation_spaces = env.observation_spaces
    action_spaces = env.action_spaces

    print(f"---------- observation spaces: {observation_spaces}")
    print(f"----------- action spaces: {action_spaces}")

    env_desc["possible_agents"] = env.possible_agents
    return env_desc


if __name__ == "__main__":
    print("=====================================================")
    print(">>>>>>>>>>>>>> SINGLE POPULATION MODE <<<<<<<<<<<<<<<")
    print("=====================================================")
    parser = argparse.ArgumentParser("General training on Google Research Football.")
    parser.add_argument(
        "--config", type=str, help="YAML configuration path.", required=True
    )

    args = parser.parse_args()

    with open(os.path.join(BASE_DIR, args.config), "r") as f:
        config = yaml.load(f)

    # env.close()

    training_config = config["training"]
    rollout_config = config["rollout"]
    rollout_config["callback"] = grf_simultaneous
    evaluation_config = config["evaluation"]
    evaluation_config["callback"] = grf_simultaneous
    env_desc = build_env_desc(config["env_description"])
    
    env = env_desc["creator"](**env_desc["config"])
    env_desc["possible_agents"] = env.possible_agents[:1]
    training_config["interface"]["observation_spaces"] = env.observation_spaces
    training_config["interface"]["action_spaces"] = env.action_spaces

    custom_config = config["algorithms"]["MAPPO"]["custom_config"]
    # FOR PPO
    # custom_config.update({"global_state_space": env.observation_spaces['team_0']})
    # FOR MAPPO
    custom_config.update({"global_state_space": env.state_space})

    run(
        group=config["group"],
        name=config["name"],
        env_description=env_desc,
        benchmark_env_description=build_env_desc(config["benchmark_env_description"]),
        agent_mapping_func=lambda agent: agent[
            :6
        ],  # e.g. "team_0_player_0" -> "team_0"
        training=training_config,
        algorithms=config["algorithms"],
        # rollout configuration for each learned policy model
        rollout=rollout_config,
        evaluation=config.get("evaluation", {}),
        global_evaluator=config["global_evaluator"],
        dataset_config=config.get("dataset_config", {}),
        parameter_server=config.get("parameter_server", {}),
        # "nash", "fictitious_self_play"
        solver="nash",
        # team_config=env.possible_players,
        worker_config=config["worker_config"],
    )
