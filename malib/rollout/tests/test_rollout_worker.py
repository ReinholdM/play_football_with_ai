# # -*- coding:utf-8 -*-
# # Create Date: 11/19/20, 1:27 PM
# # Author: ming
# # ---
# # Last Modified: 11/19/20
# # Modified By: ming
# # ---
# # Copyright (c) 2020 MARL @ SJTU
#
# import pytest
# import gym
# import numpy as np
# import threading
# import queue
#
# from malib.rollout.rollout_worker import RolloutWorker
# from malib.manager.rollout_worker_manager import RolloutWorkerManager
# from malib.utils.configs.config import POLICY_CONFIG, update_config
# from malib.algorithm.ddpg.policy import DDPG as DDPGPolicy
# from malib.utils.typing import TaskDescription
# from malib.backend.coordinator.server import CoordinatorServer as Coordinator
# from malib.utils.typing import StandardEnvReturns
#
#
# def env_creator(agent_ids, **kwargs):
#     class Wrapper:
#         """ Wrap single agent atari as multi-agent atari """
#
#         def __init__(self, agent_ids, env):
#             self._env = env
#             self._agent_ids = agent_ids
#             self.render = self._env.render
#             self.close = self._env.close
#             self.action_space = env.action_space
#             self.observation_space = env.observation_space
#
#         @property
#         def agent_ids(self):
#             return self._agent_ids
#
#         def reset(self):
#             obs = self._env.reset()
#             return {aid: obs for aid in self._agent_ids}
#
#         def step(self, action_dict) -> StandardEnvReturns:
#             action = list(action_dict.values())[0]
#             if isinstance(self.action_space, gym.spaces.Discrete) and len(action) > 1:
#                 action = np.argmax(action)
#             next_obs, reward, done, info = self._env.step(action)
#             # pack as dict
#             next_obs = {aid: next_obs for aid in self._agent_ids}
#             reward = {aid: reward for aid in self._agent_ids}
#             done = {aid: done for aid in self._agent_ids}
#             info = {aid: info for aid in self._agent_ids}
#
#             return next_obs, reward, done, info
#
#     return Wrapper(agent_ids, gym.make(**kwargs))
#
#
# @pytest.fixture
# def agent_ids():
#     return [f"agent-{i}" for i in range(1)]
#
#
# @pytest.fixture
# def env_config():
#     return {"id": "CartPole-v1"}
#
#
# @pytest.fixture
# def env(agent_ids, env_config):
#     return env_creator(agent_ids, **env_config)
#
#
# @pytest.fixture
# def policy_config(env):
#     return update_config(
#         POLICY_CONFIG,
#         {
#             "name": "ddpg-agent",
#             "action_space": env.action_space,
#             "observation_space": env.observation_space,
#             "model_config": {},
#             "custom_config": {},
#         },
#     )
#
#
# @pytest.fixture
# def rollout_config(agent_ids, env_config):
#     return {
#         "sample_sync": True,
#         "task": None,
#         "env": {"agent_ids": agent_ids, **env_config},
#         "env_creator": env_creator,
#     }
#
#
# @pytest.fixture
# def rollout_task_config():
#     return {"fragment_length": 100, "num_episodes": 1, "terminate_mode": "any"}
#
#
# @pytest.fixture
# def parameter_server_config():
#     return {"maxsize": 1}
#
#
# @pytest.fixture
# def offline_dataset_server_config():
#     return {"maxsize": 1}
#
#
# @pytest.fixture
# def rollout_worker_config(
#     parameter_server_config,
#     offline_dataset_server_config,
#     policy_config,
#     env_config,
#     rollout_config,
#     env,
# ):
#     coordinator = Coordinator(
#         {"p_config": parameter_server_config, "d_config": offline_dataset_server_config}
#     )
#     policy_cls = DDPGPolicy
#     agent_ids = env.agent_ids
#
#     # XXX(ming): we use single-thread mode by setting `task_queue` and `idle_worker_queue` as Nones
#     return {
#         "worker_index": "test-0",
#         "task_queue": None,  # task_queue,
#         "idle_worker_queue": None,  # idle_worker_queue,
#         "coordinator": coordinator,
#         "policy_config": {aid: (policy_cls, policy_config) for aid in agent_ids},
#         "rollout_config": rollout_config,
#     }
#
#
# def test_single_instance(rollout_worker_config, rollout_task_config):
#     rollout_worker = RolloutWorker(**rollout_worker_config)
#
#     # test only one task
#     task_desc = TaskDescription(identify="simple", rollout=rollout_task_config)
#     rollout_worker.start(task_desc)
#     rollout_worker.close()
#
#
# def test_rollout_worker_manager(
#     rollout_worker_config,
#     agent_ids,
#     policy_config,
#     rollout_config,
#     rollout_task_config,
#     parameter_server_config,
#     offline_dataset_server_config,
# ):
#     task_queue_size = 3
#     worker_num = 3
#     pcls = DDPGPolicy
#
#     task_queue = queue.Queue(maxsize=task_queue_size)
#     idle_worker_queue = queue.Queue(maxsize=worker_num)
#
#     rm_configs = dict(
#         coordinator_config={
#             "data_pool_client_config": {
#                 "p_config": parameter_server_config,
#                 "d_config": offline_dataset_server_config,
#             }
#         },
#         policy_config={aid: [(pcls, policy_config)] for aid in agent_ids},
#         rollout_config=rollout_config,
#         task_queue=task_queue,
#         idle_worker_queue=idle_worker_queue,
#         worker_num=worker_num,
#     )
#
#     def run_rm_manager(configs):
#         rm = RolloutWorkerManager(**configs)
#         rm.run()
#
#     task_num = 10
#
#     def run_manager(configs):
#         rm = RolloutWorkerManager(**configs)
#         rm.run()
#
#     def add_task(task_num, task_queue):
#         for i in range(task_num):
#             print(f"[add_task] Add task=simple-{i}")
#             task_desc = TaskDescription(
#                 identify=f"simple-{i}", rollout=rollout_task_config
#             )
#             task_queue.put(task_desc)
#         # task_queue.put("quit")
#
#     threading.Thread(target=add_task, args=(task_num, task_queue), daemon=True).start()
#     threading.Thread(target=add_task, args=(task_num, task_queue), daemon=True).start()
#
#     # FIXME(ming): lack of error checking
#     threading.Thread(target=run_manager, args=(rm_configs,), daemon=True).start()
#     # run_manager(rm_configs)
#
#     task_queue.join()
#     idle_worker_queue.join()
#
#     print("Workers were terminated")
