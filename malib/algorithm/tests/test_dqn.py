# import logging
# import os.path as osp
# import numpy as np
#
# from collections import defaultdict
# from pettingzoo.mpe import simple_tag_v2, simple_v2
# from torch.utils.tensorboard import SummaryWriter
#
# from malib.algorithm.dqn import DQNTrainer, POLICY_NAME
# from malib.rollout.sampler import SyncSampler
# from malib.policy.meta_policy import MetaPolicy
# from malib.utils.formatter import pretty_dict
# from malib.utils.metrics import get_metric
#
#
# BASE_DIR = osp.dirname(osp.abspath(__file__))
#
#
# env_config = {
#     "num_good": 1,
#     "num_adversaries": 1,
#     # "num_obstacles": 2,
#     "max_cycles": 25,
# }
#
# env_description = {
#     "creator": simple_tag_v2.env,
#     "config": env_config,
#     "id": "simple_tag_v2",
# }
#
# rollout_config = {
#     "fragment_length": 100,
#     "num_episode": 2,
#     "terminate_mode": "any",
#     "evaluator": "generic",
# }
#
#
# env = simple_tag_v2.env(**env_config)
# possible_agents = env.possible_agents
# observation_spaces = env.observation_spaces
# action_spaces = env.action_spaces
#
#
# meta_policies = {
#     agent: MetaPolicy(agent, observation_spaces[agent], action_spaces[agent])
#     for agent in possible_agents
# }
#
#
# for agent, meta_policy in meta_policies.items():
#     pid = f"ppo_{agent}"
#     meta_policy.reset()
#     meta_policy.add_policy(
#         pid,
#         description={
#             "registered_name": POLICY_NAME,
#             "observation_space": observation_spaces[agent],
#             "action_space": action_spaces[agent],
#             "model_config": {"hidden_dims": [128, 256, 128], "dueling": True},
#             "custom_config": {},
#         },
#     )
#
#
# trainers = {
#     agent: DQNTrainer(
#         observation_space=observation_spaces[agent],
#         action_space=action_spaces[agent],
#         policy=list(meta_policies[agent].population.values())[0],
#     )
#     for agent in possible_agents
# }
#
#
# sampler = SyncSampler(
#     None,
#     env_desc=env_description,
#     meta_policies=meta_policies,
#     meta_policy_mapping_func=lambda agent: agent,
#     metric=get_metric("simple")(possible_agents),
#     log_dir=osp.join(BASE_DIR, "logs"),
#     log_level=logging.DEBUG,
# )
#
#
# max_iteration = 500
#
# writer = SummaryWriter(log_dir=osp.join(BASE_DIR, "logs/run/simple_tag/ppo"))
# for iteration in range(max_iteration):  # iteration < max_iteration:
#     agent_batches = defaultdict(lambda: [])
#     statistic_list = []
#     num_batches = 0
#     step_so_far = 0
#     max_batches = 16
#     batch_size = 32
#     step_max = 10
#
#     sampler.ready_for_rollout(
#         fragment_length=env_config["max_cycles"] * max_batches,
#         terminate_mode="any",
#         policy_distribution={
#             agent: dict(zip(meta_policies[agent].population.keys(), [1.0]))
#             for agent in possible_agents
#         },
#         mode="rollout",
#     )
#     print("\n" + "-" * 5 + " rollout stage " + "-" * 5)
#     while step_so_far < rollout_config["fragment_length"] and num_batches < max_batches:
#         statistic, multiagent_batch = sampler.rollout()
#         statistic_list.append(statistic)
#         batches = multiagent_batch.get_cleaned_batches()
#         num_batches += 1
#         step_so_far += multiagent_batch.count
#
#         for (batch_meta_info, cleaned_batch) in batches:
#             agent_batches[batch_meta_info.meta_policy_id].append(cleaned_batch)
#
#     merged_agent_statistics = sampler.metric.merge_parsed(statistic_list)
#     for agent, statistics in merged_agent_statistics.items():
#         for k, v in statistics.items():
#             sub_name = f"{k}/{agent}"
#             writer.add_scalar(sub_name, v, iteration)
#     # merge batches and shuffle
#     tmp_batch = {}
#     for agent, batch_seq in agent_batches.items():
#         legal_keys = list(batch_seq[0].keys())
#         # shuffle
#         idx = np.arange(step_so_far)
#         np.random.shuffle(idx)
#         tmp_batch[agent] = {
#             key: np.row_stack([e[key] for e in batch_seq])[idx] for key in legal_keys
#         }
#
#     # do traninig
#     print("\n" + "-" * 5 + " training stage " + "-" * 5)
#     for agent, trainer in trainers.items():
#         # fix one agent
#         if agent == "adversary_0":
#             continue
#         # sample batch from batches
#         for step in range(step_max):
#             idx = np.random.choice(step_so_far, size=batch_size)
#             batch = {k: v[idx] for k, v in tmp_batch[agent].items()}
#             loss_stats = trainer.optimize(batch, other_agent_batches=None)
#             print(
#                 f"[iteration {iteration} / step {step + 1}({step_max})] agent - {agent}:\n{pretty_dict(loss_stats, indent=1)}"
#             )
# writer.close()
