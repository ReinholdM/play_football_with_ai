# import logging
# import os.path as osp
# import numpy as np
#
# from collections import defaultdict
# from pettingzoo.mpe import simple_tag_v2, simple_v2
# from torch.utils.tensorboard import SummaryWriter
#
# from malib.algorithm.sac import SACTrainer, POLICY_NAME
# from malib.rollout.sampler import SyncSampler
# from malib.policy.meta_policy import MetaPolicy
# from malib.utils.formatter import pretty_dict
# from malib.utils.metrics import get_metric
#
# BASE_DIR = osp.dirname(osp.abspath(__file__))
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
# env = simple_tag_v2.env(**env_config)
# possible_agents = env.possible_agents
# observation_spaces = env.observation_spaces
# action_spaces = env.action_spaces
#
# meta_policies = {
#     agent: MetaPolicy(agent, observation_spaces[agent], action_spaces[agent])
#     for agent in possible_agents
# }
#
# # TODO(ming): add policy
# for agent, meta_policy in meta_policies.items():
#     pid = f"sac_{agent}"
#     meta_policy.reset()
#     meta_policy.add_policy(
#         pid,
#         description={
#             "registered_name": POLICY_NAME,
#             "observation_space": observation_spaces[agent],
#             "action_space": action_spaces[agent],
#             "model_config": {"hidden_dims": [256, 64]},
#             "custom_config": {
#                 "gamma": 0.99,
#                 "entropy_coef": 0.1,
#                 "polyak": 0.99,
#                 "lr": 1e-4,
#             },
#         },
#     )
#
# trainers = {
#     agent: SACTrainer(
#         observation_space=observation_spaces[agent],
#         action_space=action_spaces[agent],
#         policy=list(meta_policies[agent].population.values())[0],
#     )
#     for agent in possible_agents
# }
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
# max_iteration = 500
# writer = SummaryWriter(log_dir=osp.join(BASE_DIR, "logs/run/simple_tag/sac"))
# for iteration in range(max_iteration):  # iteration < max_iteration:
#     statistic_list = []
#     agent_batches = defaultdict(lambda: [])
#     num_batches = 0
#     step_so_far = 0
#     max_batches = 64
#     batch_size = 64
#     step_max = 3
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
#     # TODO(ming): write to tensorboard summary
#     merged_agent_statistics = sampler.metric.merge_parsed(statistic_list)
#     for agent, statistics in merged_agent_statistics.items():
#         for k, v in statistics.items():
#             sub_name = f"{k}/{agent}"
#             writer.add_scalar(sub_name, v, iteration)
#     # merge batches and shuffle
#     for agent, batch_seq in agent_batches.items():
#         legal_keys = list(batch_seq[0].keys())
#         # shuffle
#         idx = np.arange(step_so_far)
#         np.random.shuffle(idx)
#         agent_batches[agent] = {
#             key: np.row_stack([e[key] for e in batch_seq])[idx] for key in legal_keys
#         }
#
#     # do traninig
#     print("\n" + "-" * 5 + " training stage " + "-" * 5)
#     for agent, trainer in trainers.items():
#         # sample batch from batches
#         for step in range(step_max):
#             idx = np.random.choice(step_so_far, size=batch_size)
#             batch = {k: v[idx] for k, v in agent_batches[agent].items()}
#             loss_stats = trainer.optimize(
#                 agent_batches[agent], other_agent_batches=None
#             )
#             print(
#                 f"[iteration {iteration} / step {step + 1}({step_max})] agent - "
#                 f"{agent}:\n{pretty_dict(loss_stats, indent=1)}"
#             )
# writer.close()
