import time
from collections import defaultdict

import numpy as np

from malib.backend.datapool.offline_dataset_server import Episode
from malib.utils.metrics import get_metric
from malib.utils.typing import AgentID, Dict
from malib.utils.preprocessor import get_preprocessor


def sequential(
    trainable_pairs,
    agent_interfaces,
    env_desc,
    metric_type,
    max_iter,
    behavior_policy_mapping=None,
):
    """Rollout in sequential manner"""
    res1, res2 = [], []
    env = env_desc.get("env", env_desc["creator"](**env_desc["config"]))
    # for _ in range(num_episode):
    env.reset()

    # metric.add_episode(f"simulation_{policy_combination_mapping}")
    metric = get_metric(metric_type)(env.possible_agents)
    if behavior_policy_mapping is None:
        for agent in agent_interfaces.values():
            agent.reset()
    behavior_policy_mapping = behavior_policy_mapping or {
        _id: agent.behavior_policy for _id, agent in agent_interfaces.items()
    }
    agent_episode = {
        agent: Episode(
            env_desc["id"], behavior_policy_mapping[agent], capacity=max_iter
        )
        for agent in (trainable_pairs or env.possible_agents)
    }

    (observations, actions, action_dists, next_observations, rewards, dones, infos,) = (
        defaultdict(lambda: []),
        defaultdict(lambda: []),
        defaultdict(lambda: []),
        defaultdict(lambda: []),
        defaultdict(lambda: []),
        defaultdict(lambda: []),
        defaultdict(lambda: []),
    )

    for aid in env.agent_iter(max_iter=max_iter):
        observation, reward, done, info = env.last()

        if isinstance(observation, dict):
            info = {"action_mask": observation["action_mask"]}
            action_mask = observation["action_mask"]
        else:
            action_mask = np.ones(
                get_preprocessor(env.action_spaces[aid])(env.action_spaces[aid]).size
            )
        observation = agent_interfaces[aid].transform_observation(
            observation, behavior_policy_mapping[aid]
        )
        observations[aid].append(observation)
        rewards[aid].append(reward)
        dones[aid].append(done)
        info["policy_id"] = behavior_policy_mapping[aid]

        if not done:
            action, action_dist, extra_info = agent_interfaces[aid].compute_action(
                observation, **info
            )
            actions[aid].append(action)
            action_dists[aid].append(action_dist)
        else:
            info["policy_id"] = behavior_policy_mapping[aid]
            action = None
        env.step(action)
        metric.step(
            aid,
            behavior_policy_mapping[aid],
            observation=observation,
            action=action,
            action_dist=action_dist,
            reward=reward,
            done=done,
            info=info,
        )

    # metric.end()

    for k in agent_episode:
        obs = observations[k]
        cur_len = len(obs)
        agent_episode[k].fill(
            **{
                Episode.CUR_OBS: np.stack(obs[: cur_len - 1]),
                Episode.NEXT_OBS: np.stack(obs[1:cur_len]),
                Episode.DONES: np.stack(dones[k][1:cur_len]),
                Episode.REWARDS: np.stack(rewards[k][1:cur_len]),
                Episode.ACTIONS: np.stack(actions[k][: cur_len - 1]),
                Episode.ACTION_DIST: np.stack(action_dists[k][: cur_len - 1]),
            }
        )
    return (
        metric.parse(
            agent_filter=tuple(trainable_pairs.keys())
            if trainable_pairs is not None
            else None
        ),
        agent_episode,
    )


def simultaneous(
    trainable_pairs,
    agent_interfaces,
    env_desc,
    metric_type,
    max_iter,
    behavior_policy_mapping=None,
):
    """Do not support next action mask.

    :param trainable_pairs:
    :param agent_interfaces:
    :param env_desc:
    :param metric_type:
    :param max_iter:
    :param behavior_policy_mapping:
    :return:
    """

    # (ziyu): Modify for MAPPO

    env = env_desc.get("env", env_desc["creator"](**env_desc["config"]))
    num_action = {aid: act_sp.n for aid, act_sp in env.action_spaces.items()}

    metric = get_metric(metric_type)(
        env.possible_agents if trainable_pairs is None else list(trainable_pairs.keys())
    )

    if behavior_policy_mapping is None:
        for agent in agent_interfaces.values():
            agent.reset()

    behavior_policy_mapping = behavior_policy_mapping or {
        _id: agent.behavior_policy for _id, agent in agent_interfaces.items()
    }

    agent_episode = {
        agent: Episode(
            env_desc["id"],
            behavior_policy_mapping[agent],
            capacity=max_iter,
            other_columns=[
                "active_mask",
                "available_action",
                "value",
                "advantage",
                "return",
            ],
        )
        for agent in (trainable_pairs or env.possible_agents)
    }

    done = False
    step = 0
    observations = env.reset()

    for agent, obs in observations.items():
        observations[agent] = agent_interfaces[agent].transform_observation(
            obs, policy_id=behavior_policy_mapping[agent]
        )

    while step < max_iter and not done:
        actions, action_dists = {}, {}
        values = {}
        share_obs = np.hstack([observations[pid] for pid in sorted(observations)])
        for agent, interface in agent_interfaces.items():
            action, action_dist, extra_info = agent_interfaces[agent].compute_action(
                observations[agent],
                policy_id=behavior_policy_mapping[agent],
                share_obs=share_obs,
            )
            actions[agent] = action
            action_dists[agent] = action_dist
            values[agent] = extra_info["value"]

        next_observations, rewards, dones, infos = env.step(actions)

        for agent, interface in agent_interfaces.items():
            obs = next_observations[agent]
            if obs is not None:
                next_observations[agent] = interface.transform_observation(
                    obs, policy_id=behavior_policy_mapping[agent]
                )
            else:
                next_observations[agent] = np.zeros_like(observations[agent])

        for agent in agent_episode:
            agent_episode[agent].insert(
                **{
                    Episode.CUR_OBS: np.expand_dims(observations[agent], 0),
                    Episode.ACTIONS: np.asarray([actions[agent]]),
                    Episode.REWARDS: np.asarray([rewards[agent]]),
                    Episode.ACTION_DIST: np.expand_dims(action_dists[agent], 0),
                    Episode.NEXT_OBS: np.expand_dims(next_observations[agent], 0),
                    Episode.DONES: np.asarray([dones[agent]]),
                    "active_mask": np.asarray([1.0]),
                    "available_action": np.ones((1, num_action[agent])),
                    "value": np.asarray([values[agent]]),
                    "advantage": np.zeros(1),
                    "return": np.zeros(1),
                }
            )
            metric.step(
                agent,
                behavior_policy_mapping[agent],
                observation=observations[agent],
                action=actions[agent],
                reward=rewards[agent],
                actionr_dist=action_dists[agent],
                done=dones[agent],
            )
        observations = next_observations
        step += 1
        done = any(dones.values())
    next_step_values = {aid: 0 for aid in observations}
    if not done:
        assert step > 0
        share_last_step_obs = np.concatenate(
            [observations[aid] for aid in sorted(observations)]
        )
        for agent, interface in agent_interfaces.items():
            _, _, extra_info = agent_interfaces[agent].compute_action(
                observations[agent],
                policy_id=behavior_policy_mapping[agent],
                share_obs=share_last_step_obs,
            )
            next_step_values[agent] = extra_info["value"]

    # FIXME(ziyu): how to load them from config
    gamma = 0.99
    gae_lambda = 0.95

    for aid, episode in agent_episode.items():
        rewards = episode.data[Episode.REWARDS]
        values = episode.data["value"]
        gae = np.zeros_like(rewards)

        for step in reversed(range(episode.size)):
            if step == episode.size - 1:
                delta = rewards[step] + gamma * next_step_values[aid]
                gae[step] = delta
            else:
                delta = rewards[step] + gamma * values[step + 1] - values[step]
                gae[step] = delta + gamma * gae_lambda * gae[step + 1]
        ret = gae + values
        episode_data = episode.data
        episode_data = {k: episode_data[k][:] for k in episode_data}
        episode_data.update({"return": ret, "advantage": gae})
        # print(episode_data)
        episode.fill(**episode_data)

    return (
        metric.parse(agent_filter=tuple(agent_episode)),
        agent_episode,
    )


def rollout_wrapper(
    agent_episodes: Dict[AgentID, Episode] = None, rollout_type="sequential"
):
    """Rollout wrapper accept a dict of episodes outside.

    Note:
        There are still some limits here, e.g. no extra columns can be transferred to inner callback

    :param Dict[AgentID,Episode] agent_episodes: A dict of agent episodes.
    :param str rollout_type: Specify rollout styles. Default to `sequential`, choices={sequential, simultaneous}.
    :return: A function
    """

    handler = sequential if rollout_type == "sequential" else simultaneous

    def func(
        trainable_pairs,
        agent_interfaces,
        env_desc,
        metric_type,
        max_iter,
        behavior_policy_mapping=None,
    ):
        statistic, episodes = handler(
            trainable_pairs,
            agent_interfaces,
            env_desc,
            metric_type,
            max_iter,
            behavior_policy_mapping=behavior_policy_mapping,
        )
        if agent_episodes is not None:
            for agent, episode in episodes.items():
                agent_episodes[agent].insert(**episode.data)
        return statistic, episodes

    return func


def get_func(name: str):
    return {
        "sequential": sequential,
        "simultaneous": simultaneous,
    }[name]
