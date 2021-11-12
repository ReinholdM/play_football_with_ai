# -*- coding: utf-8 -*-
import time
import numpy as np

from malib.backend.datapool.offline_dataset_server import Episode
from malib.utils.metrics import get_metric


def grf_simultaneous(
    trainable_pairs,
    agent_interfaces,
    env_desc,
    metric_type,
    max_iter,
    behavior_policy_mapping=None,
    role=None,
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
    
    t0 = time.time()
    env = env_desc["env"]
    # assert len(env_desc[" "]) == 1

    num_action = {aid: act_sp.n for aid, act_sp in env.action_spaces.items()}

    metric = get_metric(metric_type)(
        env.possible_agents if trainable_pairs is None else list(trainable_pairs.keys())
    )

    if behavior_policy_mapping is None:
        behavior_policy_mapping = trainable_pairs.copy()
        
        _agent_interface = next(iter(agent_interfaces.values()))
        _agent_interface.reset()
        agent_interfaces = {env_aid: _agent_interface for env_aid in env.possible_agents}
        if len(env.possible_agents) == 2:
            oppo_name = env.possible_agents[1]
            behavior_policy_mapping.update({oppo_name: _agent_interface.behavior_policy})

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
                "share_obs",
                "steps_left"
            ],
        ) if role == "rollout" else None
        for agent in (trainable_pairs or env.possible_agents)
    }

    done = False
    step = 0
    observations = env_desc["last_observation"]

    def judge_stop(step, done):
        if max_iter == -1:
            return not done
        else:
            return step < max_iter

    while judge_stop(step, done):
        actions, action_dists = {}, {}
        values = {}

        # for MAPPO
        share_team_obs = {
            aid: np.reshape(
                observations[aid], [-1, 1, np.prod(observations[aid].shape[1:])]) 
                for aid in env.possible_agents}
        # for PPO
        # share_team_obs = observations

        for agent in agent_interfaces:
            # print('observations[agent]', observations[agent])
            action, action_dist, extra_info = agent_interfaces[agent].compute_action(
                observations[agent], policy_id=behavior_policy_mapping[agent], share_obs=share_team_obs[agent])

            actions[agent] = action
            action_dists[agent] = action_dist
            values[agent] = extra_info["value"]

        next_observations, rewards, dones, infos = env.step(actions)

        for agent in agent_interfaces:
            obs = next_observations[agent]
            if obs is None:
                next_observations[agent] = np.zeros_like(observations[agent])
        
        
        for agent in agent_episode:
            if role == "rollout":
                shape0 = observations[agent].shape[:-1] # num_env x num_agent
                    # print("obs", observations[agent].shape)
                    # print("act", actions[agent].shape)
                    # print("rews", rewards[agent].shape)
                    # print("act_dist", action_dists[agent].shape)
                    # print("next_obs", next_observations[agent].shape)
                    # print("dones", dones[agent].shape)
                    # print("share_obs", share_team_obs[agent].shape)
                    # print("values", values[agent].shape)

                agent_episode[agent].insert(
                    **{
                        Episode.CUR_OBS: np.expand_dims(observations[agent], 0),
                        Episode.ACTIONS: np.asarray([actions[agent]]),
                        Episode.REWARDS: np.asarray([rewards[agent]]),
                        Episode.ACTION_DIST: np.expand_dims(action_dists[agent], 0),
                        Episode.NEXT_OBS: np.expand_dims(next_observations[agent], 0),
                        Episode.DONES: np.asarray([dones[agent]]),
                        "active_mask": np.ones((1, *shape0, 1)),
                        "available_action": np.expand_dims(
                            observations[agent][..., :num_action[agent]], 0),
                        # "value": np.expand_dims(np.repeat(values[agent][..., None], shape0[1], axis=1), axis=0),
                        "value": values[agent][None, ...],
                        "advantage": np.zeros((1, *shape0)),
                        "return": np.zeros((1, *shape0)),
                        # "share_obs": np.repeat(share_team_obs[agent][:, None, :], shape0[1], axis=1)[None, ...]
                        "share_obs": share_team_obs[agent][None, ...],
                        "steps_left": infos[agent]["steps_left"][None,:, None, :]
                    }
                )


            metric.step(
                agent,
                behavior_policy_mapping[agent],
                observation=observations[agent],
                action=actions[agent],
                reward=rewards[agent],
                action_dist=action_dists[agent],
                done=dones[agent],
                score=infos[agent]["score"],
                goal_diff=infos[agent]["goal_diff"]
            )
        observations = next_observations
        step += 1
        done = any([d.any() for d in dones.values()])
        if done:
            observations = env.reset()
    next_step_values = {aid: 0 for aid in observations}
    # print("done:", done)
    if not done and role == "rollout":
        assert step > 0
        # share_last_step_obs1 = np.concatenate(
        #     [observations[aid] for aid in sorted(observations) if "team_1" in aid]
        # )
        # share_last_step_obs0 = np.concatenate(
        #     [observations[aid] for aid in sorted(observations) if "team_0" in aid]
        # )
        # share_last_step_obs = {"team_0": share_last_step_obs0, "team_1": share_last_step_obs1}
        share_last_step_obs = {
            aid: np.reshape(
                observations[aid], [-1, 1, np.prod(observations[aid].shape[1:])]) 
                for aid in env.possible_agents}

        for agent, interface in agent_interfaces.items():
            _, _, extra_info = agent_interfaces[agent].compute_action(
                observations[agent], policy_id=behavior_policy_mapping[agent],
                share_obs=share_last_step_obs[agent]
            )
            next_step_values[agent] = extra_info["value"]

    
    env_desc["last_observation"] = observations

    gamma = 0.99
    gae_lambda = 0.95
    # print("compute GAE-lambda")
    if role == "rollout":
        for aid, episode in agent_episode.items():
            rewards = episode.data[Episode.REWARDS]
            values = episode.data["value"][:]
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
            # print(f"ret.shape: {ret.shape}, gae.shape:{gae.shape}")
            episode_data.update({"return": ret, "advantage": gae})
            for k, v in episode_data.items():
                episode_data[k] = np.reshape(v, [-1, *v.shape[2:]])
            # print(episode_data)
            episode.fill(
                **episode_data
            )

    t1 = time.time()
    # input(f"{t1-t0} ?")
    return (
        metric.parse(agent_filter=tuple(agent_episode)),
        agent_episode,
    )
