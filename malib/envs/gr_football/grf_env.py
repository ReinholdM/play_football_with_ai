# -*- coding: utf-8 -*-
import copy
from typing import Callable

import gym
from gym import spaces
import numpy as np
from pettingzoo import ParallelEnv
from gfootball import env as raw_grf_env
from malib.envs.gr_football.encoders import encoder_basic, rewarder_basic


class BaseGFootBall(ParallelEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, use_built_in_GK=True, scenario_config={}):
        super().__init__()
        self.kwarg = scenario_config
        self._raw_env = raw_grf_env.create_environment(
            # env_name=kwarg["env_name"],
            # number_of_left_players_agent_controls=kwarg["number_of_left_players_agent_controls"],
            # number_of_right_players_agent_controls=kwarg["number_of_right_players_agent_controls"],
            # representation=kwarg["representation"],
            # stacked=kwarg["stacked"],
            # logdir=kwarg["logdir"],
            # write_goal_dumps=kwarg["write_goal_dumps"],
            # write_full_episode_dumps=kwarg["write_full_episode_dumps"],
            # render=kwarg["render"],
            **scenario_config
        )
        self._use_built_in_GK = use_built_in_GK

        self._num_left = self.kwarg["number_of_left_players_agent_controls"]
        self._num_right = self.kwarg["number_of_right_players_agent_controls"]
        self._include_GK = self._num_right == 5 or self._num_left == 5
        self._use_built_in_GK = (not self._include_GK) or self._use_built_in_GK

        if self._include_GK and self._use_built_in_GK:
            assert (
                scenario_config["env_name"] == "5_vs_5"
            ), "currently only support a very specific scenario"
            assert self._num_left == 5 or self._num_left == 0
            assert self._num_right == 5 or self._num_right == 0
            self._num_right = self._num_right - 1
            self._num_left = self._num_left - 1

        self.possible_players = {
            "team_0": [
                f"team_0_player_{i+int(self._use_built_in_GK)}"
                for i in range(self._num_left)
            ],
            "team_1": [
                f"team_1_player_{i+int(self._use_built_in_GK)}"
                for i in range(self._num_right)
            ],
        }
        self.possible_teams = ["team_0", "team_1"]

        self._repr_mode = scenario_config["representation"]
        self._rewarder = rewarder_basic

        self.possible_agents = []
        for team_players in self.possible_players.values():
            self.possible_agents.extend(team_players)
        self.n_agents = len(self.possible_agents)
        self._episode_step = -1
        self._build_interacting_spaces()

    def reset(self):
        self._prev_obs = self._raw_env.reset()
        self.agents = self.possible_agents
        self.dones = dict(zip(self.agents, [False] * self.n_agents))
        self.scores = dict(zip(self.agents, [{"scores": [0.0]}] * self.n_agents))
        self._episode_step = 0
        return self._get_obs()
    
    def seed(self, seed=None):
        self._raw_env.seed(seed)
        return self.reset()

    def step(self, action_dict):
        action_list = []
        if self._include_GK and self._use_built_in_GK and self._num_left > 0:
            action_list.append(19)
        for i, player_id in enumerate(sorted(action_dict)):
            if self._include_GK and self._use_built_in_GK and i == self._num_left:
                # which means action_dict is of size greater than num_left,
                #  and the first one is the goal keeper
                action_list.append(19)
            action_list.append(action_dict[player_id])
        obs, rew, done, info = self._raw_env.step(action_list)

        # done = done or (self._episode_step == self._game_duration)
        self._episode_step += 1
        obs = copy.deepcopy(obs)  # FIXME(ziyu): since the underlying env have cache obs
        rew = rew.tolist()
        if self._include_GK and self._use_built_in_GK:
            self._pop_list_for_built_in_GK(obs)
            self._pop_list_for_built_in_GK(rew)

        reward = [
            self._rewarder.calc_reward(_r, _prev_obs, _obs)
            for _r, _prev_obs, _obs in zip(rew, self._prev_obs, obs)
        ]
        self._prev_obs = obs
        score = []
        goal_diff = []
        if done:
            for i, _obs in enumerate(obs):
                my_score, opponent_score = _obs["score"]
                if my_score > opponent_score:
                    score.append([1.0])
                elif my_score == opponent_score:
                    score.append([0.5])
                else:
                    score.append([0.0])
                goal_diff.append([my_score-opponent_score])
        else:
            score = [[0.0]] * len(obs)
            goal_diff = [[0.0]] * len(obs)

        steps_left = []
        for i, _obs in enumerate(obs):
            steps_left.append([_obs["steps_left"]])

        reward = self._wrap_list_to_dict(reward)
        done = self._wrap_list_to_dict([done] * len(obs))
        info = [info.copy() for _ in range(len(obs))]
        for i, _info in enumerate(info):
            _info["score"] = score[i]
            _info["goal_diff"] = goal_diff[i]
            _info["steps_left"] = steps_left[i]

        info = self._wrap_list_to_dict(info)

        return self._get_obs(), reward, done, info

    def _build_interacting_spaces(self):
        # FIXME(ziyu): limit the controllable agent's action and let it can't use built-in 19
        self._num_actions = 19
        self.action_spaces = {
            player_id: spaces.Discrete(self._num_actions)
            for player_id in self.possible_agents
        }

        if self._repr_mode == "raw":
            self._feature_encoder = encoder_basic.FeatureEncoder()
            # self._feature_encoder = self.kwarg["encoder"]
        self._raw_env.reset()
        obs = self._get_obs()
        # self._raw_env.close()
        self.observation_spaces = {
            player_id: spaces.Box(
                low=-10.0, high=10.0, shape=obs[player_id].shape, dtype=np.float32
            )
            # spaces.Dict(
            #     {
            #         "observation": spaces.Box(low=-10., high=10., shape=obs[player_id]["observation"].shape, dtype=np.float32),
            #         "action_mask": spaces.Box(low=0, high=1, shape=(self._num_actions,), dtype=np.int8)
            #     }
            # )
            for player_id in self.possible_agents
        }
        self.state_space = gym.spaces.Dict(
            **{k: v for k, v in self.observation_spaces.items() if "team_0" in k}
        )

    def _get_obs(self):
        if self._repr_mode == "raw":
            obs = self._build_observation_from_raw()
        else:
            assert not self._use_built_in_GK
            obs = self._raw_env.observation()
        return self._wrap_list_to_dict(obs)

    def _build_observation_from_raw(self):
        """
        get the observation of all player's in teams
        """

        def encode_obs(raw_obs):
            obs = self._feature_encoder.encode(raw_obs)

            obs_cat = np.hstack(
                [np.array(obs[k], dtype=np.float32).flatten() for k in sorted(obs)]
            )

            # action_mask = np.pad(raw_obs["sticky_actions"], [0, 9], constant_values=1)
            # return {"action_mask": action_mask, "observation": obs_cat}
            return obs_cat

        raw_obs_list = self._raw_env.observation()
        # tmp4check = [obs["active"] for obs in raw_obs_list]
        # assert -1 not in tmp4check, tmp4check
        if self._include_GK and self._use_built_in_GK:
            self._pop_list_for_built_in_GK(raw_obs_list)
        encode_obs_list = [encode_obs(r_obs) for r_obs in raw_obs_list]

        return encode_obs_list

    def _wrap_list_to_dict(self, data_in_list):
        # res = {k: [] for k in self.possible_teams}
        # team_name = self.possible_teams[0]
        # for i, data in enumerate(data_in_list):
        #     if i < self._num_left:
        #         res[team_name].append(data)
        #     else:
        #         team_name = self.possible_teams[1]
        #         res[team_name].append(data)
        # data_in_dict = {
        #     team_id: dict(zip(self.possible_players[team_id], res[team_id]))
        #     for team_id in self.possible_teams
        # }
        return dict(zip(self.possible_agents, data_in_list))

    def _pop_list_for_built_in_GK(self, data_list):
        assert self._include_GK and self._use_built_in_GK
        if self._num_left > 0:
            data_list.pop(0)
        if self._num_right > 0:
            data_list.pop(-self._num_right)

    def close(self):
        self._raw_env.close()


class ParameterSharingWrapper:
    def __init__(self, base_env: BaseGFootBall, parameter_sharing_mapping: Callable):
        """
        :param base_env: the environment where agents share their parameters
        :param parameter_sharing_mapping: how to share the parameters
        """

        self._env = base_env
        self.possible_agents = sorted(
            list(
                set(parameter_sharing_mapping(aid) for aid in base_env.possible_agents)
            )
        )
        self._ps_mapping_func = parameter_sharing_mapping
        self._ps_buckets = {aid: [] for aid in self.possible_agents}
        for aid in base_env.possible_agents:
            self._ps_buckets[parameter_sharing_mapping(aid)].append(aid)
        self._ps_buckets = {
            aid: sorted(self._ps_buckets[aid]) for aid in self.possible_agents
        }
        self.action_spaces = {
            aid: base_env.action_spaces[self._ps_buckets[aid][0]]
            for aid in self.possible_agents
        }
        self.observation_spaces = {
            aid: base_env.observation_spaces[self._ps_buckets[aid][0]]
            for aid in self.possible_agents
        }
        self._state = None

    @property
    def state_space(self):
        return self._env.state_space

    def reset(self):
        base_obs = self._env.reset()
        obs = self._build_from_base_dict(base_obs)
        return {aid: self._concate_obs(_obs) for aid, _obs in obs.items()}
    
    def seed(self, seed=None):
        base_obs = self._env.seed(seed)
        obs = self._build_from_base_dict(base_obs)
        return {aid: self._concate_obs(_obs) for aid, _obs in obs.items()}

    def _build_from_base_dict(self, base_dict):
        ans = {aid: [] for aid in self.possible_agents}
        for base_aid, data in base_dict.items():
            ans[self._ps_mapping_func(base_aid)].append(data)

        return ans

    def step(self, action_dict):
        base_action = self._extract_to_base(action_dict)
        obs, reward, done, info = self._env.step(base_action)
        # obs = {aid: self._concate_obs(_obs) for aid, _obs in self._build_from_base_dict(obs).items()}
        def f(x_dict, reduce_func):
            x_dict = self._build_from_base_dict(x_dict)
            return {k: reduce_func(v) for k, v in x_dict.items()}

        return (
            f(obs, self._concate_obs),
            f(reward, np.vstack),
            f(done, np.vstack),
            f(info, lambda x: x[0]),
        )

    def _extract_to_base(self, from_dict):
        to_dict = {}
        for aid, bucket in self._ps_buckets.items():
            for i, base_aid in enumerate(bucket):
                to_dict[base_aid] = from_dict[aid][i]
        return to_dict

    def _concate_obs(self, obs_list):
        # pure_obs_list = [obs["observation"] for obs in obs_list]
        # pure_action_mask_list = [obs["action_mask"] for obs in obs_list]
        # return {"observation": np.vstack(pure_obs_list),
        #         "action_mask": np.vstack(pure_action_mask_list)}
        return np.vstack(obs_list)

    def close(self):
        self._env.close()


if __name__ == "__main__":
    from malib.envs.gr_football import default_config

    # default_config["other_config_options"] = {"action_set": "v2"}
    default_config["scenario_config"]["env_name"] = "malib_5_vs_5"
    env = BaseGFootBall(**default_config)
    env = ParameterSharingWrapper(env, lambda x: x[:6])
    print(env.possible_agents)
    print(env.observation_spaces)
    print(env.action_spaces)
    env.reset()
    done = False
    while True:
        actions = {aid: np.zeros(4, dtype=int) for aid in env.possible_agents}
        obs, reward, done, info = env.step(actions)
