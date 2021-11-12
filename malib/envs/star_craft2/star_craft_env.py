from collections import defaultdict

import gym
import numpy as np
from gym import spaces
from pettingzoo import ParallelEnv
from smac.env import StarCraft2Env as sc_env
from smac.env.starcraft2.starcraft2 import StarCraft2Env

from malib.envs.vector_env import VectorEnv
from malib.utils.typing import Dict, Any, List, Tuple, Union, AgentID


agents_list = {
    "3m": [f"Marine_{i}" for i in range(3)],
    "8m": [f"Marine_{i}" for i in range(8)],
    "25m": [f"Marine_{i}" for i in range(25)],
    "2s3z": [f"Stalkers_{i}" for i in range(2)] + [f"Zealots_{i}" for i in range(3)],
    "3s5z": [f"Stalkers_{i}" for i in range(3)] + [f"Zealots_{i}" for i in range(5)],
}
# FIXME(ziyu): better ways or complete the rest map information


def get_agent_names(map_name):
    if map_name in agents_list:
        return agents_list[map_name]
    else:
        return None


class SC2Env(ParallelEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, **kwargs):
        super(SC2Env, self).__init__()
        self.smac_env = sc_env(**kwargs)
        self.env_info = self.smac_env.get_env_info()
        self.kwargs = kwargs
        self.n_agents = self.smac_env.n_agents

        self.possible_agents = agents_list.get(
            self.kwargs["map_name"], [f"{i}" for i in range(self.n_agents)]
        )
        self.agents = self.possible_agents

        n_obs = self.env_info["obs_shape"]
        num_actions = self.env_info["n_actions"]
        self.global_state_space = gym.spaces.Box(
            low=0, high=1, shape=(self.env_info["state_shape"],)
        )
        self.observation_spaces = dict(
            zip(
                self.possible_agents,
                [
                    spaces.Box(low=0, high=1, shape=(n_obs,), dtype=np.int8)
                    for _ in range(self.n_agents)
                ],
            )
        )

        self.action_spaces = dict(
            zip(
                self.possible_agents,
                [spaces.Discrete(num_actions) for _ in range(self.n_agents)],
            )
        )

    def reset(self):
        """only return observation not return state"""
        self.agents = self.possible_agents
        obs_t, state_t = self.smac_env.reset()
        action_mask = np.array(self.smac_env.get_avail_actions())

        obs = {aid: obs_t[i] for i, aid in enumerate(self.agents)}
        state = {aid: state_t for aid in self.agents}
        action_mask = {aid: action_mask[i] for i, aid in enumerate(self.agents)}

        return obs, state, action_mask

    def step(self, actions):
        act_list = [actions[aid] for aid in self.agents]
        reward, terminated, info = self.smac_env.step(act_list)
        next_obs_t = self.smac_env.get_obs()
        next_state = self.get_state()
        next_action_mask = np.array(self.smac_env.get_avail_actions())

        rew_dict = {aid: reward for aid in self.agents}
        done_dict = {aid: terminated for aid in self.agents}
        next_obs_dict = {aid: next_obs_t[i] for i, aid in enumerate(self.agents)}
        next_state_dict = {aid: next_state for aid in self.agents}
        action_mask = {aid: next_action_mask[i] for i, aid in enumerate(self.agents)}

        info_dict = {
            aid: {**info, "action_mask": next_action_mask[i]}
            for i, aid in enumerate(self.agents)
        }

        return (
            next_obs_dict,
            next_state_dict,
            rew_dict,
            done_dict,
            action_mask,
            info_dict,
        )

    def get_state(self):
        return self.smac_env.get_state()

    def render(self, mode="human"):
        """not implemented now in smac"""
        # self._env.render()
        pass

    def close(self):
        self.smac_env.close()


class VecStarCraft(VectorEnv):
    def __init__(
        self,
        observation_spaces: Dict[AgentID, gym.Space],
        action_spaces: Dict[AgentID, gym.Space],
        env_configs: Dict[str, Any],
        num_envs: int,
    ):
        super().__init__(
            observation_spaces, action_spaces, SC2Env, env_configs, num_envs
        )

    def reset(self) -> Any:
        obs = defaultdict(list)
        state = defaultdict(list)
        action_mask = defaultdict(list)

        for env in self.envs:
            agent_obs, agent_state, agent_action_mask = env.reset()

            for agent, _obs in agent_obs.items():
                obs[agent].append(_obs)
                state[agent].append(agent_state[agent])
                action_mask[agent].append(agent_action_mask[agent])

        return obs, state, action_mask

    def step(self, actions: Dict[AgentID, List]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        next_obs_batched = defaultdict(list)
        next_state_batched = defaultdict(list)
        rewards_batched = defaultdict(list)
        dones_batched = defaultdict(list)
        action_mask_batched = defaultdict(list)
        info_batched = defaultdict(list)

        for i, env in enumerate(self.envs):
            _actions = {aid: a[i] for aid, a in actions.items()}
            next_agent_obs, next_state, rewards, dones, action_masks, infos = env.step(
                _actions
            )
            for agent, obs in next_agent_obs.items():
                next_obs_batched[agent].append(obs)
                rewards_batched[agent].append(rewards[agent])
                dones_batched[agent].append(dones[agent])
                action_mask_batched[agent].append(action_masks[agent])
                info_batched[agent].append(infos[agent])
                next_state_batched[agent].append(next_state[agent])

        return (
            next_obs_batched,
            next_state_batched,
            rewards_batched,
            dones_batched,
            action_mask_batched,
            info_batched,
        )


def make_env(num_envs: int = 1, **kwargs) -> Union[VecStarCraft, StarCraft2Env]:
    if num_envs == 1:
        return SC2Env(**kwargs)
    elif num_envs > 1:
        env = SC2Env(**kwargs)
        action_spaces = env.action_spaces
        observation_spaces = env.observation_spaces
        env.close()
        return VecStarCraft(observation_spaces, action_spaces, kwargs, num_envs)
    else:
        raise ValueError(
            f"num of environment cannot smaller than 1 (num_envs={num_envs})"
        )
