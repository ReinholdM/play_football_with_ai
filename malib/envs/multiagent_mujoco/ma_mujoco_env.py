# -*- coding: utf-8 -*-
from pettingzoo import ParallelEnv

# from .src.multiagent_mujoco.mujoco_multi import MujocoMulti
from src.multiagent_mujoco.mujoco_multi import MujocoMulti


# TODO(ziyu): figure out how to add action mask,
#  and is there any necessary action in fact?
class MaMujocoEnv(ParallelEnv):
    def __init__(self, **kwargs):
        self.env_args = kwargs
        self.mj_env = MujocoMulti(env_args=kwargs)

        env_info = self.mj_env.get_env_info()
        self.n_agents = env_info["n_agents"]

        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.agents = []

        self.observation_spaces = dict(
            zip(self.possible_agents, self.mj_env.observation_space)
        )
        self.action_spaces = dict(zip(self.possible_agents, self.mj_env.action_space))

    def seed(self, seed=None):
        self.mj_env = MujocoMulti(env_args=self.env_args)
        self.agents = []

    def reset(self):
        self.agents = self.possible_agents
        obs_t = self.mj_env.reset()

        return dict(zip(self.possible_agents, obs_t))

    def step(self, actions):
        act_list = [actions[aid] for aid in self.agents]
        reward, terminated, info = self.mj_env.step(act_list)
        next_obs_t = self.mj_env.get_obs()
        rew_dict = {aid: reward for aid in self.agents}
        done_dict = {aid: terminated for aid in self.agents}
        next_obs_dict = dict(zip(self.agents, next_obs_t))
        info_dict = {aid: info.copy() for aid in self.agents}

        if terminated:
            self.agents = []
        return next_obs_dict, rew_dict, done_dict, info_dict

    def get_state(self):
        return self.mj_env.get_state()

    def get_available_actions(self):
        return dict(zip(self.agents, self.mj_env.get_avail_actions()))

    def render(self, mode="human"):
        self.mj_env.render()

    def close(self):
        self.mj_env.close()


# Please specify LD_LIBRARY_PATH
if __name__ == "__main__":

    env_config = {
        "scenario": "HalfCheetah-v2",
        "agent_conf": "2x3",
        "agent_obsk": 0,
        "episode_limit": 1000,
    }
    env = MaMujocoEnv(**env_config)
    print(env.possible_agents)

    for aid, obsp in env.observation_spaces.items():
        print(aid, type(obsp))

    obs = env.reset()
    while True:
        act_dict = {}
        # legal_acts = env.get_available_actions()
        for i, aid in enumerate(env.agents):
            # avail_actions_ind = np.nonzero(legal_acts[aid])[0]
            # action = np.random.uniform(-1.0, 1.0, n_actions)
            action = env.action_spaces[aid].sample()
            act_dict[aid] = action
        print(act_dict)
        print(obs)
        next_obs, rew, done, info = env.step(act_dict)
        print(rew, done)
        print(info)
        obs = next_obs
        if all(done.values()):
            break
        print()
