import gym

from malib.utils.typing import Dict, AgentID, Any, List, Tuple


class VectorEnv:
    def __init__(
        self,
        observation_spaces: Dict[AgentID, gym.Space],
        action_spaces: Dict[AgentID, gym.Space],
        creator: type,
        configs: Dict[str, Any],
        num_envs: int,
    ):
        self.envs = [creator(**configs) for _ in range(num_envs)]
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.possible_agents = list(observation_spaces.keys())

        self._num_envs = num_envs
        self._creator = creator
        self._configs = configs.copy()

        print(f"Create {num_envs} environments")

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def env_creator(self):
        return self._creator

    @property
    def env_configs(self):
        return self._configs

    @classmethod
    def from_envs(cls, envs: List, config: Dict[str, Any]):
        """Generate vectorization environment from exisiting environments."""

        observation_spaces = envs[0].observation_spaces
        action_spaces = envs[0].action_spaces

        vec_env = cls(observation_spaces, action_spaces, config, 0)
        vec_env.add_envs(envs=envs)

        return vec_env

    def add_envs(self, envs: List = None, num: int = 0):
        """Add exisiting `envs` or `num` new environments to this vectorization environment. If `envs` is not empty or None, the `num` will be ignored."""

        if envs and len(envs) > 0:
            for env in envs:
                self.envs.append(env)
                self._num_envs += 1
            print(f"added {len(envs)} exisiting environments.")
        elif num > 0:
            for _ in range(num):
                self.envs.append(self.env_creator(**self.env_configs))
                self._num_envs += 1
            print(f"created {num} new environments.")

    def reset(self) -> Any:
        raise NotImplementedError

    def step(self, actions: Dict[AgentID, List]) -> Tuple[Dict, Dict, Dict, Dict]:
        raise NotImplementedError

    def close(self):
        for env in self.envs:
            env.close()
