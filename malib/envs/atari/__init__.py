from collections import defaultdict
import importlib
import supersuit
import gym

from malib.envs.vector_env import VectorEnv
from malib.utils.typing import (
    Dict,
    Any,
    List,
    ActionSpaceType,
    Tuple,
    Sequence,
    AgentID,
)


def nested_env_creator(ori_creator: type, wrappers: Sequence[Dict]) -> type:
    """Wrap original atari environment creator with multiple wrappers"""

    def creator(**env_config):
        env = ori_creator(**env_config)
        # parse wrappers
        for wconfig in wrappers:
            name = wconfig["name"]
            params = wconfig["params"]

            wrapper = getattr(
                supersuit, name
            )  # importlib.import_module(f"supersuit.{env_desc['wrapper']['name']}")

            if isinstance(params, Sequence):
                env = wrapper(env, *params)
            elif isinstance(params, Dict):
                env = wrapper(env, **params)
            else:
                raise TypeError(f"Unexpected type: {type(params)}")
        return env

    return creator


class VecAtari(VectorEnv):
    def __init__(
        self,
        observation_spaces: Dict[AgentID, gym.Space],
        action_spaces: Dict[AgentID, gym.Space],
        configs: Dict[str, Any],
        num_envs: int,
    ):
        super().__init__(observation_spaces, action_spaces, make_env, configs, num_envs)
        self.max_iter = 100

    def reset(self) -> Any:
        """Default run in parallel mode"""

        agent_obs_list = defaultdict(list)
        for env in self.envs:
            agent_obs = env.reset()
            for agent, obs in agent_obs.items():
                agent_obs_list[agent].append(obs)

        return agent_obs_list

    def step(self, actions: Dict[AgentID, List]) -> Tuple[Dict, Dict, Dict, Dict]:
        agent_obs_batched = defaultdict(list)
        agent_reward_batched = defaultdict(list)
        agent_done_batched = defaultdict(list)
        agent_info_batched = defaultdict(list)

        for i, env in enumerate(self.envs):
            _actions = {aid: a[i] for aid, a in actions.items()}
            agent_obs, agent_reward, agent_done, agent_info = env.step(_actions)
            for agent, obs in agent_obs.items():
                agent_obs_batched[agent].append(obs)
                agent_reward_batched[agent].append(agent_reward[agent])
                agent_done_batched[agent].append(agent_done[agent])
                agent_info_batched[agent].append(agent_info[agent])

        return (
            agent_obs_batched,
            agent_reward_batched,
            agent_done_batched,
            agent_info_batched,
        )

    def close(self):
        return super().close()


def make_env(env_id, num_envs: int = 1, parallel=True, **env_configs) -> Any:
    env_module = env_module = importlib.import_module(f"pettingzoo.atari.{env_id}")
    ori_caller = env_module.env if not parallel else env_module.parallel_env
    wrappers = (
        env_configs.pop("wrappers") if env_configs.get("wrappers") is not None else []
    )
    wrapped_caller = nested_env_creator(ori_caller, wrappers)

    if num_envs == 1:
        return wrapped_caller(**env_configs)
    elif num_envs > 1:
        return VecAtari(wrapped_caller, env_configs, num_envs)
    else:
        raise ValueError(
            f"Num of environments should larger than 0, (num_envs={num_envs})"
        )
