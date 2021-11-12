import os
import pytest
import ray
import time
import logging

from tabulate import tabulate
from collections import namedtuple

from ray.util import ActorPool

from malib.envs.atari import VecAtari, make_env
from malib.rollout.rollout_worker import Func
from malib.envs.agent_interface import AgentInterface
from malib.algorithm.random.policy import RandomPolicy
from malib.utils.typing import AgentID, Dict, Tuple


TimeStamp = namedtuple("TimeStamp", "start, end")
LOGGER = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class VecFunc(Func):
    def __init__(self, exp_cfg, env_desc):
        Func.__init__(self, exp_cfg, env_desc)

    def vec_run(self, agent_interfaces, callback):
        return callback(agent_interfaces=agent_interfaces, env=self.env)

    def add_envs(self, maximum: int):
        existing_env_num = getattr(self.env, "num_envs", 1)

        if not isinstance(self.env, VecAtari):
            self.env = VecAtari.from_envs([self.env], self.env_desc["config"])

        if existing_env_num == maximum:
            return self.env.num_envs

        self.env.add_envs(num=maximum - existing_env_num)

        return self.env.num_envs


def random_sequential(agent_interfaces, env) -> Tuple[TimeStamp, int]:
    for interface in agent_interfaces.values():
        interface.reset()

    start_time = time.time()
    obs_dict = env.reset()
    done = False
    cnt = 0
    act_dict = {}
    while not done and cnt < env.max_iter:
        for aid, obs_seq in obs_dict.items():
            obs_seq = agent_interfaces[aid].transform_observation(obs_seq)
            act_dict[aid], _, _ = agent_interfaces[aid].compute_action(obs_seq)
        next_obs_dict, _, done_dict, _ = env.step(act_dict)
        obs_dict = next_obs_dict
        # sync done
        done = any(
            [any(v) if not isinstance(v, bool) else v for v in done_dict.values()]
        )
        cnt += 1
    end_time = time.time()
    return TimeStamp(start_time, end_time), cnt * getattr(env, "num_envs", 1)


@pytest.fixture
def episode_segs():
    return range(1, 30, 2)


@pytest.fixture
def actor_size():
    max_mum = os.cpu_count()
    LOGGER.info(f"got maximum: {max_mum}")
    return range(1, max_mum, 2)


@pytest.fixture
def gen_seg_condition(episode_segs, actor_size):
    # total_episode_num = 500
    res = []
    for e_seg in episode_segs:
        for a_size in actor_size:
            res.append((e_seg, a_size))
    return res


@pytest.fixture
def exp_cfg():
    return {"expr_group": "test", "expr_name": "atari"}


@pytest.fixture
def env_desc():
    env_desc = {
        "creator": make_env,
        "config": {
            "env_id": "basketball_pong_v1",
            "num_players": 2,
            "obs_type": "grayscale_image",
            "wrappers": [
                {"name": "resize_v0", "params": [84, 84]},
                {"name": "dtype_v0", "params": ["float32"]},
                {
                    "name": "normalize_obs_v0",
                    "params": {"env_min": 0.0, "env_max": 1.0},
                },
            ],
        },
    }

    env = env_desc["creator"](**env_desc["config"])
    env_desc["possible_agents"] = env.possible_agents.copy()
    env.close()
    del env

    return env_desc


@pytest.fixture
def agent_interfaces(env_desc) -> Dict[AgentID, AgentInterface]:
    env = env_desc["creator"](**env_desc["config"])

    interfaces = {
        aid: AgentInterface(
            aid,
            env.observation_spaces[aid],
            env.action_spaces[aid],
            None,
            policies={
                "random_1": RandomPolicy(
                    "random_1",
                    env.observation_spaces[aid],
                    env.action_spaces[aid],
                    None,
                    None,
                ),
                "random_2": RandomPolicy(
                    "random_2",
                    env.observation_spaces[aid],
                    env.action_spaces[aid],
                    None,
                    None,
                ),
            },
            sample_dist={"random_1": 0.4, "random_2": 0.6},
        )
        for aid in env.possible_agents
    }

    return interfaces


def test_vectorization(exp_cfg, env_desc, agent_interfaces):
    vecfunc = VecFunc(exp_cfg, env_desc)
    for e_seg in range(3):
        num_envs = vecfunc.add_envs(maximum=e_seg)
        vecfunc.vec_run(agent_interfaces, random_sequential)
        LOGGER.info(f"report num_envs: {num_envs}")
    vecfunc.close()


def test_fps(exp_cfg, env_desc, gen_seg_condition, agent_interfaces):
    """FPS evaluation under MAAtari settings"""

    ray.init()

    timestamp = time.time()
    log_path = os.path.join(BASE_DIR, f"{timestamp}.log")
    file_handler = logging.FileHandler(log_path)
    LOGGER.addHandler(file_handler)

    LOGGER.info("Start testing ......")
    remote_func = VecFunc.as_remote()
    table = []
    headers = ["Episode Seg", "Actor Size", "FPS (num/s)" "Time (s)"]
    actors = []

    for _ in range(0, gen_seg_condition[-1][-1]):
        actors.append(remote_func.remote(exp_cfg, env_desc))
    actor_pool = ActorPool(actors)
    for e_seg, a_size in gen_seg_condition:
        env_desc["config"]["num_envs"] = e_seg

        num_envs = ray.get([actor.add_envs.remote(maximum=e_seg) for actor in actors])

        # dones = [actors[idx].vec_run.remote(agent_interfaces, random_sequential) for idx in range(a_size)]
        res = actor_pool.map_unordered(
            lambda a, v: a.vec_run.remote(agent_interfaces, random_sequential),
            range(a_size),
        )

        time_start = float("inf")
        time_end = float("-inf")
        total_transition_num = 0

        for time_stamp, transition_num in res:
            time_start = min(time_start, time_stamp.start)
            time_end = max(time_end, time_stamp.end)
            total_transition_num += transition_num

        fps = total_transition_num / (time_end - time_start)
        time_consump = time_end - time_start
        table.append([e_seg, a_size, fps, time_consump])
        LOGGER.info(
            f"* Done for episode_seg={e_seg}, actor_size={a_size}, FPS={fps}, time={time_consump}"
        )

    # destroy actors
    for actor in actors:
        ray.get(actor.close.remote())
        # actor.stop.remote()
        actor.__ray_terminate__.remote()

    print(tabulate(table, headers=headers, tablefmt="pretty"))
    ray.shutdown()
