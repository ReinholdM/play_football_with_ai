"""
Implementation of async rollout worker.
"""
import multiprocessing
import time
from collections import defaultdict

import numpy as np
from numpy.lib.npyio import save

import ray
from ray.util import ActorPool

import uuid
from malib import settings
from malib.backend.datapool.offline_dataset_server import Episode, MultiAgentEpisode
from malib.envs.agent_interface import AgentInterface
from malib.rollout import rollout_func
from malib.rollout.base_worker import BaseRolloutWorker
from malib.utils.logger import Log, get_logger
from malib.utils.typing import Any, Dict, BehaviorMode, MetricEntry, Tuple, Sequence, MetricType

# XXX(ziyu): for football
from malib.envs.gr_football.vec_wrapper import DummyVecEnv, SubprocVecEnv
from functools import partial

VecEnv = SubprocVecEnv

class Func:
    def __init__(self, exp_cfg, env_desc, benchmark_env_desc):
        self.logger = get_logger(
            log_level=settings.LOG_LEVEL,
            log_dir=settings.LOG_DIR,
            name=f"rolloutfunc_executor_{uuid.uuid1()}",
            remote=settings.USE_REMOTE_LOGGER,
            mongo=settings.USE_MONGO_LOGGER,
            **exp_cfg,
        )
        # init environment here
        self.env_desc = env_desc
        self.benchmark_env_desc = benchmark_env_desc

    @classmethod
    def as_remote(
        cls,
        num_cpus: int = None,
        num_gpus: int = None,
        memory: int = None,
        object_store_memory: int = None,
        resources: dict = None,
    ) -> type:
        """Return a remote class for Actor initialization."""

        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources,
        )(cls)

    def run(
        self,
        trainable_pairs,
        agent_interfaces,
        # env_desc,
        metric_type,
        max_iter,
        policy_mapping,
        num_episode,
        callback,
        role="rollout",
        agent_mapping_func=lambda x: x,
        dataset_server=None,
        save_interval=100,
    ):
        assert role != "rollout"
        if "built_in_ai" in policy_mapping.values():
            role = "test"
            policies = list(policy_mapping.values())
            policies.pop(policies.index("built_in_ai"))
            policy_mapping = {"team_0": policies[0]}
        if role == "simulation" and len(set(policy_mapping.values())) == 1:
            # print("GOT IT! ", policy_mapping)
            return [{aid: {"score": MetricEntry(0.5, "mean"), "goal_diff": MetricEntry(0.0, "mean")} for aid in policy_mapping}], 0
            
        env_desc = self.env_desc if role == "simulation" else self.benchmark_env_desc
        ith = 0
        statics, data = [], []
        if isinstance(callback, str):
            callback = rollout_func.get_func(callback)

        env_for_desc = env_desc["creator"](**env_desc["config"])
        mapped_agent_interfaces = {
            aid: agent_interfaces[agent_mapping_func(aid)]
            for aid in env_for_desc.possible_agents
        }
        env_for_desc.close()

        episode_seg = max(1, multiprocessing.cpu_count() - 8)  # XXX(ziyu): maybe should leave some for pytorch cpu forward
        total_steps = num_episode // episode_seg
        episodes = [episode_seg] * total_steps + [num_episode - episode_seg * total_steps]
        data_size = 0

        seed = np.random.randint(0, 65536)

        while ith < len(episodes):
            if episodes[ith] == 0:
                break
            try:
                env_desc["env"] = VecEnv(
                    [partial(env_desc["creator"], **env_desc["config"])] * episodes[ith]
                )
                env_desc["last_observation"] = env_desc["env"].seed(seed)
                tmp_statistic, tmp_data = callback(
                    trainable_pairs=trainable_pairs,
                    agent_interfaces=mapped_agent_interfaces,
                    env_desc=env_desc,
                    metric_type=metric_type,
                    max_iter=max_iter,
                    behavior_policy_mapping=policy_mapping,
                    role=role,
                )
                env_desc["env"].close()
                del env_desc["env"]
            except Exception as e:
                print(role, trainable_pairs)
                print(policy_mapping)
                for aid, ait in agent_interfaces.items():
                    print(aid, ait.policies.keys())
                raise e
            tmp_statistic = {agent_mapping_func(k): v for k, v in tmp_statistic.items()}
            tmp_data = {agent_mapping_func(k): v for k, v in tmp_data.items()}

            statics.append(tmp_statistic)
            if role == "rollout":
                data.append(tmp_data)
                data_size += list(tmp_data.values())[0].size
            ith += 1

            if dataset_server and ith % save_interval == 0:
                dataset_server.save.remote(data)
                data = []
            
            seed += 1024

        if dataset_server:
            dataset_server.save.remote(data)

        # merged_data = defaultdict(list)
        # merged_capacity = defaultdict(lambda: 0)
        # for d in data:
        #     for aid, episode in d.items():
        #         # merged_data[aid].append(episode)
        #         merged_capacity[aid] += episode.size

        # data2send = {
        # aid: Episode.concatenate(*merged_data[aid], capacity=merged_capacity[aid])
        # for aid in merged_data
        # }

        return statics, data_size

    def benchmark_rollout(
        self,
        trainable_pairs,
        agent_interfaces,
        # env_desc,
        # benchmark_env_desc,
        metric_type,
        max_iter,
        num_episode,
        callback,
        role,
        dataset_server,
        benchmark_ratio=0.0,
        save_interval=1,
    ):
        t0 = time.time()
        assert role == "rollout"
        ith = 0
        statics, data = [], []
        if isinstance(callback, str):
            callback = rollout_func.get_func(callback)

        data_size = 0
        # print("Func.benchmark_rollout, benchmark_ratio:", benchmark_ratio)
        num_normal, num_benchmark = 0, 0

        episode_seg = num_episode
        # episode_seg = num_episode
        total_steps = num_episode // episode_seg
        episodes = [episode_seg] * total_steps + [num_episode - episode_seg * total_steps]
        seed = np.random.randint(0, 65536)
        while ith < len(episodes):
            if episodes[ith] == 0:
                break
            if np.random.rand() < benchmark_ratio:

                def test_agent_mapping_func(aid, to_team):
                    aid_split = aid.split("_")
                    aid_split[1] = str(to_team)
                    return "_".join(aid_split)

                team = next(iter(trainable_pairs))[5]
                agent_mapping_func = lambda x: test_agent_mapping_func(x, team)

                trainable_pairs = {
                    test_agent_mapping_func(k, 0): v for k, v in trainable_pairs.items()
                }
                callback_env = self.benchmark_env_desc
                num_benchmark += episodes[ith]
            else:
                agent_mapping_func = lambda x: x
                callback_env = self.env_desc
                num_normal += episodes[ith]
            if "env" not in callback_env:
                # print(f"benchmark rollout, create env")
                callback_env["env"] = DummyVecEnv(
                    [partial(callback_env["creator"], **callback_env["config"])] * episodes[ith]
                )
                if "last_observation" in callback_env:
                    del callback_env["last_observation"]
            if "last_observation" not in callback_env:
                callback_env["last_observation"] = callback_env["env"].seed(seed)
                seed += 1024

            mapped_agent_interfaces = {
                aid: agent_interfaces[agent_mapping_func(aid)]
                for aid in callback_env["possible_agents"]
            }

            tmp_statistic, tmp_data = callback(
                trainable_pairs=trainable_pairs,
                agent_interfaces=mapped_agent_interfaces,
                env_desc=callback_env,
                metric_type=metric_type,
                max_iter=max_iter,
                behavior_policy_mapping=None,
                role=role,
            )
            # callback_env["env"].close()
            tmp_statistic = {agent_mapping_func(k): v for k, v in tmp_statistic.items()}
            tmp_data = {agent_mapping_func(k): v for k, v in tmp_data.items()}
            trainable_pairs = {agent_mapping_func(k): v for k, v in trainable_pairs.items()}

            for item_name, metric_entry in list(tmp_statistic.values())[0].items():
                ori_tag = metric_entry.tag
                ori_tag_split_list = ori_tag.split('/')
                ori_tag_split_list[0] = agent_mapping_func(ori_tag_split_list[0])
                new_tag = '/'.join(ori_tag_split_list)
                metric_entry.tag = new_tag

            statics.append(tmp_statistic)
            data.append(tmp_data)
            data_size += list(tmp_data.values())[0].size
            ith += 1

            if ith % save_interval == 0:
                dataset_server.save.remote(data)
                data = []

            dataset_server.save.remote(data)
            # print(f"After benchmark rollout, normal rollout={num_normal}, benchmark rollout={num_benchmark}")
        return statics, data_size

    def close(self):
        if self.env is not None:
            self.env.close()


class RolloutWorker(BaseRolloutWorker):
    """For experience collection and simulation, the operating unit is env.AgentInterface"""

    def __init__(
        self,
        worker_index: Any,
        env_desc: Dict[str, Any],
        metric_type: str,
        benchmark_env_desc,
        remote: bool = False,
        **kwargs,
    ):

        """Create a rollout worker instance.

        :param worker_index: Any, indicate rollout worker
        :param env_desc: Dict[str, Any], the environment description
        :param metric_type: str, name of registered metric handler
        :param remote: bool, tell this rollout worker work in remote mode or not, default by False
        """
        BaseRolloutWorker.__init__(
            self,
            worker_index,
            env_desc,
            metric_type,
            benchmark_env_desc,
            remote,
            **kwargs,
        )

        parallel_num = kwargs.get("parallel_num", 0)
        if parallel_num:
            parallel_num = max(parallel_num, 5)
            RemoteFunc = Func.as_remote()
            self.actors = [
                RemoteFunc.remote(kwargs["exp_cfg"], env_desc, benchmark_env_desc)
                for _ in range(parallel_num)
            ]
            self.actor_pool = ActorPool(self.actors)
        self._parallel_num = parallel_num

    def ready_for_sample(self, policy_distribution=None):
        """Reset policy behavior distribution.
        :param policy_distribution: Dict[AgentID, Dict[PolicyID, float]], the agent policy distribution
        """

        return BaseRolloutWorker.ready_for_sample(self, policy_distribution)

    def _rollout(
        self, threaded, episode_seg, **kwargs
    ) -> Tuple[Sequence[Dict], Sequence[Any]]:
        """Helper function to support rollout."""
        assert threaded
        data_size = 0
        if threaded:
            stat_data_tuples = self.actor_pool.map_unordered(
                lambda a, v: a.benchmark_rollout.remote(
                    num_episode=v,
                    **kwargs,
                ),
                episode_seg,
            )
        else:
            stat_data_tuples = []
            for v in episode_seg:
                statistics, data = Func.run(None, num_episode=v, **kwargs)
                stat_data_tuples.append((statistics, data))
        statistic_seq = []
        for statis, num_transition in stat_data_tuples:
            # for aid, episode in data.items():
            # merged_data[aid].append(episode)
            # merged_capacity[aid] += episode.size
            statistic_seq.append(statis)
            data_size += num_transition

        return statistic_seq, data_size

    def _simulation(self, threaded, combinations, **kwargs):
        """Helper function to support simulation."""
        if threaded:
            print(f"got simulation task: {len(combinations)}")
            # FIXME(ming): map or unordered_map
            res = self.actor_pool.map(
                lambda a, combination: a.run.remote(
                    trainable_pairs=None,
                    policy_mapping={aid: v[0] for aid, v in combination.items()},
                    **kwargs,
                ),
                combinations,
            )
            # depart res into two parts
            statis = [e[0] for e in res]
            return statis, None
        else:
            statis = []
            for comb in combinations:
                tmp, _ = Func.run(
                    None,
                    trainable_pairs=None,
                    policy_mapping={aid: v[0] for aid, v in comb.items()},
                    **kwargs,
                )
                statis.append(tmp)
            return statis, None

    def _test(self, threaded, combinations, **kwargs):
        """
        test the latest combination, each with the benchmark env
        combinations is actually trainable_pairs
        """
        assert threaded
        agent_mapping_func = kwargs.pop("agent_mapping_func", lambda x: x)
        # FIXME(ziyu): hard coded here making limited and ugly use for grfootball

        num_teams = 1
        generation = set(pid[0].split("_")[-1] for pid in combinations.values())
        assert len(generation) == 1, generation
        generation = next(iter(generation))
        res = {}
        for i in range(num_teams):

            def test_agent_mapping_func(aid, to_team):
                aid_split = aid.split("_")
                aid_split[1] = str(to_team)
                return "_".join(aid_split)

            self.actor_pool.submit(
                lambda a, v: a.run.remote(
                    trainable_pairs=None,
                    agent_mapping_func=lambda x: agent_mapping_func(
                        test_agent_mapping_func(x, i)
                    ),
                    policy_mapping={
                        test_agent_mapping_func(aid, 0): v[0]
                        for aid, v in combinations.items()
                        if f"team_{i}" in aid
                    },
                    **kwargs,
                ),
                None,
            )
            stats = self.actor_pool.get_next()[0]
            # print(i, stats[0].keys())
            # stats = [{test_agent_mapping_func(aid, i): v for aid, v in _stats.items()}
            #          for _stats in stats]
            with Log.stat_feedback() as (
                statistic_seq,
                merged_statistics,
            ):
                statistic_seq.extend(stats)
            merged_statistics = merged_statistics[0]
            _stats = next(iter(merged_statistics.values()))
            # FIXME(ziyu): currently just print it.
            print(
                f"\n>>>>>>>>>>>  Test In BenchmarkEnv  <<<<<<<<<<<\n"
                f"TEAM: {i}, GENERATION: {generation}\n"
                f"REWARD: {_stats[MetricType.REWARD]}, SCORE: {_stats['score']}, WIN: {_stats['win']}\n"
                f"==============================================\n"
            )
            self.logger.send_scalar(
                tag=f"Test/Team_{i}/Reward",
                content=_stats[MetricType.REWARD],
                global_step=generation,
            )
            self.logger.send_scalar(
                tag=f"Test/Team_{i}/Score",
                content=_stats["score"],
                global_step=generation,
            )
            self.logger.send_scalar(
                tag=f"Test/Team_{i}/Win", content=_stats["win"], global_step=generation
            )
            # with Log.stat_feedback(log=False, logger=self.logger) as (
            #         statistic_seq,
            #         merged_statistics,
            # ):
            #     statistic_seq.extend(stats)
            merged_statistics = {
                agent_mapping_func(test_agent_mapping_func(k, i)): v
                for k, v in merged_statistics.items()
            }
            res.update(merged_statistics)
        return res, None

    def sample(self, *args, **kwargs) -> Tuple[Sequence[Dict], Sequence[Any]]:
        """Sample function. Support rollout and simulation. Default in threaded mode.

        :param args:
        :param kwargs:
        :return: A tuple
        """

        callback = kwargs["callback"]
        behavior_policy_mapping = kwargs.get("behavior_policy_mapping", None)
        num_episodes = kwargs["num_episodes"]
        trainable_pairs = kwargs.get("trainable_pairs", None)
        threaded = kwargs.get("threaded", True)
        explore = kwargs.get("explore", True)
        fragment_length = kwargs.get("fragment_length", 1000)
        # Add test for grfootball
        role = kwargs["role"]  # rollout or simulation or test

        if explore:
            for aid, interface in self._agent_interfaces.items():
                if aid in trainable_pairs:
                    interface.set_behavior_mode(BehaviorMode.EXPLORATION)
                else:
                    interface.set_behavior_mode(BehaviorMode.EXPLOITATION)
        else:
            for interface in self._agent_interfaces.values():
                interface.set_behavior_mode(BehaviorMode.EXPLOITATION)

        if role == "rollout":
            return self._rollout(
                threaded,
                num_episodes,
                trainable_pairs=trainable_pairs,
                agent_interfaces=self._agent_interfaces,
                # env_desc=self._env_description,
                # benchmark_env_desc=self._benchmark_env_desc,
                metric_type=self._metric_type,
                max_iter=fragment_length,
                callback=callback,
                role="rollout",
                dataset_server=self._offline_dataset,
                benchmark_ratio=kwargs["benchmark_ratio"],
            )
        elif role == "simulation":
            return self._simulation(
                threaded,
                behavior_policy_mapping,
                agent_interfaces=self._agent_interfaces,
                # env_desc=self._env_description,
                metric_type=self._simulation_metric_type,
                max_iter=fragment_length,
                callback=callback,
                num_episode=num_episodes,
                role="simulation",
            )
        elif role == "test":
            return self._test(
                threaded,
                behavior_policy_mapping,
                agent_interfaces=self._agent_interfaces,
                # env_desc=self._benchmark_env_desc,
                metric_type=self._simulation_metric_type,
                max_iter=fragment_length,
                callback=callback,
                num_episode=num_episodes,
                role=role,
            )

    # @Log.method_timer(enable=False)
    def update_population(self, agent, policy_id, policy):
        """Update population with an existing policy instance"""

        agent_interface: AgentInterface = self._agent_interfaces[agent]
        agent_interface.policies[policy_id] = policy

    def close(self):
        BaseRolloutWorker.close(self)
        for actor in self.actors:
            actor.close.remote()
            actor.stop.remote()
            actor.__ray_terminate__.remote()
