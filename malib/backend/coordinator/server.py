"""
The coordinator server bridges tasks like training, rollouts, and payoff updates by parsing task requests, generating
new task descriptions, and dispatch them. This coordinator server implementation inherits from `BaseCoordinator`, it is
a special case for large-scale multi-agent learning actually.
"""

import os
import threading

import copy
import time
import traceback
from typing import List, Dict
import numpy as np
from numpy.core.numeric import roll

import ray

from malib import settings
from malib.gt.algos.elo_rating import EloManager
from malib.utils.formatter import pretty_print as pp
from malib.utils.typing import (
    AgentID,
    TaskDescription,
    TaskRequest,
    TaskType,
    RolloutDescription,
    TrainingDescription,
    EvaluateResult,
    TrainingFeedback,
    SimulationDescription,
    AgentInvolveInfo,
    BColors,
    RolloutFeedback,
)
from malib.utils.logger import get_logger
from malib.evaluator import get_evaluator, Evaluator
from malib.manager.rollout_worker_manager import RolloutWorkerManager
from malib.manager.training_manager import TrainingManager
from malib.evaluator.utils.payoff_manager import PayoffManager
from malib.evaluator.utils.team_payoff_manager import TeamPayoffManager
from malib.backend.coordinator.base_coordinator import BaseCoordinator
from malib.gt.nash_solver import build_solver

import pickle


@ray.remote
class CoordinatorServer(BaseCoordinator):
    """Coordinator server maintains the payoff matrix and serves for the task assignment."""

    def push(self, **kwargs):
        pass

    def pull(self, **kwargs):
        pass

    def __init__(
        self,
        **kwargs,
    ):
        """Create a coordinator server instance."""

        BaseCoordinator.__init__(self)

        self._configs = kwargs

        self._terminate = False
        self._pending_trainable_pairs = {}

        # maintain the population sets.
        self._populations = {
            agent: set()
            for agent in self._configs["env_description"]["possible_agents"]
        }
        assert (
            len(self._populations) > 0
        ), "no possible agents detected, please specify it in the env_description"

        print("Use solver:", self._configs["solver"])
        solver_fn = build_solver(self._configs["solver"])
        # payoff manager responses for the payoff management of all agents
        if self._configs["global_evaluator"]["name"] == "generic":
            self._payoff_manager = PayoffManager(
                self._configs["env_description"]["possible_agents"],
                kwargs["exp_cfg"],
                solver_fn(),
            )
        elif self._configs["global_evaluator"]["name"] == "psro":
            # self._payoff_manager = TeamPayoffManager(
            #     # FIXME(ziyu): specific version for football
            #     team_config=kwargs["team_config"],
            #     exp_cfg=kwargs["exp_cfg"],
            #     solver=solver_fn()
            # )
            self._payoff_manager = PayoffManager(
                self._configs["env_description"]["possible_agents"],
                kwargs["exp_cfg"],
                solver_fn(),
            )

        else:
            raise ValueError(
                "Unknow Global Evaluator Type {}"
                % {self._configs["global_evaluator"]["name"]}
            )
        self._generation = 0
        # hyper_evaluator: determine global convergence achievement or not
        self._hyper_evaluator: Evaluator = get_evaluator(
            self._configs["global_evaluator"]["name"]
        )(**self._configs["global_evaluator"]["config"])

        self._rollout_worker_manager = None
        self._training_manager = None
        self._offline_dataset = None
        self._lock = threading.Lock()
        self._exp_cfg = kwargs["exp_cfg"]
        self._logger = get_logger(
            log_level=settings.LOG_LEVEL,
            log_dir=settings.LOG_DIR,
            remote=settings.USE_REMOTE_LOGGER,
            mongo=settings.USE_MONGO_LOGGER,
            name="coordinator",
            **kwargs["exp_cfg"],
        )

        if self._configs["training"]["use_bot"]:
            print("################ USE BUILT-IN-BOT ###############")
            self._payoff_manager.get_pending_matchups(
                "team_0", "built_in_ai", policy_config={})
            self._payoff_manager.update_payoff(RolloutFeedback(
                worker_idx=None,
                agent_involve_info=None,
                statistics={"team_0": {"score":0.5, "goal_diff": 0.0}, "team_1": {"score":0.5, "goal_diff": 0.0}},
                policy_combination={"team_0": ("built_in_ai", None), 
                    "team_1": ("built_in_ai", None)},
                episode_cnt=1
            ))

    def start(self):
        self._rollout_worker_manager = RolloutWorkerManager(
            worker_config=self._configs["worker_config"],
            rollout_config=self._configs["rollout"],
            env_desc=self._configs["env_description"],
            benchmark_env_desc=self._configs["benchmark_env_description"],
            exp_cfg=self._exp_cfg,
        )

        self._training_manager = TrainingManager(
            algorithms=self._configs["algorithms"],
            env_desc=self._configs["env_description"],
            interface_config=self._configs["training"]["interface"],
            training_agent_mapping=self._configs["agent_mapping_func"],
            training_config=self._configs["training"]["config"],
            initial_policy_config=self._configs["training"].get("initial_policy", None),
            exp_cfg=self._exp_cfg,
        )
        retries = 100
        while True:
            try:
                #         if self._coordinator is None:
                #             self._coordinator = ray.get_actor(settings.COORDINATOR_SERVER_ACTOR)

                #         if self._parameter_server is None:
                #             self._parameter_server = ray.get_actor(
                #                 settings.PARAMETER_SERVER_ACTOR
                #             )

                if self._offline_dataset is None:
                    self._offline_dataset = ray.get_actor(
                        settings.OFFLINE_DATASET_ACTOR
                    )
                    print("get OfflineDataset")
                # self._status = Status.IDLE
                break
            except Exception as e:
                retries -= 1
                if retries == 0:
                    self.logger.error("reached maximum retries")
                    raise RuntimeError(traceback.format_exc())
                else:
                    self._logger.warning(
                        f"waiting for coordinator server initialization ... \n"
                    )
                    time.sleep(1)
        self._training_manager.init()
        self._logger.info("Coordinator server started")

    def request(self, task_request: TaskRequest):
        """Handling task request"""

        if task_request.task_type == TaskType.SIMULATION:
            # content is TrainingFeedback
            task_request = self._training_manager.retrieve_information(task_request)
            pending_matches = []
            for (
                env_aid,
                ptup,
            ) in task_request.content.agent_involve_info.trainable_pairs.items():
                if task_request.content.trainable:
                    self._pending_trainable_pairs[env_aid] = ptup
                # else:
                #     self.gen_test_task(task_request, {env_aid: ptup})
                # create matches here
                print(">>>>>>>>>>>>>>>>>>>> server add policy:", env_aid, ptup[0])
                pending_matches.extend(
                    self._payoff_manager.get_pending_matchups(env_aid, *ptup)
                )
            if len(pending_matches) > 0:
                self.gen_simulation_task(task_request, pending_matches)
        elif task_request.task_type == TaskType.EVALUATE:
            """Requests from rollout worker after rollout tasks done, or agent.AgentInterface after optimize tasks done.
            Evaluate task here aims to determine whether to do simulation or terminate task directly.
            """
            # task_request = self._rollout_worker_manager.retrieve_information(
            #     task_request
            # )
            populations = task_request.content.agent_involve_info.populations
            trainable_pairs = task_request.content.agent_involve_info.trainable_pairs
            pending_matches = []
            for env_aid, ptup in trainable_pairs.items():
                pending_matches.extend(
                    self._payoff_manager.get_pending_matchups(env_aid, *ptup)
                )

            if len(pending_matches) == 0:
                self._logger.warning(
                    BColors.WARNING + "repeated policy id detected!" + BColors.ENDC
                )
                for env_aid, ptup in trainable_pairs.items():
                    self._pending_trainable_pairs[env_aid] = ptup
            else:
                self.gen_simulation_task(task_request, pending_matches)
        elif task_request.task_type == TaskType.UPDATE_PAYOFFTABLE:
            """Update payoff table after simulations, and then generate new policies"""
            # task_request = self._rollout_worker_manager.retrieve_information(
            #     task_request
            # )
            with self._lock:
                self.update_payoff_table(task_request)
        elif task_request.task_type == TaskType.ROLLOUT:
            task_request = self._training_manager.retrieve_information(task_request)
            self.gen_rollout_task(task_request)
        elif task_request.task_type == TaskType.OPTIMIZE:
            task_request = self._training_manager.retrieve_information(task_request)
            self.gen_optimization_task(task_request.content.agent_involve_info)
        elif task_request.task_type == TaskType.BENCHMARK_RECORD:
            content: RolloutFeedback = task_request.content
            # print("BENCHMARK_RECORD:",
            #       content.statistics,
            #       # task_request.content.policy_combination,
            #       content.episode_cnt)
            for team_id, statistics in content.statistics.items():
                scores = {
                    team_id
                    + "_"
                    + content.policy_combination[team_id][0]: statistics["score"]
                }
                scores["built_in_ai"] = 1.0 - statistics["score"]
                # print(scores)
                self._payoff_manager.record_new_match_result(
                    scores, content.episode_cnt
                )
        elif task_request.task_type == TaskType.TERMINATE:
            self._terminate = True
        else:
            raise TypeError(f"Unexpected task type: {task_request.task_type}")

    def update_payoff_table(self, task_request: TaskRequest):
        """Update payoff table, add evaluated policies and generate new policies
        if all policies finished their simulation.
        """

        rollout_feedback = task_request.content
        self._payoff_manager.update_payoff(rollout_feedback)

        agent_involve_info = rollout_feedback.agent_involve_info
        population_mapping = agent_involve_info.populations
        # filter policy configuration
        population_mapping = {
            mpid: [pt[0] for pt in ptlist]
            for mpid, ptlist in population_mapping.items()
        }

        for env_aid, p_tuple in agent_involve_info.trainable_pairs.items():
            if rollout_feedback.trainable:
                self._pending_trainable_pairs[env_aid] = p_tuple
            if p_tuple[0] not in population_mapping[env_aid]:
                population_mapping[env_aid].append(p_tuple[0])
        for env_aid, p_tuple in self._pending_trainable_pairs.items():
            if p_tuple[0] not in population_mapping[env_aid]:
                population_mapping[env_aid].append(p_tuple[0])

        all_done = self._payoff_manager.check_done(population_mapping)
        # print(f"-------- all_done={all_done}, {len(self._pending_trainable_pairs)},{len(self._populations)}")
        # print("--------------- payoff:", self._payoff_manager.payoffs)
        if all_done and len(self._pending_trainable_pairs) == len(self._populations):
            self._logger.info("All pending payoffs have been updated")

            self._logger.debug(
                f"sending policy adding task with pending trainable pairs:"
                f"\n{pp(self._pending_trainable_pairs)}"
            )
            # gen new population mapping
            new_population_mapping = copy.copy(population_mapping)
            # check not trainable
            # for agent, ele in new_population_mapping.items():
            #     if (
            #         self._pending_trainable_pairs[agent][0]
            #         in new_population_mapping[agent]
            #     ):
            #         continue
            #     new_population_mapping[agent].append(
            #         self._pending_trainable_pairs[agent][0]
            #     )
            state_id = ray.put(new_population_mapping)
            print("---------- new pop mapping:", new_population_mapping)
            # TODO(ming): require a better implementation, or move this branch to payoff_manager
            # if len(next(iter(new_population_mapping.values()))) > 1:
            if True:
                _population_mapping = new_population_mapping.copy()
                if self._configs["training"]["use_bot"]:
                    _population_mapping["team_0"].append("built_in_ai")
                equilibrium = self._payoff_manager.compute_equilibrium(
                    _population_mapping.copy()
                )
                self._payoff_manager.update_equilibrium(
                    _population_mapping.copy(), equilibrium
                )
                # oracle payoffs: payoff aggregation with a equilibrium (Nash / Coordination / ...)
                oracle_payoffs = None
                # weighted payoffs: payoff aggregation with the learned best response and fixed opponent policies
                weighted_payoffs = None
                self._generation = len(next(iter(new_population_mapping.values())))
                # self.clear_offline_dataset()
                if self._configs["global_evaluator"]["name"] == "psro":
                    # exp = self._training_manager.get_exp(equilibrium)
                    # print("######### payoff:")
                    # print(list(self._payoff_manager.payoffs.values())[0].table)
                    # self.gen_test_task(task_request, self._pending_trainable_pairs)
                    elo = self._payoff_manager.get_elo(new_population_mapping.copy())
                    rpp = self._payoff_manager.get_rpp(_population_mapping.copy())
                    print("######### payoff:", self._payoff_manager.payoffs)
                    print("######### equilibriumn:", equilibrium)
                    print("######### elo:", elo)
                    print("######### rpp:", rpp)

                    # dump payoff manager
                    dump_path = os.path.join(
                        settings.LOG_DIR,
                        self._exp_cfg["expr_group"],
                        self._exp_cfg["expr_name"],
                        "payoff_manager.pkl"
                    )
                    with open(dump_path, "wb") as f:
                        pickle.dump({"elo": elo, "payoff_table": self._payoff_manager.payoffs}, f)

                    for team_id, probs in equilibrium.items():
                        max_key = max(probs, key=probs.get)
                        self._logger.send_scalar(
                            tag=f"Elo/{team_id}",
                            content=elo[team_id][max_key],
                            global_step=self._generation,
                        )
                        print(
                            f"{team_id} major elo, {max_key}: {elo[team_id][max_key]}"
                        )

            else:
                weighted_payoffs = None
                oracle_payoffs = None
            evaluate_result = self._hyper_evaluator.evaluate(
                # content here should be
                task_request.content,
                weighted_payoffs=weighted_payoffs,
                oracle_payoffs=oracle_payoffs,
                trainable_mapping=self._pending_trainable_pairs,
            )
            if evaluate_result[EvaluateResult.CONVERGED]:
                self._terminate = True
            else:

                self._pending_trainable_pairs = {}
                for aid in self._training_manager.groups:
                    self.gen_add_policy_task(aid, state_id)
        else:
            self._logger.warning(
                f"payoff evaluation for policies doesn't finish yet, skip policy adding."
            )

    def gen_test_task(self, task_request, new_policies):
        # agent_involve_info: AgentInvolveInfo = task_request.content.agent_involve_info
        # load default episode length ?
        max_episode_length = self._configs["evaluation"].get("max_episode_length", 1000)
        num_episodes = self._configs["evaluation"].get("num_episodes", 2)
        callback = self._configs["rollout"]["callback"]
        task_desc = TaskDescription(
            task_type=TaskType.SIMULATION,
            content=SimulationDescription(
                callback=callback,
                max_episode_length=max_episode_length,
                agent_involve_info=task_request.content.agent_involve_info,
                policy_combinations=[new_policies],
                num_episodes=num_episodes,
                trainable=None
            ),
            state_id=None,
        )
        self._rollout_worker_manager.test(task_desc)

    def gen_simulation_task(self, task_request: TaskRequest, matches: List):
        """Generate simulation task for a group of agents"""
        agent_involve_info: AgentInvolveInfo = task_request.content.agent_involve_info
        # load default episode length ?
        max_episode_length = self._configs["evaluation"].get("max_episode_length", 1000)
        num_episodes = self._configs["evaluation"].get("num_episodes", 2)
        callback = self._configs["rollout"]["callback"]
        task_desc = TaskDescription(
            task_type=TaskType.SIMULATION,
            content=SimulationDescription(
                callback=callback,
                max_episode_length=max_episode_length,
                agent_involve_info=agent_involve_info,
                policy_combinations=matches,
                num_episodes=num_episodes,  # self._evaluate_config["num_simulation"] * 5
                trainable=task_request.content.trainable
            ),
            state_id=None,
        )
        self._rollout_worker_manager.simulate(task_desc)

    def gen_optimization_task(self, agent_involve_info: AgentInvolveInfo):
        task_desc = TaskDescription(
            task_type=TaskType.OPTIMIZE,
            content=TrainingDescription(
                agent_involve_info=agent_involve_info,
                stopper=self._configs["training"]["config"]["stopper"],
                batch_size=self._configs["training"]["config"]["batch_size"],
                update_interval=self._configs["training"]["config"]["update_interval"],
            ),
            state_id=None,
        )
        self._training_manager.optimize(task_desc)

    def gen_add_policy_task(self, aid: str, state_id):
        """Generate policy adding task then dispatch to one agent interface
        :param aid: str, agent interface id
        :param state_id:
        """

        # tag current task with state_id
        task_desc = TaskDescription(
            task_type=TaskType.ADD_POLICY, content=None, state_id=state_id
        )
        self._training_manager.add_policy(aid, task_desc)

    def gen_rollout_task(self, task_request: TaskRequest):
        """Generate rollout task by parsing task request from AgentActor"""

        assert isinstance(task_request.content, TrainingFeedback)

        populations = task_request.content.agent_involve_info.populations
        population_mapping = {}
        for k, v in populations.items():
            assert len(v) > 0, v
            population_mapping[k] = [p[0] for p in v]
        agent_involve_info = task_request.content.agent_involve_info

        if self._configs["training"]["use_bot"]:
            population_mapping["team_0"].append("built_in_ai")
        if all([len(p_list) for p_list in population_mapping.values()]):
            policy_distribution = self._payoff_manager.get_equilibrium(
                population_mapping
            )
            if self._configs["training"]["use_bot"]:
                benchmark_ratio = policy_distribution["team_0"]["built_in_ai"]
                policy_dist_arr = np.array([policy_distribution["team_0"][pid] 
                    for pid in policy_distribution["team_0"] if pid != "built_in_ai"])
                if policy_dist_arr.sum() <= 1e-2:
                    policy_dist = np.ones_like(policy_dist_arr) / policy_dist_arr.shape[0]
                else:
                    policy_dist = policy_dist_arr / policy_dist_arr.sum()
                del policy_distribution["team_0"]["built_in_ai"]
                policy_distribution["team_0"] = dict(zip(policy_distribution["team_0"].keys(), 
                                                     policy_dist.tolist()))
            else:
                benchmark_ratio = 0.0
            
        else:
            policy_distribution = {}
            benchmark_ratio = 1.0 if self._configs["training"]["use_bot"] else 0.0
        print(f"ROLLOUT, benchmark ratio: {benchmark_ratio}, policy_dist:{policy_distribution}")
        
        rollout_config = self._configs["rollout"]
        task = TaskDescription(
            task_type=TaskType.ROLLOUT,
            content=RolloutDescription(
                agent_involve_info=agent_involve_info,
                policy_distribution=policy_distribution,
                fragment_length=rollout_config["fragment_length"],
                num_episodes=rollout_config["num_episodes"],
                stopper=rollout_config["stopper"],
                stopper_config=rollout_config["stopper_config"],
                terminate_mode=rollout_config["terminate"],
                mode=rollout_config["mode"],
                callback=rollout_config["callback"],
                episode_seg=rollout_config["episode_seg"],
                benchmark_ratio=benchmark_ratio
            ),
            state_id=None,
        )

        self._rollout_worker_manager.rollout(task_desc=task)

    def clear_offline_dataset(self):
        # wait untill clear done.
        ray.get(self._offline_dataset.clear.remote())

    def is_terminate(self):
        return self._terminate

    def terminate(self):
        self._training_manager.terminate()
        self._rollout_worker_manager.terminate()
