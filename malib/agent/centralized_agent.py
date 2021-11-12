"""
Centralized agent interface is designed for multi-point optimization
"""

from copy import deepcopy

import gym
import ray
import time

from malib.algorithm import get_algorithm_space
from malib.utils import metrics
from malib.utils.typing import (
    Dict,
    Any,
    AgentID,
    PolicyID,
    MetricEntry,
    Callable,
    Tuple,
    BufferDescription,
)
from malib.agent.ctde_agent import CTDEAgent
from malib.algorithm.common.policy import Policy


class CentralizedAgent(CTDEAgent):
    def save(self, model_dir):
        pass

    def load(self, model_dir):
        pass

    def __init__(
        self,
        assign_id: str,
        env_desc: Dict[str, Any],
        algorithm_candidates: Dict[str, Any],
        training_agent_mapping: Callable,
        observation_spaces: Dict[AgentID, gym.spaces.Space],
        action_spaces: Dict[AgentID, gym.spaces.Space],
        exp_cfg: Dict[str, Any],
        population_size: int = -1,
        algorithm_mapping: Callable = None,
    ):
        assert (
            env_desc.get("teams") is not None
        ), f"Env description should specify the teams, {env_desc}"

        CTDEAgent.__init__(
            self,
            assign_id=assign_id,
            env_desc=env_desc,
            algorithm_candidates=algorithm_candidates,
            training_agent_mapping=training_agent_mapping,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            exp_cfg=exp_cfg,
            population_size=population_size,
            algorithm_mapping=algorithm_mapping,
        )

        self._teams = deepcopy(env_desc["teams"])

        self._agent_to_team = {}
        _ = [
            self._agent_to_team.update({v: k for v in ids})
            for k, ids in self._teams.items()
        ]

    def request_data(
        self, buffer_desc: Dict[AgentID, BufferDescription]
    ) -> Tuple[Dict, str]:
        """Request training data from remote `OfflineDatasetServer`.

        Note:
            This method could only be called in multi-instance scenarios. Or, `OfflineDataset` and `CoordinatorServer`
            have been started.

        :param Dict[AgentID,BufferDescription] buffer_desc: A dictionary of agent buffer descriptions.
        :return: A tuple of agent batches and information.
        """

        batch, info = None, ""

        while batch is None:
            batch, info = ray.get(
                self._offline_dataset.sample.remote(buffer_desc, alignment=True)
            )
            if batch is None:
                time.sleep(1)

        return batch, info

    def optimize(
        self,
        policy_ids: Dict[AgentID, PolicyID],
        batch: Dict[AgentID, Any],
        training_config: Dict[str, Any],
    ) -> Dict[AgentID, Dict[str, MetricEntry]]:
        res = {}
        # extract a group of policies
        for tid, env_agents in self._teams.items():
            t_policies = {
                env_aid: self.policies[policy_ids[env_aid]] for env_aid in env_agents
            }
            trainer = self.get_trainer(tid)
            trainer.reset(t_policies, training_config)
            # share batch
            # batch = batch[env_agents[0]]
            batch = trainer.preprocess(batch, other_policies=t_policies)
            res[tid] = metrics.to_metric_entry(
                trainer.optimize(batch.copy()), prefix=tid
            )
        return res

    def add_policy_for_agent(
        self, env_agent_id: AgentID, trainable: bool
    ) -> Tuple[PolicyID, Policy]:
        assert env_agent_id in self.agent_group(), (env_agent_id, self.agent_group())
        algorithm_conf = self.get_algorithm_config(env_agent_id)
        pid = self.default_policy_id_gen(algorithm_conf)

        if pid in self.policies:
            return pid, self.policies[pid]
        else:
            algorithm_space = get_algorithm_space(algorithm_conf["name"])
            custom_config = algorithm_conf.get("custom_config", {})
            # group spaces into a new space
            team_agents = self._teams[self._agent_to_team[env_agent_id]]
            custom_config.update(
                {
                    "global_state_space": custom_config.get("global_state_space")
                    or gym.spaces.Dict(
                        {
                            "obs": gym.spaces.Dict(
                                **{
                                    aid: self._observation_spaces[aid]
                                    for aid in team_agents
                                }
                            ),
                            "act": gym.spaces.Discrete(len(team_agents)),
                        }
                    )
                }
            )

            policy = algorithm_space.policy(
                registered_name=algorithm_conf["name"],
                observation_space=self._observation_spaces[env_agent_id],
                action_space=self._action_spaces[env_agent_id],
                model_config=algorithm_conf.get("model_config", {}),
                custom_config=custom_config,
            )

            # mapping agent to tid
            for tid, env_agent_ids in self._teams.items():
                if env_agent_id in env_agent_ids and tid not in self._trainers:
                    self._trainers[tid] = algorithm_space.trainer(tid)
                    # assign team agents to it
                    self._trainers[tid].agents = env_agent_ids.copy()

            self.register_policy(pid, policy)
            return pid, policy
