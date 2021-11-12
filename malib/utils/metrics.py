from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Sequence, List, Dict, Any

from operator import mul
from functools import reduce

from malib import settings
from malib.utils.typing import AgentID, MetricType, PolicyID, MetricEntry
import numpy as np


def to_metric_entry(data: Dict[str, Any], prefix=""):
    """Convert a dict of metrics to a dict or metric entries.

    :param Dict[str,Any] data: Raw metrics dict.
    :return: A dict of metric entries.
    """

    res: Dict[str, MetricEntry] = {}
    for k, v in data.items():
        if isinstance(v, MetricEntry):
            res[k] = v
        else:
            res[k] = MetricEntry(
                value=v,
                agg="mean",
                tag=f"{prefix}/{k}",
                log=settings.STATISTIC_FEEDBACK,
            )
    return res


class Metric(metaclass=ABCMeta):
    def __init__(self, agents: List[AgentID]):
        self._agents = agents
        self._episode_data = dict()
        self._statistics = dict()

    @abstractmethod
    def step(self, agent_id, policy_id, **kwargs):
        pass

    @abstractmethod
    def parse(self, agent_filter=None):
        """Parse episode data and filter with given keys (agent level)"""
        pass

    @staticmethod
    def merge_parsed(
        agent_result_seq: Sequence[Dict[AgentID, Any]]
    ) -> Dict[AgentID, Dict[str, float]]:
        pass

    def reset(self):
        self._episode_data = dict()
        self._statistics = dict()


class SimpleMetrics(Metric):
    def __init__(self, agents: List[AgentID]):
        super(SimpleMetrics, self).__init__(agents)
        self._episode_data = {
            MetricType.REWARD: defaultdict(lambda: []),
        }
        self._statistics = defaultdict(
            lambda: defaultdict(
                lambda: MetricEntry(value=0, agg="mean", tag="", log=False)
            )
        )
        self._pids = {}

    def step(self, agent_id, policy_id, **kwargs):
        self._episode_data[MetricType.REWARD][agent_id].append(kwargs["reward"])
        # XXX(ming): from SC2
        # if MetricType.WON in kwargs["info"]:
        #     self._episode_data[MetricType.WON][agent_id].append(
        #         float(kwargs["info"][MetricType.WON])
        #     )
        self._pids[agent_id] = policy_id

    def parse(self, agent_filter=None) -> Dict[AgentID, Dict[str, MetricEntry]]:
        # FIXME(ming): un-compatible return
        for item_key, agent_data in self._episode_data.items():
            # if filter is not None use filter else agents
            for aid in agent_filter or self._agents:
                if self._pids.get(aid) is not None:
                    prefix = f"{aid}/{self._pids[aid]}"
                else:
                    prefix = f"{aid}"
                if item_key == MetricType.REWARD:
                    self._statistics[aid][MetricType.REWARD] = MetricEntry(
                        value=sum(agent_data[aid]),
                        agg="mean",
                        tag=f"{prefix}/{MetricType.REWARD}",
                        log=True,
                    )
        return self._statistics

    @staticmethod
    def merge_parsed(agent_result_seq: Sequence):
        agent_res = {}
        for agent_result in agent_result_seq:
            for agent_id, result in agent_result.items():
                if agent_res.get(agent_id, None) is None:
                    agent_res[agent_id] = {
                        MetricType.REWARD: 0,
                        # MetricType.LIVE_STEP: 0.0,
                    }
                agent_res[agent_id][MetricType.REWARD] += result[
                    MetricType.REWARD
                ].value / len(agent_result_seq)
        return agent_res

    def reset(self):
        self._episode_data = {
            MetricType.REWARD: defaultdict(lambda: []),
        }
        self._statistics = defaultdict(lambda: {MetricType.REWARD: 0.0})


class JointDistMetric(Metric):
    class Meta:
        REWARD = MetricType.REWARD
        ACTION_DIST = "action_dist"

    def __init__(self, agents: List[AgentID]):
        # must be list here
        agents = list(agents)
        super(JointDistMetric, self).__init__(agents)
        self._episode_data = {
            MetricType.REWARD: defaultdict(lambda: []),
            "action_dist": defaultdict(lambda: []),
        }
        self._statistics = defaultdict(
            lambda: defaultdict(
                lambda: MetricEntry(value=0, agg="mean", tag="", log=False)
            )
        )
        self._pids = {}

    def step(self, agent_id, policy_id, **kwargs):
        self._episode_data[self.Meta.REWARD][agent_id].append(kwargs[MetricType.REWARD])
        self._episode_data[self.Meta.ACTION_DIST][agent_id].append(
            kwargs[self.Meta.ACTION_DIST]
        )
        self._pids[agent_id] = policy_id

    def _cum_reward_on_joint_dist(self, main, others):
        """Calculate cumulative reward using joint policy distribution"""

        rewards = self._episode_data[MetricType.REWARD][main]
        all_dist = self._episode_data[self.Meta.ACTION_DIST]
        main_dist = [0.0 for _ in range(len(all_dist[main]))]

        if len(others):
            for i, _ in enumerate(main_dist):
                main_dist[i] = reduce(mul, [1.0] + [all_dist[o][i] for o in others])
        else:
            # return all ones
            main_dist = [1.0] * len(main_dist)

        # the head reward from sequential mode is no use
        total_reward = sum(
            [r * dist for dist, r in zip(main_dist, rewards[-len(main_dist) :])]
        )
        return total_reward

    def parse(self, agent_filter=None) -> Dict[AgentID, Dict[str, MetricEntry]]:
        """Parse episode data, return an agent wise MetricEntry dictionary"""

        # if filter is not None use filter else agents
        for i, aid in enumerate(agent_filter or self._agents):
            others = self._agents[:i] + self._agents[i + 1 :]
            if self._pids.get(aid) is not None:
                prefix = f"{aid}/{self._pids[aid]}"
            else:
                prefix = f"{aid}"
            self._statistics[aid][MetricType.REWARD] = MetricEntry(
                value=self._cum_reward_on_joint_dist(aid, others),
                agg="mean",
                tag=f"{prefix}/{MetricType.REWARD}",
                log=True,
            )
        return self._statistics

    @staticmethod
    def merge_parsed(
        agent_result_seq: Sequence[Dict[AgentID, Any]]
    ) -> Dict[AgentID, Dict[str, float]]:
        agent_res = {}
        for agent_result in agent_result_seq:
            for agent_id, result in agent_result.items():
                if agent_res.get(agent_id, None) is None:
                    agent_res[agent_id] = {
                        MetricType.REWARD: 0,
                    }
                if isinstance(result[MetricType.REWARD], MetricEntry):
                    e = result[MetricType.REWARD].value
                else:
                    e = result[MetricType.REWARD]
                agent_res[agent_id][MetricType.REWARD] += e / len(agent_result_seq)

        return agent_res

    def reset(self):
        self._episode_data = {
            MetricType.REWARD: defaultdict(lambda: []),
        }
        self._statistics = defaultdict(lambda: {MetricType.REWARD: 0.0})


class SC2Metric(SimpleMetrics):
    def __init__(self, agents: List[AgentID]):
        super(SC2Metric, self).__init__(agents)
        self._episode_data["battle_won"] = defaultdict(lambda: [])

    def step(self, agent_id, policy_id, **kwargs):
        super(SC2Metric, self).step(agent_id, policy_id, **kwargs)
        self._episode_data["battle_won"][agent_id].append(
            float(kwargs["info"].get("battle_won", False))
        )

    def parse(self, agent_filter=None) -> Dict[AgentID, Dict[str, MetricEntry]]:
        for item_key, agent_data in self._episode_data.items():
            # if filter is not None use filter else agents
            for aid in agent_filter or self._agents:
                if self._pids.get(aid) is not None:
                    prefix = f"{aid}/{self._pids[aid]}"
                else:
                    prefix = f"{aid}"
                if item_key == MetricType.REWARD:
                    self._statistics[aid][MetricType.REWARD] = MetricEntry(
                        value=sum(agent_data[aid]),
                        agg="mean",
                        tag=f"{prefix}/{MetricType.REWARD}",
                        log=True,
                    )
                if item_key == "battle_won":
                    self._statistics[aid]["battle_won"] = MetricEntry(
                        value=sum(agent_data[aid]),
                        agg="mean",
                        tag=f"{prefix}/battle_won",
                        log=True,
                    )
        return self._statistics


class GFootballMetric(SimpleMetrics):
    def __init__(self, agents: List[AgentID]):
        super(GFootballMetric, self).__init__(agents)
        self._episode_data["score"] = defaultdict(lambda: [])
        self._episode_data["win"] = defaultdict(lambda: [])
        self._episode_data["num_shot"] = defaultdict(lambda: [])
        self._episode_data["num_pass"] = defaultdict(lambda: [])
        self._episode_data["goal_diff"] = defaultdict(lambda: [])

    def step(self, agent_id, policy_id, **kwargs):
        super(GFootballMetric, self).step(agent_id, policy_id, **kwargs)
        score = kwargs.get("score", 0)
        self._episode_data["score"][agent_id].append(score)
        self._episode_data["goal_diff"][agent_id].append(kwargs["goal_diff"])
        self._episode_data["win"][agent_id].append(score == 1.0)
        action = kwargs["action"]
        self._episode_data["num_shot"][agent_id].append((action == 12).sum(-1))
        self._episode_data["num_pass"][agent_id].append(
            np.logical_and(action <= 11, action >= 9).sum(axis=-1)
        )


    def parse(self, agent_filter=None) -> Dict[AgentID, Dict[str, MetricEntry]]:
        for item_key, agent_data in self._episode_data.items():
            # if filter is not None use filter else agents
            for aid in agent_filter or self._agents:
                if self._pids.get(aid) is not None:
                    prefix = f"{aid}/{self._pids[aid]}"
                else:
                    prefix = f"{aid}"

                if item_key in [
                    MetricType.REWARD,
                    "score",
                    "win",
                    "num_shot",
                    "num_pass",
                    "goal_diff"
                ]:
                    # print(f"parse item_key: {item_key}, {len(agent_data[aid])}, {agent_data[aid][0].shape}", end=" ")
                    # input("??")
                    self._statistics[aid][item_key] = MetricEntry(
                        value=np.mean(sum(agent_data[aid])),
                        agg="mean",
                        tag=f"{prefix}/{item_key}",
                        log=True,
                    )
                    # print(f"value={self._statistics[aid][item_key].value}")
        return self._statistics


METRIC_TYPES = {
    "simple": SimpleMetrics,
    "jointdist": JointDistMetric,
    "sc2": SC2Metric,
    "grf": GFootballMetric,
}


def get_metric(metric_type: str):
    """Return a metric handler with given name.

    :param str metric_type: Registered metric type.
    """
    return METRIC_TYPES[metric_type]
