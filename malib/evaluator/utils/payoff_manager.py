# -*- coding: utf-8 -*-

import copy
import itertools
from typing import List, Union, Sequence, Dict, Tuple, Any

import nashpy as nash
import numpy as np

from malib import agent, settings
from malib.evaluator.utils.payoff_table import PayoffTable
from malib.gt.algos.elo_rating import EloManager
from malib.utils.logger import get_logger
from malib.utils.notations import deprecated
from malib.utils.typing import (
    AgentID,
    RolloutFeedback,
    PolicyID,
    PolicyConfig,
    MetricType,
)
from malib.gt.nash_solver.nash_solver import DefaultSolver, SelfPlaySolver
from threading import Lock


class PayoffManager:
    def __init__(
        self,
        agent_names: Sequence,
        exp_cfg: Dict[str, Any],
        solver=DefaultSolver(),
    ):
        """Create a payoff manager with agent names

        :param Sequence agent_names: a sequence of names which indicate players in the game
        :param str solve_method: the method used to solve the game, "fictitious_play" or "alpharank", default is "fictitious_play"
        """
        self.agents = agent_names
        self.num_player = len(agent_names)
        self.symmetric = self.num_player == 1
        self.solver = solver

        # a map for each player in which is a list
        self._policy = {an: [] for an in agent_names}
        self._policy_idx = {an: {} for an in agent_names}
        self._policy_config = {an: [] for an in agent_names}

        # table for each player
        if self.symmetric:
            self._only_agent = self.agents[0]
            self._fake_agents = ["team_0", "team_1"]
            self._payoff_tables = {
                self._only_agent: PayoffTable("SYMMETRIC TABLE", self._fake_agents)
            }
            print("PayoffTable: behave in SYMMETRIC ...")
        else:
            self._payoff_tables = {
                agent: PayoffTable(agent, self.agents) for agent in self.agents
            }

        # a list store equilibria, in which is a dict of the
        #  population distribution of each player
        self._equilibrium = {}
        self.logger = get_logger(
            log_level=settings.LOG_LEVEL,
            log_dir=settings.LOG_DIR,
            name="payoff_manager",
            remote=settings.USE_REMOTE_LOGGER,
            mongo=settings.USE_MONGO_LOGGER,
            **exp_cfg,
        )
        self._elo_manager = EloManager(K=16, default_elo=1000)
        self._lock = Lock()

    @property
    def payoffs(self):
        """
        :return: a copy of the payoff tables, which is a dict of PayoffTable objects.
        """
        return {aid: p_table.table.copy() for aid, p_table in self._payoff_tables.items()}

    def get_payoff_table(self):
        return list(self._payoff_tables.values())[0].table

    @property
    def equilibrium(self):
        return self._equilibrium

    def check_done(self, population_mapping: Dict):
        """Check whether all payoff values have been updated, a population_mapping
        will be hashed as a key to retrieve the simulation status table shared by
        related agents.

        :param Dict population_mapping: a dict of (agent_name, policy
        """

        # XXX(ming): another more efficient method is to check simulation done with
        #  sub-matrix extraction
        #  >>> policy_comb_idx = self._get_combination_index(policy_mapping)
        #  >>> all_done = np.alltrue(simulation[policy_comb_idx])
        all_done = True
        # print("---------- Is_Simlation_Done:", population_mapping)
        with self._lock:
            if self.symmetric:
                pid_seq = population_mapping[self._only_agent]
                all_done = self._payoff_tables[self._only_agent].is_simulation_done(
                    dict(zip(self._fake_agents, [pid_seq]*2))
                )
            else:
                for agent in population_mapping.keys():
                    all_done = self._payoff_tables[agent].is_simulation_done(population_mapping)
                    if not all_done:
                        break

            return all_done

    def get_rpp(self, population_mapping):
        eqb = self.get_equilibrium(population_mapping)
        def build_array(eqb, aid):
            return np.array([eqb[aid][pid] for pid in self._policy[aid]])
        
        if self.symmetric:
            eqb_1 = build_array(eqb, self._only_agent)
            fake_pop_map = dict(zip(self._fake_agents, [population_mapping[self._only_agent]]*2))
            payoff_mat = self._payoff_tables[self._only_agent][fake_pop_map]
            
            res1 = eqb_1 @ payoff_mat @ eqb_1
            res2 = eqb_1 @ (1 - payoff_mat) @ eqb_1
            return dict(zip(self._fake_agents, [res1, res2]))
        else:
            agent_1, agent_2 = self.agents
            eqb_1, eqb_2 = build_array(eqb, agent_1), build_array(eqb, agent_2)
            payoff_table_1, payoff_table_2 = (
                self._payoff_tables[agent_1],
                self._payoff_tables[agent_2],
            )
            res1 = eqb_1 @ payoff_table_1[population_mapping] @ eqb_2
            res2 = eqb_1 @ payoff_table_2[population_mapping] @ eqb_2

            return {agent_1: res1, agent_2: res2}

    def aggregate(
        self,
        equilibrium: Dict[AgentID, Dict[PolicyID, float]],
        brs: Dict[AgentID, PolicyID] = None,
    ) -> Dict[AgentID, float]:
        """Return weighted or nash payoff value"""

        res = {agent_id: 0.0 for agent_id in equilibrium}
        population_combination = {
            agent: list(e.keys()) for agent, e in equilibrium.items()
        }

        # retrieve partial payoff matrix
        if brs is None:
            res = {
                agent: self._payoff_tables[agent][population_combination]
                for agent in self.agents
            }  # self.get_selected_table(population_combination)
        else:
            # m*m*...*1*...*m
            for agent in brs.keys():
                tmp_comb = copy.copy(population_combination)
                # temporary replace the population of the ego agent
                # for computing the weighted payoff value: trainable policy vs. other agents
                tmp_comb[agent] = [brs[agent]]
                res[agent] = self._payoff_tables[agent][
                    tmp_comb
                ]  # self.get_selected_table(tmp_comb)

        # then aggregate the payoff matrix along axis
        weight_vectors = [
            np.asarray([list(equilibrium[agent].values())]) for agent in self.agents
        ]

        if brs is None:
            # in case of computing nash values
            weight_mat = np.asarray([[1.0]])
            for vector in weight_vectors:
                weight_mat = np.einsum("ij,j...->i...", vector.T, weight_mat)
                weight_mat = np.expand_dims(weight_mat, axis=0)
            weight_mat = np.squeeze(weight_mat, axis=0)
            weight_mat = np.squeeze(weight_mat, axis=-1)
            weight_mat = np.transpose(weight_mat)
            for agent in self.agents:
                assert weight_mat.shape == res[agent].shape, (
                    weight_mat.shape,
                    res[agent].shape,
                    equilibrium[agent],
                )
                res[agent] = (res[agent] * weight_mat).sum()
        else:
            # in case of computing
            # weight_mat = np.asarray([[1.0]])
            for agent in brs.keys():
                # ignore this one
                tmp = np.asarray([[1.0]])
                agent_axis = self.agents.index(agent)
                for i, vector in enumerate(weight_vectors):
                    if i == agent_axis:
                        continue
                    tmp = np.einsum("ij,j...->i...", vector.T, tmp)
                    tmp = np.expand_dims(tmp, axis=0)
                tmp = np.squeeze(tmp, axis=-1)
                tmp = np.squeeze(tmp, axis=0)
                tmp = np.transpose(tmp)
                tmp = np.expand_dims(tmp, axis=agent_axis)
                assert tmp.shape == res[agent].shape, (
                    tmp.shape,
                    res[agent].shape,
                    equilibrium[agent],
                    i,
                    tmp_comb,
                    agent,
                )
                res[agent] = (res[agent] * tmp).sum()
                # weight_mat = np.einsum("ij,j...->i...", weight_vectors[i].T, weight_mat)
                # weight_mat = np.expand_dims(weight_mat, axis=0)

        return res

    def update_payoff(self, content: RolloutFeedback):
        """Update the payoff table, and set the corresponding simulation_flag to True

        :param RolloutFeedback content: a RolloutFeedback with policy_combination that specifies the entry to update
         and statistics which stores the value to update
        """
        with self._lock:
            pop_comb = {
                agent: pid for agent, (pid, _) in content.policy_combination.items()
            }
            # print("------------- Update_Payoff:", pop_comb, content.statistics)
            
            if self.symmetric:
                _payoff_table = self._payoff_tables[self._only_agent]
                score = content.statistics[self._only_agent]["score"]
                gd = content.statistics[self._only_agent]["goal_diff"]
                if len(set(pop_comb.values())) > 1:
                    reversed_pop_comb = dict(zip(pop_comb.keys(), reversed(list(pop_comb.values()))))
                    _payoff_table[reversed_pop_comb] = -gd
                    _payoff_table.set_simulation_done(reversed_pop_comb)
                    scores = {
                        pid: score if aid == self._only_agent else 1 - score
                        for aid, pid in pop_comb.items()
                    }
                    self._elo_manager.record_new_match_result(scores, content.episode_cnt)
                
                _payoff_table[pop_comb] = gd
                _payoff_table.set_simulation_done(pop_comb)
                
                
            else:
                scores = {}
                for agent in self.agents:
                    self._payoff_tables[agent][pop_comb] = content.statistics[
                        agent
                    ][MetricType.REWARD]
                    self._payoff_tables[agent].set_simulation_done(pop_comb)
                    # self._done_table[agent][population_combination] = True

                    scores[agent + "_" + pop_comb[agent]] = content.statistics[
                        agent
                    ]["score"]
                self._elo_manager.record_new_match_result(scores, content.episode_cnt)


    def record_new_match_result(self, scores, episode_cnt):
        self._elo_manager.record_new_match_result(scores, episode_cnt)

    def get_elo(self, population_mapping):
        ans = {}
        for aid, p_seq in population_mapping.items():
            ans[aid] = {}
            for pid in p_seq:
                record_name = pid if self.symmetric else aid + "_" + pid
                ans[aid][pid] = self._elo_manager[record_name]
        ans["built_in_ai"] = self._elo_manager["built_in_ai"]
        return ans

    @deprecated
    def _add_matchup_result(
        self,
        policy_combination: List[Tuple[PolicyID, PolicyConfig]],
        payoffs: Union[List, np.ndarray],
    ):
        """
        add payoffs to each table, call it only after self._expand_table
        """
        policy_mapping: List[PolicyID] = [p_tuple[0] for p_tuple in policy_combination]
        idx2add = self._get_combination_index(policy_mapping)
        # self.[idx2add] = 1

        for i, a_name in enumerate(self.agents):
            # self._payoff_tables[a_name][idx2add] = payoffs[i]
            self._payoff_tables[a_name][policy_combination] = payoffs[i]

    def compute_equilibrium(
        self, population_mapping: Dict[PolicyID, Sequence[PolicyID]]
    ) -> Dict[PolicyID, Dict[PolicyID, float]]:
        """Compute nash equilibrium of given populations

        :param Dict[PolicyID,Sequence[PolicyID]] population_mapping: a dict from agent_name to a sequence of policy ids
        :return: the nash equilibrium which is a dict from agent_name to a dict from policy id to float
        """
        # sub_payoff_matrix = self.get_selected_table(population_combination)
        if self.symmetric:
            fake_pop_map = dict(zip(self._fake_agents, [population_mapping[self._only_agent]]*2))
            _sub_payoff_mat = self._payoff_tables[self._only_agent][fake_pop_map]
            sub_payoff_matrix = [_sub_payoff_mat, -_sub_payoff_mat]
        else:
            sub_payoff_matrix = [
                self._payoff_tables[agent][population_mapping] for agent in self.agents
            ]
        # print("Compute NE, payoff matrix", sub_payoff_matrix)
        if sub_payoff_matrix[0].shape == (1, 1):
            res = {
                agent: dict(zip(p, [1 / max(1, len(p))] * len(p)))
                for agent, p in population_mapping.items()
            }
        else:
            eps = self.solver.solve(sub_payoff_matrix)
            dist = [e.tolist() for e in eps]

            res = {
                agent: dict(zip(p, dist[i]))
                for i, (agent, p) in enumerate(population_mapping.items())
            }
        return res

    def update_equilibrium(
        self,
        population_mapping: Dict[PolicyID, Sequence[PolicyID]],
        eqbm: Dict[PolicyID, Dict[PolicyID, float]],
    ):
        """Update the equilibrium of certain population mapping in the payoff table
        :param Dict[PolicyID,Sequence[PolicyID]] population_mapping: a dict from agent_name to a sequence of policy ids
        :param Dict[PolicyID,Dict[PolicyID,float]] eqbm: the nash equilibrium which is a dict from agent_name to a dict from policy id to float
        """
        hash_key = self._hash_population_mapping(population_mapping)
        self._equilibrium[hash_key] = eqbm.copy()

    def get_equilibrium(
        self, population_mapping: Dict[AgentID, Sequence[PolicyID]]
    ) -> Dict[AgentID, Dict[PolicyID, Union[float, np.ndarray]]]:
        """Get the equilibrium stored in the payoff manager

        :param Dict[AgentID,Sequence[PolicyID]] population_mapping: a dict from agent_name to a sequence of policy ids
        :return: the nash equilibrium which is a dict from agent_name to a dict from policy id to float
        >>> eqbm = {"player_0": {"policy_0": 1.0, "policy_1": 0.0}, "player_1": {"policy_0": 0.3, "policy_1": 0.7}}
        >>> population_mapping = {"player_0": ["policy_0", "policy_1"], "player_1": ["policy_0", "policy_1"]}
        >>> self.update_equilibrium(population_mapping, eqbm)
        >>> self.get_equilibrium(population_mapping)
        ... {"player_0": {"policy_0": 1.0, "policy_1": 0.0}, "player_1": {"policy_0": 0.3, "policy_1": 0.7}}
        """
        hash_key = self._hash_population_mapping(population_mapping)
        agent = list(population_mapping.keys())[0]
        assert hash_key in self._equilibrium, (
            hash_key,
            self._equilibrium.keys(),
            self._payoff_tables[agent].table.shape,
            self._payoff_tables[agent].table,
        )
        eq = self._equilibrium[hash_key]

        return eq.copy()

    def _hash_population_mapping(
        self, population_mapping: Dict[PolicyID, Sequence[PolicyID]]
    ) -> str:
        """
        currently make it to a string
        """
        sorted_mapping = {}
        ans = ""
        for an in self.agents:
            ans += an + ":"
            sorted_mapping[an] = sorted(population_mapping[an])
            for pid in sorted_mapping[an]:
                ans += pid + ","
            ans += ";"
        return ans

    def _get_pending_matchups(
        self, agent_name: AgentID, policy_id: PolicyID, policy_config: Dict[str, Any]
    ) -> List[Dict]:
        """Generate match description with policy combinations"""

        agent_policy_list = []
        if self.symmetric:
            agent_policy_list.append([(policy_id, policy_config)])
            agent_policy_list.append(list(zip(self._policy[agent_name], self._policy_config[agent_name])))
        else:
            for an in self.agents:
                if an == agent_name:
                    agent_policy_list.append([(policy_id, policy_config)])
                else:
                    # skip empty policy
                    if len(self._policy[an]) == 0:
                        continue
                    # append all other agent policies
                    agent_policy_list.append(
                        list(zip(self._policy[an], self._policy_config[an]))
                    )

        # if other agents has no available policies, return an empty list
        if len(agent_policy_list) < len(self.agents):
            return []

        pending_comb_list = [comb for comb in itertools.product(*agent_policy_list)]
        return [
            {an: pending_comb[i] for i, an in enumerate(self._fake_agents if self.symmetric else self.agents)}
            for pending_comb in pending_comb_list
        ]

    def get_pending_matchups(
        self, agent_name: AgentID, policy_id: PolicyID, policy_config: Dict[str, Any]
    ) -> List[Dict]:
        """Add a new policy for an agent and get the needed matches.

        :param AgentID agent_name: the agent name for which a new policy will be added
        :param PolicyID policy_id: the policy to be added
        :param Dict[str,Any] policy_config: the config of the added policy
        :return: a list of match combinations, each is a dict from agent_name to a tuple of policy_id and policy_config
        """
        with self._lock:
            print(f"Add Policy, {agent_name}, {policy_id}")
            if policy_id in self._policy[agent_name]:
                return []

            # May have some problems for concurrent version, but we have no demand for a concurrent payoff table ...
            self._policy_idx[agent_name][policy_id] = len(self._policy[agent_name])
            self._policy[agent_name].append(policy_id)
            self._policy_config[agent_name].append(policy_config)
            ans = self._get_pending_matchups(agent_name, policy_id, policy_config)
        
        # print("pending combs:", ans)

        return ans
