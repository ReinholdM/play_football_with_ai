import numpy as np

from malib.gt.algos.elo_rating import EloManager
from malib.gt.nash_solver import DefaultSolver
from malib.utils.typing import RolloutFeedback, MetricType, AgentID, PolicyID
from malib.evaluator.utils.payoff_manager import PayoffManager
from typing import Dict, Sequence, Callable, List, Any, Union


class TeamPayoffManager:
    def __init__(
        self, team_config: Dict[str, Sequence], exp_cfg, solver=DefaultSolver()
    ):
        """Constructor
        :param team_config: a dict of team name to a list of player names to determine the team information
            e.g. {"team_0": ["p1_0", "p2_0"], "team_1": ["p3_0", "p4_0"]}
            ps: the last part after "_" indicates the number of generations
        """
        self._team_config = team_config
        self._player2team = {}
        for tid, player_ids in team_config.items():
            for player_id in player_ids:
                self._player2team[player_id] = tid

        self._teams = list(sorted(team_config.keys()))
        self._payoff_manager = PayoffManager(self._teams, exp_cfg, solver)
        self._elo_manager = EloManager(K=16, default_elo=1000)

        # a buffer to store the pending request
        # and when the buffer size is enough, flush and return pending matchups
        self._get_pending_cmd_buffer = {team_id: {} for team_id in self._teams}
        self._generations = {team_id: {} for team_id in self._teams}

    def get_elo(self, population_mapping):
        team_population_mapping = self._build_generation_from_underlying_population(
            population_mapping
        )
        res = {team_id: {} for team_id in team_population_mapping}
        for team_id, gen_seq in team_population_mapping.items():
            for gen in gen_seq:
                res[team_id][gen] = self._elo_manager[
                    self._build_elo_name(team_id, gen)
                ]

        return res

    def get_rpp(self, population_mapping):
        team_population_mapping = self._build_generation_from_underlying_population(
            population_mapping
        )
        return self._payoff_manager.get_rpp(team_population_mapping)

    def get_pending_matchups(
        self, agent_name: AgentID, policy_id: PolicyID, policy_config: Dict[str, Any]
    ) -> List[Dict]:
        team_name = self._player2team[agent_name]
        self._get_pending_cmd_buffer[team_name][agent_name] = (policy_id, policy_config)
        if len(self._get_pending_cmd_buffer[team_name]) == len(
            self._team_config[team_name]
        ):
            generation_set = set(
                pid.split("_")[-1]
                for pid, _ in self._get_pending_cmd_buffer[team_name].values()
            )
            assert len(generation_set) == 1, generation_set
            generation = list(generation_set)[0]
            self._generations[team_name][generation] = self._get_pending_cmd_buffer[
                team_name
            ]
            self._get_pending_cmd_buffer[team_name] = {}

            team_pending_matchups = self._payoff_manager.get_pending_matchups(
                team_name, generation, {}
            )

            pending_matchups = []
            for team_comb in team_pending_matchups:
                comb_list = [
                    self._generations[team_name][generation]
                    for team_name, (generation, _) in team_comb.items()
                ]
                comb = {}
                for _comb in comb_list:
                    comb.update(_comb)
                pending_matchups.append(comb)

            return pending_matchups
        else:
            return []

    def aggregate(
        self,
        equilibrium: Dict[AgentID, Dict[PolicyID, float]],
        brs: Dict[AgentID, PolicyID] = None,
    ) -> Dict[AgentID, float]:
        return None

    def check_done(self, population_mapping):
        team_population_mapping = self._build_generation_from_underlying_population(
            population_mapping
        )
        return self._payoff_manager.check_done(team_population_mapping)

    def update_payoff(self, content: RolloutFeedback):
        population_combination = {
            agent: pid for agent, (pid, _) in content.policy_combination.items()
        }

        # FIXME(ziyu): may have bugs?
        team_population_mapping = self._build_generation_from_underlying_population(
            population_combination
        )

        # (ziyu:) for football, let it be the score value, which is win_rate + 0.5 * draw_rate
        team_stats = {}
        team_scores = {}
        print(content.statistics)
        for agent_name, stats in content.statistics.items():
            team_name = self._player2team[agent_name]
            team_stats[team_name] = {
                MetricType.REWARD: stats[MetricType.REWARD],
            }
            elo_name = self._build_elo_name(
                team_name, team_population_mapping[team_name][0]
            )
            team_scores[elo_name] = stats["score"]
            if len(team_stats) == len(self._teams):
                break

        print("Record:", team_scores, content.episode_cnt)
        self._elo_manager.record_new_match_result(
            team_scores, iter_cnt=content.episode_cnt
        )

        team_rollout_feed_back = RolloutFeedback(
            agent_involve_info=None,
            statistics=team_stats,
            policy_combination={
                team_id: (generation, {})
                for team_id, generation in team_population_mapping.items()
            },
            worker_idx="",
        )

        self._payoff_manager.update_payoff(team_rollout_feed_back)

    def compute_equilibrium(
        self, population_mapping: Dict[PolicyID, Sequence[PolicyID]]
    ) -> Dict[PolicyID, Dict[PolicyID, float]]:
        team_population_mapoping = self._build_generation_from_underlying_population(
            population_mapping
        )
        team_eqbm = self._payoff_manager.compute_equilibrium(team_population_mapoping)
        return self._build_eqbm_from_team(team_eqbm)

    def get_equilibrium(
        self, population_mapping: Dict[AgentID, Sequence[PolicyID]]
    ) -> Dict[AgentID, Dict[PolicyID, Union[float, np.ndarray]]]:
        team_population_mapoping = self._build_generation_from_underlying_population(
            population_mapping
        )
        team_eqbm = self._payoff_manager.get_equilibrium(team_population_mapoping)
        return self._build_eqbm_from_team(team_eqbm)

    def update_equilibrium(
        self,
        population_mapping: Dict[PolicyID, Sequence[PolicyID]],
        eqbm: Dict[PolicyID, Dict[PolicyID, float]],
    ):
        team_population_mapoping = self._build_generation_from_underlying_population(
            population_mapping
        )
        team_eqbm = {}
        for player_id, eqbm_p in eqbm.items():
            team_name = self._player2team[player_id]
            if team_name in team_eqbm:
                continue
            else:
                team_eqbm[team_name] = {}
            for pid, eqbm in eqbm_p.items():
                generation = pid.split("_")[-1]
                team_eqbm[team_name][generation] = eqbm
            if len(team_eqbm) == len(self._teams):
                break

        self._payoff_manager.update_equilibrium(team_population_mapoping, team_eqbm)

    def _build_elo_name(self, team_id, gen):
        return team_id + "_gen_" + gen

    def _build_eqbm_from_team(self, team_eqbm):
        population_eqbm = {player_id: {} for player_id in self._player2team}
        for team, eqbm_t in team_eqbm.items():
            for gen, prob_t in eqbm_t.items():
                for player_id, (pid, _) in self._generations[team][gen].items():
                    population_eqbm[player_id][pid] = prob_t

        return population_eqbm

    def _build_generation_from_underlying_population(self, population_mapping):
        # FIXME(ziyu): I only consider the very specific case for football
        #  team name should be the prefix of player name
        #  and player name's last number (after "_") must be the generation number
        team_policies = {team_id: set() for team_id in self._teams}

        for player_id, pid_seq in population_mapping.items():
            team_id = self._player2team[player_id]
            if type(pid_seq) is list:
                pid_seq = set(pid_seq)
            elif type(pid_seq) is str:
                pid_seq = {pid_seq}
            generation_seq = set(pid.split("_")[-1] for pid in pid_seq)
            team_policies[team_id] = list(generation_seq)

        return team_policies


if __name__ == "__main__":
    team_config = {
        f"team_{j}": [f"team_{j}_player_{i}" for i in range(5)] for j in range(2)
    }
    players = [f"team_{j}_player_{i}" for i in range(5) for j in range(2)]
    tm = TeamPayoffManager(team_config, {})
    policies = {
        pid: (f"MAPPO_{i}_0", f"MAPPO_{i}_0_cfg") for i, pid in enumerate(players)
    }

    combs = []
    for player, (pid, pcfg) in policies.items():
        combs.extend(tm.get_pending_matchups(player, pid, pcfg))

    rollout_feed_backs = [
        RolloutFeedback(
            agent_involve_info=None,
            policy_combination=comb,
            statistics={player_id: {MetricType.REWARD: 0.5} for player_id in players},
            worker_idx="",
        )
        for comb in combs
    ]

    for rfb in rollout_feed_backs:
        tm.update_payoff(rfb)
    p_mp = {player_id: [f"MAPPO_{i}_0"] for i, player_id in enumerate(players)}
    eqbm = tm.compute_equilibrium(p_mp)
    tm.update_equilibrium(p_mp, eqbm)
    print(tm.get_equilibrium(p_mp))
    policies = {
        pid: (f"MAPPO_{i}_1", f"MAPPO_{i}_1_cfg")
        for i, pid in enumerate(team_config["team_0"])
    }
    combs = []
    for player, (pid, pcfg) in policies.items():
        combs.extend(tm.get_pending_matchups(player, pid, pcfg))

    rollout_feed_backs = [
        RolloutFeedback(
            agent_involve_info=None,
            policy_combination=comb,
            statistics={player_id: {MetricType.REWARD: 0.5} for player_id in players},
            worker_idx="",
        )
        for comb in combs
    ]

    for rfb in rollout_feed_backs:
        tm.update_payoff(rfb)

    policies = {
        pid: (f"MAPPO_{i}_1", f"MAPPO_{i}_1_cfg")
        for i, pid in enumerate(team_config["team_1"])
    }
    combs = []
    for player, (pid, pcfg) in policies.items():
        combs.extend(tm.get_pending_matchups(player, pid, pcfg))

    rollout_feed_backs = [
        RolloutFeedback(
            agent_involve_info=None,
            policy_combination=comb,
            statistics={player_id: {MetricType.REWARD: 0.5} for player_id in players},
            worker_idx="",
        )
        for comb in combs
    ]

    for rfb in rollout_feed_backs:
        tm.update_payoff(rfb)
    p_mp = {
        player_id: [f"MAPPO_{i}_0", f"MAPPO_{i}_1"]
        for i, player_id in enumerate(players)
    }
    print(tm.compute_equilibrium(p_mp))
    print(tm.check_done(p_mp))
