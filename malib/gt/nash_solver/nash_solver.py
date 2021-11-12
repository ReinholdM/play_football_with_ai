# -*- coding: utf-8 -*-
from malib.gt.nash_solver import fictitious_play
from open_spiel.python.egt import alpharank, utils as alpharank_utils
import numpy as np
from abc import ABCMeta, abstractmethod


class Solver(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def solve(self, payoff_seq):
        raise NotImplementedError


class SelfPlaySolver(Solver):
    def __init__(self):
        super(SelfPlaySolver, self).__init__()

    def build_solution_array(self, payoff_seq):
        payoff = payoff_seq[0]
        for i in range(len(payoff_seq) - 1):
            assert payoff.shape == payoff_seq[i + 1].shape
        return [np.zeros(width) for width in payoff.shape]

    def solve(self, payoff_seq):
        array_list = self.build_solution_array(payoff_seq)
        for array in array_list:
            array[-1] = 1.0

        return array_list


class FictitiousSelfPlaySolver(SelfPlaySolver):
    def __init__(self):
        super(FictitiousSelfPlaySolver, self).__init__()

    def solve(self, payoff_seq):
        array_list = self.build_solution_array(payoff_seq)
        for i, array in enumerate(array_list):
            array_list[i] += 1 / len(array)

        return array_list


class DefaultSolver(Solver):
    """A Solver to find certain solution concept, e.g. nash equilibrium."""

    def __init__(self):
        super(DefaultSolver, self).__init__()

    def fictitious_play(self, payoffs_seq):
        """solve the game with fictitious play, only suppoort 2-player games

        :param payoffs_seq: a sequence of the game's payoff matrix, which can be of length one or two, when of length one, just as take [M, -M] as input
        :return: the nash equilirium computed by fictious play, which order is corresponding to *payoff_seq*
        """

        *_, eqs = iter(fictitious_play(*payoffs_seq, 10000))
        eqs = [tuple(map(lambda x: x / np.sum(x), eqs))]
        return eqs[0]

    def alpharank(self, payoffs_seq):
        """Use alpharank to solve the game, for more details, you can check https://github.com/deepmind/open_spiel/blob/master/docs/alpha_rank.md

        :param payoffs_seq: a sequence of empirical payoffs
        :return: the solution computed by alpharank, which is a sequnce of np.ndarray of probability in each population
        """

        def remove_epsilon_negative_probs(probs, epsilon=1e-9):
            """Removes negative probabilities that occur due to precision errors."""
            if len(probs[probs < 0]) > 0:  # pylint: disable=g-explicit-length-test
                # Ensures these negative probabilities aren't large in magnitude, as that is
                # unexpected and likely not due to numerical precision issues
                print("Probabilities received were: {}".format(probs[probs < 0]))
                assert np.alltrue(
                    np.min(probs[probs < 0]) > -1.0 * epsilon
                ), "Negative Probabilities received were: {}".format(probs[probs < 0])

                probs[probs < 0] = 0
                probs = probs / np.sum(probs)
            return probs

        def get_alpharank_marginals(payoff_tables, pi):
            """Returns marginal strategy rankings for each player given joint rankings pi.

            Args:
              payoff_tables: List of meta-game payoff tables for a K-player game, where
                each table has dim [n_strategies_player_1 x ... x n_strategies_player_K].
                These payoff tables may be asymmetric.
              pi: The vector of joint rankings as computed by alpharank. Each element i
                corresponds to a unique integer ID representing a given strategy profile,
                with profile_to_id mappings provided by
                alpharank_utils.get_id_from_strat_profile().

            Returns:
              pi_marginals: List of np.arrays of player-wise marginal strategy masses,
                where the k-th player's np.array has shape [n_strategies_player_k].
            """
            num_populations = len(payoff_tables)

            if num_populations == 1:
                return pi
            else:
                num_strats_per_population = (
                    alpharank_utils.get_num_strats_per_population(
                        payoff_tables, payoffs_are_hpt_format=False
                    )
                )
                num_profiles = alpharank_utils.get_num_profiles(
                    num_strats_per_population
                )
                pi_marginals = [np.zeros(n) for n in num_strats_per_population]
                for i_strat in range(num_profiles):
                    strat_profile = alpharank_utils.get_strat_profile_from_id(
                        num_strats_per_population, i_strat
                    )
                    for i_player in range(num_populations):
                        pi_marginals[i_player][strat_profile[i_player]] += pi[i_strat]
                return pi_marginals

        joint_distr = alpharank.sweep_pi_vs_epsilon(payoffs_seq)
        joint_distr = remove_epsilon_negative_probs(joint_distr)
        marginals = get_alpharank_marginals(payoffs_seq, joint_distr)

        return marginals

    def solve(self, payoffs_seq):
        if len(payoffs_seq) <= 2:  # or self._solve_method == "fictitious_play":
            return self.fictitious_play(payoffs_seq)
        elif len(payoffs_seq) > 2:  # or self._solve_method == "alpharank":
            return self.alpharank(payoffs_seq)
