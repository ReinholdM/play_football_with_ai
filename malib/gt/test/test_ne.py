# -*- coding: utf-8 -*-
from malib.gt.nash_solver.nash_solver import compute_two_player_nash
from malib.evaluator.utils.payoff_manager import PayoffManager
import pytest
import itertools
import numpy as np


@pytest.fixture
def agent_names():
    return ["player_0", "player_1"]


@pytest.fixture
def payoff_manager(agent_names):
    return PayoffManager(agent_names)


@pytest.fixture
def policy_to_be_added(agent_names):
    policys = ["r", "p", "s"]
    return {an: policys for an in agent_names}


@pytest.fixture
def target_payoff_table():
    return np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])


def test_payoff_table_add_policy(payoff_manager, policy_to_be_added):
    for an, policies in policy_to_be_added.items():
        for policy in policies:
            pending_list = payoff_manager.add_policy(an, policy)
            print(pending_list)


def test_update_payoff(payoff_manager, policy_to_be_added, target_payoff_table):
    test_payoff_table_add_policy(payoff_manager, policy_to_be_added)
    A = target_payoff_table
    B = -A
    ps = ["r", "p", "s"]
    for i1, i2 in itertools.product(range(len(ps)), repeat=2):
        print(f"add {[ps[i1], ps[i2]]}: {[A[i1, i2], B[i1, i2]]} to payoff table")
        payoff_manager._add_matchup_result([ps[i1], ps[i2]], [A[i1, i2], B[i1, i2]])
    print(payoff_manager.payoffs)


@pytest.fixture
def select_policy_mapping(agent_names, policy_to_be_added):
    return {an: policy_to_be_added[an][:2] for an in agent_names}


def test_get_selected_table(
    agent_names,
    payoff_manager,
    policy_to_be_added,
    target_payoff_table,
    select_policy_mapping,
):
    test_update_payoff(payoff_manager, policy_to_be_added, target_payoff_table)
    print(select_policy_mapping)
    subpayoff = payoff_manager.get_selected_table(select_policy_mapping)
    print(subpayoff)


def test_calc_ne(
    agent_names,
    payoff_manager,
    policy_to_be_added,
    target_payoff_table,
    select_policy_mapping,
):
    test_update_payoff(payoff_manager, policy_to_be_added, target_payoff_table)

    # payoffs = [payoff_manager.payoffs[an] for an in agent_names]
    # eq = compute_two_player_nash(*payoffs)
    # payoff_manager.update_equilibrium(eq[0])

    payoffs = payoff_manager.get_selected_table(select_policy_mapping)
    eqs = compute_two_player_nash(*payoffs.values())
    eqs = eqs[0]
    eqb = {an: eqs[i] for i, an in enumerate(payoffs.keys())}
    payoff_manager.update_equilibrium(select_policy_mapping, eqb)
    print(payoff_manager.get_equilibrium(select_policy_mapping))

    select_policy_mapping = {an: ["r", "p", "s"] for an in agent_names}

    print("++++++++++++++++++++++++++++++++")
    print(payoff_manager.equilibrium)
    print(payoff_manager.get_equilibrium(select_policy_mapping))

    print("+++++++++++++++++++++++++++++++++++++")

    payoffs = payoff_manager.get_selected_table(select_policy_mapping)
    eqs = compute_two_player_nash(*payoffs.values())
    eqs = eqs[0]
    eqb = {an: eqs[i] for i, an in enumerate(payoffs.keys())}
    payoff_manager.update_equilibrium(select_policy_mapping, eqb)
    print(payoff_manager.get_equilibrium(select_policy_mapping))
    print(payoff_manager.equilibrium)
