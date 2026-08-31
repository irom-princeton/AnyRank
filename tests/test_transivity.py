"""Regression tests for AnyRank transitive closure handling."""

from itertools import combinations

from multitest.transivity import (
    transitive_closure_relations,
    transitive_decision_times,
)


def test_noisy_cycle_quarantines_all_implied_edges():
    decisions = {(0, 1): 1, (1, 2): 1, (2, 3): 1, (0, 3): -1}

    result = transitive_closure_relations(
        list(decisions),
        decisions,
        n_policies=4,
    )

    all_pairs = set(combinations(range(4), 2))
    assert result["contradictions"] == all_pairs
    assert result["resolved"] == set()
    assert result["unresolved"] == all_pairs
    assert result["implied"] == set()
    assert all(a != b for a, b in result["closure"])


def test_clean_chain_reports_safe_implied_edges():
    decisions = {(0, 1): 1, (1, 2): 1, (2, 3): 1}

    result = transitive_closure_relations(
        list(decisions),
        decisions,
        n_policies=4,
    )

    assert result["contradictions"] == set()
    assert result["resolved"] == set(combinations(range(4), 2))
    assert result["unresolved"] == set()
    assert result["implied"] == {(0, 2), (1, 3), (0, 3)}
    assert all(a != b for a, b in result["implied"])


def test_disputed_edge_does_not_create_implied_edge():
    decisions = {(0, 1): 1, (1, 0): 1, (1, 2): 1}

    result = transitive_closure_relations(
        list(decisions),
        decisions,
        n_policies=3,
    )

    assert result["contradictions"] == {(0, 1)}
    assert result["resolved"] == {(1, 2)}
    assert result["unresolved"] == {(0, 1), (0, 2)}
    assert result["implied"] == set()


def test_decision_times_quarantine_noisy_cycle():
    decisions = {(0, 1): 1, (1, 2): 1, (2, 3): 1, (0, 3): -1}
    times = {(0, 1): 10, (1, 2): 20, (2, 3): 30, (0, 3): 40}

    result = transitive_decision_times(times, decisions, n_policies=4)

    all_pairs = set(combinations(range(4), 2))
    assert set(result) == all_pairs
    assert {pair for pair, values in result.items() if values["direction"]} == set()
    assert all(values["earliest_time"] is None for values in result.values())
    assert result[(0, 3)]["direct_time"] == 40


def test_decision_times_keep_clean_chain_implied_timings():
    decisions = {(0, 1): 1, (1, 2): 1, (2, 3): 1}
    times = {(0, 1): 10, (1, 2): 20, (2, 3): 30}

    result = transitive_decision_times(times, decisions, n_policies=4)

    assert result[(0, 2)] == {
        "direction": 1,
        "earliest_time": 20,
        "direct_time": None,
        "saved_trials": 0,
    }
    assert result[(1, 3)] == {
        "direction": 1,
        "earliest_time": 30,
        "direct_time": None,
        "saved_trials": 0,
    }
    assert result[(0, 3)] == {
        "direction": 1,
        "earliest_time": 30,
        "direct_time": None,
        "saved_trials": 0,
    }
    assert result[(0, 1)]["direct_time"] == 10
    assert result[(1, 2)]["direct_time"] == 20
    assert result[(2, 3)]["direct_time"] == 30


def test_decision_times_populate_reverse_direct_time():
    decisions = {(1, 0): 1}
    times = {(1, 0): 7}

    result = transitive_decision_times(times, decisions, n_policies=2)

    assert result[(0, 1)] == {
        "direction": -1,
        "earliest_time": 7,
        "direct_time": 7,
        "saved_trials": 0,
    }
