"""Regression tests for AnyRank transitive closure handling."""

from itertools import combinations

from multitest.transivity import transitive_closure_relations


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
