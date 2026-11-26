"""Regression tests for transitive closure inside the graphical rejection loop.

Both graphical runners must record implied comparisons in the direction given
by the transitive closure, and must include the rejection being processed when
computing that closure. Hypotheses use the canonical orientation (a, b) with
a < b; a stored direction of +1 means mu_a < mu_b and -1 means mu_a > mu_b.
"""

from itertools import combinations

import pytest

from multitest.sequential_graphical import (
    SequentialGraphicalTest as PassiveSequentialGraphicalTest,
)
from multitest.sequential_graphical_active import (
    SequentialGraphicalTest as ActiveSequentialGraphicalTest,
)

RUNNERS = pytest.mark.parametrize(
    "runner_cls",
    [PassiveSequentialGraphicalTest, ActiveSequentialGraphicalTest],
    ids=["passive", "active"],
)
DECISIVE = 1e-4
UNDECIDED = 0.9


def _run_graphical_step(runner_cls, num_policies, decided, p_value_types,
                        preset_directions=True):
    """Run one graphical rejection step with transitivity enabled.

    Args:
        decided: Mapping from canonical hypothesis to its true direction (+1 or
            -1). These hypotheses receive decisive p-values; all others are
            undecided.
        p_value_types: Mapping from canonical hypothesis to its p-value type.
        preset_directions: If True, decided hypotheses enter with their
            direction already recorded, as when the pairwise test has decided
            at the current alpha. If False, every direction starts unset, as
            when recycled alpha makes a hypothesis rejectable mid-step.
    """
    hypotheses = list(combinations(range(num_policies), 2))
    sgt = runner_cls(
        num_policies=num_policies,
        null_hypotheses=[(0.5, 0.5)] * len(hypotheses),
        total_alpha=0.05,
    )
    p_values = [DECISIVE if h in decided else UNDECIDED for h in hypotheses]
    hypotheses_correct = {
        h: (float(decided[h]) if preset_directions and h in decided else 0.0)
        for h in hypotheses
    }
    types = {i: p_value_types[h] for i, h in enumerate(hypotheses)}
    rejected, _graphs, _alpha_at_rejected, directions = (
        sgt.sequential_graphical_test(
            p_values,
            [],
            hypotheses_correct,
            hypotheses,
            num_policies,
            types,
            allow_transitive=True,
        )
    )
    return hypotheses, rejected, directions


@RUNNERS
@pytest.mark.parametrize("implied_p_type", ["alternative", "null", "N/A"])
def test_implied_direction_ignores_undecided_pair_p_type(
    runner_cls, implied_p_type,
):
    # 0 < 1 and 1 < 2 are decided; they imply 0 < 2 whatever the (0, 2)
    # test's own undecided p-value type suggests.
    decided = {(0, 1): 1, (1, 2): 1}
    p_value_types = {(0, 1): "alternative", (1, 2): "alternative",
                     (0, 2): implied_p_type}

    hypotheses, rejected, directions = _run_graphical_step(
        runner_cls, 3, decided, p_value_types,
    )

    assert rejected == set(range(len(hypotheses)))
    assert directions == {(0, 1): 1.0, (0, 2): 1.0, (1, 2): 1.0}


@RUNNERS
@pytest.mark.parametrize("implied_p_type", ["alternative", "null", "N/A"])
def test_implied_reverse_direction_ignores_undecided_pair_p_type(
    runner_cls, implied_p_type,
):
    # 1 < 0 and 2 < 1 are decided; they imply 2 < 0, stored as (0, 2) = -1.
    decided = {(0, 1): -1, (1, 2): -1}
    p_value_types = {(0, 1): "null", (1, 2): "null", (0, 2): implied_p_type}

    hypotheses, rejected, directions = _run_graphical_step(
        runner_cls, 3, decided, p_value_types,
    )

    assert rejected == set(range(len(hypotheses)))
    assert directions == {(0, 1): -1.0, (0, 2): -1.0, (1, 2): -1.0}


@RUNNERS
def test_closure_includes_rejection_without_preset_direction(runner_cls):
    # No direction is recorded before the step, so the rejection being
    # processed must have its direction stored before the closure is computed.
    decided = {(0, 1): 1, (1, 2): 1}
    p_value_types = {(0, 1): "alternative", (1, 2): "alternative",
                     (0, 2): "N/A"}

    hypotheses, rejected, directions = _run_graphical_step(
        runner_cls, 3, decided, p_value_types, preset_directions=False,
    )

    assert rejected == set(range(len(hypotheses)))
    assert directions == {(0, 1): 1.0, (0, 2): 1.0, (1, 2): 1.0}


def _four_policy_chain(runner_cls, preset_directions):
    # 0 < 1 < 2 < 3 decided on adjacent pairs; undecided pairs carry adverse
    # or missing p-value types.
    decided = {(0, 1): 1, (1, 2): 1, (2, 3): 1}
    p_value_types = {h: "alternative" for h in decided}
    p_value_types.update({(0, 2): "null", (0, 3): "null", (1, 3): "N/A"})

    hypotheses, rejected, directions = _run_graphical_step(
        runner_cls, 4, decided, p_value_types,
        preset_directions=preset_directions,
    )

    assert rejected == set(range(len(hypotheses)))
    assert directions == {h: 1.0 for h in hypotheses}


@RUNNERS
def test_four_policy_chain_with_preset_directions(runner_cls):
    _four_policy_chain(runner_cls, preset_directions=True)


@RUNNERS
def test_four_policy_chain_without_preset_directions(runner_cls):
    _four_policy_chain(runner_cls, preset_directions=False)
