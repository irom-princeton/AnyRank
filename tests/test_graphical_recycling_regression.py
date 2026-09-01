"""Tests for alpha recycling on mirrored continuous NSM tests.

These tests use AnyRank's `SequentialGraphicalTest` factory so they exercise the
same mirrored test construction and alpha reassignment shape used at runtime.
They verify that changing alpha recomputes both child cutoffs, reads back
consistently, and preserves accumulated test state.
"""

import numpy as np
import pytest

from multitest.sequential_graphical_active import SequentialGraphicalTest


@pytest.fixture
def graphical_test():
    """Create a small graphical test with three comparable policies."""
    null_hypotheses = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)]
    sgt = SequentialGraphicalTest(
        num_policies=3, null_hypotheses=null_hypotheses, total_alpha=0.05
    )
    return sgt


class TestAlphaCutoffRecycling:
    """Verify alpha reassignment recomputes both mirrored child cutoffs."""

    def test_recycling_updates_both_child_cutoffs(self, graphical_test):
        sgt = graphical_test
        nsm_test = sgt.initialize_graphical_nsm_test(Nmax=200, alpha=sgt.alpha[0])

        old_alt_cutoff = nsm_test._test_for_alternative._cutoff
        old_null_cutoff = nsm_test._test_for_null._cutoff
        assert old_alt_cutoff == pytest.approx(old_null_cutoff)

        new_alpha = sgt.alpha[0] * 2.0
        nsm_test.alpha = new_alpha

        assert nsm_test._test_for_alternative._cutoff == pytest.approx(1.0 / new_alpha)
        assert nsm_test._test_for_null._cutoff == pytest.approx(1.0 / new_alpha)
        # Both cutoffs must change when the recycled alpha changes.
        assert nsm_test._test_for_null._cutoff != old_null_cutoff


class TestAlphaReadBackCoherence:
    """Verify alpha reads back consistently after reassignment."""

    def test_alpha_reads_back_after_recycling(self, graphical_test):
        sgt = graphical_test
        nsm_test = sgt.initialize_graphical_nsm_test(Nmax=200, alpha=sgt.alpha[0])

        new_alpha = 0.025
        nsm_test.alpha = new_alpha

        assert nsm_test._test_for_alternative.alpha == new_alpha
        assert nsm_test._test_for_null.alpha == new_alpha
        # Wrapper read fans out to alternative child via __getattr__.
        assert nsm_test.alpha == new_alpha


class TestStatePreservationUnderRecycling:
    """Verify alpha reassignment preserves accumulated mirrored-test state."""

    def test_recycling_preserves_all_running_state(self, graphical_test):
        sgt = graphical_test
        nsm_test = sgt.initialize_graphical_nsm_test(Nmax=200, alpha=sgt.alpha[0])

        # Step enough to build non-initial state.
        nsm_test.step(0, 1)
        nsm_test.step(0, 1)
        nsm_test.step(1, 0)

        alt_mart = nsm_test._test_for_alternative._martingale
        alt_pv = nsm_test._test_for_alternative._p_value
        alt_t = nsm_test._test_for_alternative._t
        alt_lambda_len = len(nsm_test._test_for_alternative._store_lambda_params)
        null_mart = nsm_test._test_for_null._martingale
        null_pv = nsm_test._test_for_null._p_value
        null_t = nsm_test._test_for_null._t
        null_lambda_len = len(nsm_test._test_for_null._store_lambda_params)
        wrapper_pv = nsm_test._p_value
        wrapper_pt = nsm_test._p_type

        # Recycle alpha as AnyRank does.
        nsm_test.alpha = sgt.alpha[0] * 3.0

        assert nsm_test._test_for_alternative._martingale == alt_mart
        assert nsm_test._test_for_alternative._p_value == alt_pv
        assert nsm_test._test_for_alternative._t == alt_t
        assert len(nsm_test._test_for_alternative._store_lambda_params) == alt_lambda_len
        assert nsm_test._test_for_null._martingale == null_mart
        assert nsm_test._test_for_null._p_value == null_pv
        assert nsm_test._test_for_null._t == null_t
        assert len(nsm_test._test_for_null._store_lambda_params) == null_lambda_len
        assert nsm_test._p_value == wrapper_pv
        assert nsm_test._p_type == wrapper_pt


class TestGraphicalCleanupBehavior:
    """Regression coverage for integration cleanup behavior."""

    def test_alpha_at_rejected_is_keyed_by_hypothesis(self, graphical_test):
        sgt = graphical_test
        hypotheses = [(0, 1), (0, 2), (1, 2)]
        hypotheses_correct = {hyp: 0.0 for hyp in hypotheses}
        p_values = np.zeros(len(hypotheses))
        p_value_types = ["alternative"] * len(hypotheses)

        _rejected, _graphs, alpha_at_rejected, _updated = (
            sgt.sequential_graphical_test(
                p_values,
                [],
                hypotheses_correct,
                hypotheses,
                num_policies=3,
                p_value_types=p_value_types,
                allow_transitive=False,
            )
        )

        assert alpha_at_rejected
        for hypothesis, alpha in alpha_at_rejected:
            assert hypothesis in hypotheses
            assert alpha > 0.0

    def test_batch_run_does_not_write_martingale_plots_by_default(
        self, tmp_path, monkeypatch,
    ):
        sgt = SequentialGraphicalTest(
            num_policies=2,
            null_hypotheses=[(0.5, 0.5)],
            total_alpha=0.05,
        )
        assert sgt.enable_martingale_plots is False

        policy_data = np.column_stack([
            np.ones(20),
            np.zeros(20),
        ])
        monkeypatch.chdir(tmp_path)

        sgt.sequential_graphical_multitest(
            ordered_hypotheses_policy_indices=[(0, 1)],
            policy_data=policy_data,
            Nmax=20,
            alpha_per_hypothesis=sgt.alpha.copy(),
            bernoulli=True,
            allow_transitive=False,
            verbose=False,
        )

        assert not (tmp_path / "imglib").exists()
