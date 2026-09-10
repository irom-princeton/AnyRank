"""Tests for alpha recycling on mirrored continuous NSM tests.

These tests use AnyRank's `SequentialGraphicalTest` factory so they exercise the
same mirrored test construction and alpha reassignment shape used at runtime.
They verify that changing alpha recomputes both child cutoffs, reads back
consistently, and preserves accumulated test state.
"""

import numpy as np
import pytest

from multitest import sequential_graphical_active as sga_module
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

    def test_batch_rng_default_uses_global_numpy_choice(
        self, tmp_path, monkeypatch,
    ):
        sgt = SequentialGraphicalTest(
            num_policies=2,
            null_hypotheses=[(0.5, 0.5)],
            total_alpha=0.05,
        )
        policy_data = np.column_stack([
            np.ones(20),
            np.zeros(20),
        ])
        calls = []

        def fake_choice(a, p):
            calls.append((a, p.copy()))
            return 0

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(np.random, "choice", fake_choice)

        sgt.sequential_graphical_multitest(
            ordered_hypotheses_policy_indices=[(0, 1)],
            policy_data=policy_data,
            Nmax=20,
            alpha_per_hypothesis=sgt.alpha.copy(),
            bernoulli=True,
            allow_transitive=False,
            verbose=False,
        )

        assert calls
        assert calls[0][0] == 1
        assert calls[0][1] == pytest.approx([1.0])

    def test_batch_rng_argument_bypasses_global_numpy_choice(
        self, tmp_path, monkeypatch,
    ):
        sgt = SequentialGraphicalTest(
            num_policies=2,
            null_hypotheses=[(0.5, 0.5)],
            total_alpha=0.05,
        )
        policy_data = np.column_stack([
            np.ones(20),
            np.zeros(20),
        ])

        class ChoiceRecorder:
            def __init__(self):
                self.calls = []

            def choice(self, a, p):
                self.calls.append((a, p.copy()))
                return 0

        def fail_global_choice(*_args, **_kwargs):
            raise AssertionError("explicit rng must bypass np.random.choice")

        rng = ChoiceRecorder()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(np.random, "choice", fail_global_choice)

        sgt.sequential_graphical_multitest(
            ordered_hypotheses_policy_indices=[(0, 1)],
            policy_data=policy_data,
            Nmax=20,
            alpha_per_hypothesis=sgt.alpha.copy(),
            bernoulli=True,
            allow_transitive=False,
            verbose=False,
            rng=rng,
        )

        assert rng.calls
        assert rng.calls[0][0] == 1
        assert rng.calls[0][1] == pytest.approx([1.0])


class TestActiveLoopPerformanceShape:
    """Regression coverage for the performance-oriented active-loop cleanups."""

    def _make_two_policy_test(self):
        sgt = SequentialGraphicalTest(
            num_policies=2,
            null_hypotheses=[(0.5, 0.5)],
            total_alpha=0.05,
        )
        policy_data = np.column_stack([
            np.ones(20),
            np.zeros(20),
        ])
        return sgt, policy_data

    def test_default_run_is_quiet(self, capsys, monkeypatch, tmp_path):
        sgt, policy_data = self._make_two_policy_test()
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

        captured = capsys.readouterr()
        assert "Running means" not in captured.out
        assert "At policy complexity" not in captured.out
        assert "Candidate hypotheses to reject" not in captured.out
        assert "Evaluations:" not in captured.out

    def test_verbose_run_still_emits_diagnostics(
        self, capsys, monkeypatch, tmp_path,
    ):
        sgt, policy_data = self._make_two_policy_test()
        monkeypatch.chdir(tmp_path)

        sgt.sequential_graphical_multitest(
            ordered_hypotheses_policy_indices=[(0, 1)],
            policy_data=policy_data,
            Nmax=20,
            alpha_per_hypothesis=sgt.alpha.copy(),
            bernoulli=True,
            allow_transitive=False,
            verbose=True,
        )

        captured = capsys.readouterr()
        assert "Running means" in captured.out
        # Verbose must also reach the inner rejection diagnostics from
        # sequential_graphical_test; the outer verbose flag is propagated.
        assert "Candidate hypotheses to reject" in captured.out

    def test_hyp_completed_vs_time_tail_matches_final_value(
        self, monkeypatch, tmp_path,
    ):
        # Use four policies with strongly-separated Bernoulli rates so all six
        # unordered pairwise hypotheses resolve well before Tmax and leave a
        # real padding tail. This exercises the tail-fill fix: with multiple
        # hypotheses, np.mean(hypotheses_completed) takes intermediate values
        # (1/6, 2/6, ..., 6/6), so the old `+=` tail accumulated running sums
        # and produced values far above 1.0 in the padding region.
        monkeypatch.chdir(tmp_path)

        rates = [0.02, 0.35, 0.68, 0.98]
        num_policies = len(rates)
        hypotheses = [
            (i, j)
            for i in range(num_policies)
            for j in range(i + 1, num_policies)
        ]
        null_hypotheses = [(0.5, 0.5)] * len(hypotheses)
        sgt = SequentialGraphicalTest(
            num_policies=num_policies,
            null_hypotheses=null_hypotheses,
            total_alpha=0.05,
        )

        rng = np.random.default_rng(7)
        Nmax = 100
        policy_data = np.column_stack([
            rng.binomial(1, p, size=Nmax).astype(float) for p in rates
        ])

        result = sgt.sequential_graphical_multitest(
            ordered_hypotheses_policy_indices=hypotheses,
            policy_data=policy_data,
            Nmax=Nmax,
            alpha_per_hypothesis=sgt.alpha.copy(),
            bernoulli=True,
            allow_transitive=False,
            verbose=False,
        )
        hyp_completed_vs_time = result[-1]

        Tmax = num_policies * Nmax + 1
        assert hyp_completed_vs_time.shape == (Tmax + 1,)
        assert np.all(hyp_completed_vs_time >= 0.0)
        # Tail-fill correctness: under the old `+=` pattern the padding tail
        # accumulated running means and could exceed 1.0 by a large margin;
        # under the fixed `=` pattern the tail is clamped to the final
        # observed completion fraction. All values must therefore stay within
        # [0, 1] up to floating tolerance.
        assert np.all(hyp_completed_vs_time <= 1.0 + 1e-12)
        # After early completion, the remaining tail equals the last observed
        # value rather than accumulating additional running sums.
        assert hyp_completed_vs_time[-1] == pytest.approx(
            np.max(hyp_completed_vs_time)
        )

    def test_prefix_replay_is_removed(self, monkeypatch, tmp_path):
        # Construct data that would trigger the deleted replay guard:
        #   hypotheses_completed[k] > 0.5 AND hypotheses_correct[hyp] == 0.0
        #   AND KK[k] < policy_data.shape[0]
        # We use tied data on H0 = (0, 1) so that hypothesis exhausts to
        # KK[0] = Nmax without a decision (hypotheses_correct stays 0.0), while
        # H1 = (0, 2) has decisive data so the outer loop keeps iterating past
        # H0's exhaustion. Under the old code, the replay block would fire on
        # H0 at the top of every remaining iteration.
        monkeypatch.chdir(tmp_path)

        sgt = SequentialGraphicalTest(
            num_policies=3,
            null_hypotheses=[(0.5, 0.5), (0.5, 0.5)],
            total_alpha=0.05,
        )
        n_rows = 80
        Nmax = 15
        policy_data = np.column_stack([
            np.zeros(n_rows),
            np.zeros(n_rows),
            np.ones(n_rows),
        ])

        original = sga_module.ContinuousNsmTest

        class GuardedContinuousNsmTest(original):
            def run_on_sequence(self, *args, **kwargs):
                raise AssertionError(
                    "sequential_graphical_multitest must not replay prefixes "
                    "via ContinuousNsmTest.run_on_sequence"
                )

        monkeypatch.setattr(
            sga_module, "ContinuousNsmTest", GuardedContinuousNsmTest,
        )

        result = sgt.sequential_graphical_multitest(
            ordered_hypotheses_policy_indices=[(0, 1), (0, 2)],
            policy_data=policy_data,
            Nmax=Nmax,
            alpha_per_hypothesis=sgt.alpha.copy(),
            bernoulli=True,
            allow_transitive=False,
            verbose=False,
        )

        rejected = result[1]
        decision_times = result[2]
        hypotheses_correct = result[7]
        hyp_completed_vs_time = result[-1]

        # H0 must remain undecided so the completed-undecided guard would have
        # been reached under the old code; H1 must reject so the loop continues
        # long enough for that state to be observed.
        assert hypotheses_correct[(0, 1)] == 0.0
        assert decision_times[(0, 1)] == "N/A"
        assert 1 in rejected
        assert decision_times[(0, 2)] != "N/A"
        # Both hypotheses must reach the completed state; because H0 is never
        # decided, its completion can only come from KK[0] >= Nmax with
        # policy_data.shape[0] > Nmax, i.e. the exact condition the deleted
        # replay guard checked.
        assert np.max(hyp_completed_vs_time) == pytest.approx(1.0)
        # And with the replay block removed, ContinuousNsmTest.run_on_sequence
        # is never called at all.
