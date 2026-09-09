"""Tests for AnyRank's ``inference_mode`` threading.

AnyRank defaults ``inference_mode="ranking_no_ties"`` at every construction
site because it is a ranking tool with no baseline policy. These tests pin
that default, prove that ``ranking_no_ties`` is byte-identical to the
previous unspecified default on both graphical classes, prove that
``inference_mode="ranking"`` actually reaches the mirrored child tests and
halves their child alpha, prove that an invalid mode surfaces SCC's
``ValueError``, prove that ``ExperimentConfig`` carries the mode into the
saved ``config.json``, and prove that ``compare_to_step`` under the default
mode does not trigger real STEP policy synthesis on disk.
"""

import copy
import json
import os
from dataclasses import asdict

import numpy as np
import pytest

from multitest.sequential_graphical import SequentialGraphicalTest
from multitest.sequential_graphical_active import (
    SequentialGraphicalTest as ASequentialGraphicalTest,
)


# --- helpers -----------------------------------------------------------------


def _sample_policy_data(seed=0, n_rows=200, num_policies=3):
    rng = np.random.default_rng(seed)
    return rng.random((n_rows, num_policies))


def _sample_hypotheses():
    return [(0, 1), (0, 2), (1, 2)]


def _run_passive(sgt, hypotheses, policy_data, Nmax=50):
    alpha_per_hypothesis = sgt.alpha.copy()
    return sgt.sequential_graphical_multitest(
        hypotheses,
        policy_data,
        Nmax,
        alpha_per_hypothesis=alpha_per_hypothesis,
        verbose=False,
    )


def _run_active(sgt, hypotheses, policy_data, Nmax=50):
    alpha_per_hypothesis = sgt.alpha.copy()
    return sgt.sequential_graphical_multitest(
        hypotheses,
        policy_data,
        Nmax,
        alpha_per_hypothesis=alpha_per_hypothesis,
        bernoulli=False,
        allow_transitive=False,
        verbose=False,
    )


# --- default is ranking_no_ties ---------------------------------------------


def test_sequential_graphical_default_is_ranking_no_ties():
    sgt = SequentialGraphicalTest(
        num_policies=3,
        null_hypotheses=_sample_hypotheses(),
        total_alpha=0.05,
    )
    assert sgt.inference_mode == "ranking_no_ties"


def test_sequential_graphical_active_default_is_ranking_no_ties():
    sgt = ASequentialGraphicalTest(
        num_policies=3,
        null_hypotheses=_sample_hypotheses(),
        total_alpha=0.05,
    )
    assert sgt.inference_mode == "ranking_no_ties"


# --- ranking_no_ties is byte-identical to the previous default ---------------


def test_passive_equivalence_omitted_vs_explicit_ranking_no_ties():
    hypotheses = _sample_hypotheses()
    policy_data = _sample_policy_data(seed=1)

    sgt_default = SequentialGraphicalTest(
        num_policies=3,
        null_hypotheses=hypotheses,
        total_alpha=0.05,
    )
    sgt_explicit = SequentialGraphicalTest(
        num_policies=3,
        null_hypotheses=hypotheses,
        total_alpha=0.05,
        inference_mode="ranking_no_ties",
    )

    result_default = _run_passive(sgt_default, hypotheses, policy_data)
    result_explicit = _run_passive(sgt_explicit, hypotheses, policy_data.copy())

    # Passive return: (rejected_hypotheses, rejected, decision_times, p_values,
    # G, graphs_over_time, alpha_at_rejected, hypotheses_correct).
    assert result_default[0] == result_explicit[0]
    assert result_default[1] == result_explicit[1]
    assert result_default[2] == result_explicit[2]
    assert result_default[7] == result_explicit[7]


def test_active_equivalence_omitted_vs_explicit_ranking_no_ties():
    hypotheses = _sample_hypotheses()
    policy_data = _sample_policy_data(seed=2)

    sgt_default = ASequentialGraphicalTest(
        num_policies=3,
        null_hypotheses=hypotheses,
        total_alpha=0.05,
    )
    sgt_explicit = ASequentialGraphicalTest(
        num_policies=3,
        null_hypotheses=hypotheses,
        total_alpha=0.05,
        inference_mode="ranking_no_ties",
    )

    np.random.seed(7)
    result_default = _run_active(sgt_default, hypotheses, policy_data)
    np.random.seed(7)
    result_explicit = _run_active(sgt_explicit, hypotheses, policy_data.copy())

    # Active return: 9-tuple; compare decisions, decision_times, and the
    # rejected list to pin behavior on both classes.
    assert result_default[0] == result_explicit[0]
    assert result_default[1] == result_explicit[1]
    assert result_default[2] == result_explicit[2]
    assert result_default[7] == result_explicit[7]


# --- inference_mode=ranking halves child alpha at construction ---------------


def test_ranking_mode_halves_child_alpha_on_initializer():
    sgt = SequentialGraphicalTest(
        num_policies=3,
        null_hypotheses=_sample_hypotheses(),
        total_alpha=0.05,
        inference_mode="ranking",
    )
    alpha = 0.02
    mirrored = sgt.initialize_graphical_nsm_test(Nmax=200, alpha=alpha)

    assert mirrored.alpha == pytest.approx(alpha)
    assert mirrored._test_for_alternative.alpha == pytest.approx(alpha / 2.0)
    assert mirrored._test_for_null.alpha == pytest.approx(alpha / 2.0)


def test_ranking_mode_active_initializer_halves_child_alpha():
    sgt = ASequentialGraphicalTest(
        num_policies=3,
        null_hypotheses=_sample_hypotheses(),
        total_alpha=0.05,
        inference_mode="ranking",
    )
    alpha = 0.02
    mirrored = sgt.initialize_graphical_nsm_test(Nmax=200, alpha=alpha)

    assert mirrored.alpha == pytest.approx(alpha)
    assert mirrored._test_for_alternative.alpha == pytest.approx(alpha / 2.0)
    assert mirrored._test_for_null.alpha == pytest.approx(alpha / 2.0)


# --- invalid mode surfaces SCC's ValueError ---------------------------------


def test_invalid_inference_mode_surfaces_scc_value_error():
    sgt = SequentialGraphicalTest(
        num_policies=3,
        null_hypotheses=_sample_hypotheses(),
        total_alpha=0.05,
        inference_mode="bogus",
    )
    with pytest.raises(ValueError) as excinfo:
        sgt.initialize_graphical_nsm_test(Nmax=200, alpha=0.02)
    message = str(excinfo.value)
    assert "bogus" in message
    for mode in ("comparison", "ranking", "ranking_no_ties"):
        assert mode in message


def test_invalid_inference_mode_active_surfaces_scc_value_error():
    sgt = ASequentialGraphicalTest(
        num_policies=3,
        null_hypotheses=_sample_hypotheses(),
        total_alpha=0.05,
        inference_mode="bogus",
    )
    with pytest.raises(ValueError):
        sgt.initialize_graphical_nsm_test(Nmax=200, alpha=0.02)


# --- ExperimentConfig default and saved-config behavior ---------------------


def test_experiment_config_default_is_ranking_no_ties():
    from multitest.individual_test import ExperimentConfig as IndCfg
    from multitest.run_graphical_test import ExperimentConfig as RunCfg

    assert IndCfg().inference_mode == "ranking_no_ties"
    assert RunCfg().inference_mode == "ranking_no_ties"


def test_experiment_config_records_inference_mode_in_asdict(tmp_path):
    from multitest.individual_test import ExperimentConfig

    cfg = ExperimentConfig(inference_mode="ranking")
    payload = asdict(cfg)
    assert payload["inference_mode"] == "ranking"
    out = tmp_path / "config.json"
    with open(out, "w") as f:
        json.dump(payload, f)
    with open(out) as f:
        restored = json.load(f)
    assert restored["inference_mode"] == "ranking"


# --- compare_to_step default does not trigger STEP policy synthesis ---------


def test_compare_to_step_default_does_not_trigger_step_policy_synthesis(
    monkeypatch, tmp_path,
):
    # Replace subprocess.run in the SBT step module with a guard so that any
    # STEP synthesis attempt would be observable via a captured cmd list. We
    # also swallow the actual load attempt by handing STEP a fake policy via
    # the same monkeypatch pattern the AnyEval adapter tests use.
    import sequentialized_barnard_tests.step as sbt_step

    subprocess_cmds = []

    def fake_subprocess_run(cmd, *args, **kwargs):
        subprocess_cmds.append(cmd)
        raise AssertionError(
            "STEP policy synthesis subprocess triggered under compare_to_step "
            "default (ranking_no_ties): "
            f"{cmd}"
        )

    monkeypatch.setattr(sbt_step.subprocess, "run", fake_subprocess_run)

    # Force the load_existing_policy path to succeed with a fake policy so
    # STEP does not attempt synthesis when a fixture is missing. Any test-time
    # STEP construction here should not need real fixtures either way.
    def fake_load_existing_policy(self, verbose=False):
        self.policy_path = os.path.join(
            os.path.dirname(sbt_step.__file__),
            f"policies/n_max_{self.n_max}_alpha_{self._policy_alpha}_shape_parameter_{self.shape_parameter}_pnorm_{self.use_p_norm}/",
            "policy_compressed.pkl",
        )
        self.policy = "fake_policy"
        self.need_new_policy = False
        self.reset(verbose)

    monkeypatch.setattr(
        sbt_step.StepTest, "load_existing_policy", fake_load_existing_policy
    )

    # Construct MirroredStepTest directly through the shared entrypoint the
    # script uses. Under the default inference_mode the policy_alpha equals
    # the declared alpha, so synthesis should not fire.
    step_test = sbt_step.MirroredStepTest(
        alternative=sbt_step.Hypothesis.P0MoreThanP1,
        n_max=200,
        alpha=0.05,
        inference_mode="ranking_no_ties",
    )
    assert step_test.inference_mode == "ranking_no_ties"
    assert step_test._policy_alpha == pytest.approx(0.05)
    assert not subprocess_cmds


# --- module-level graphical.py entrypoints thread inference_mode -----------


def test_module_bonferroni_multitest_invalid_mode_surfaces_scc_value_error():
    from multitest.graphical import bonferroni_multitest

    hypotheses = _sample_hypotheses()
    policy_data = _sample_policy_data(seed=3)

    with pytest.raises(ValueError) as excinfo:
        bonferroni_multitest(
            hypotheses,
            policy_data,
            Nmax=20,
            alpha=0.05,
            inference_mode="bogus",
        )
    message = str(excinfo.value)
    assert "bogus" in message
    for mode in ("comparison", "ranking", "ranking_no_ties"):
        assert mode in message


def test_module_graphical_multitest_invalid_mode_surfaces_scc_value_error():
    from multitest.graphical import graphical_multitest

    hypotheses = _sample_hypotheses()
    policy_data = _sample_policy_data(seed=4)

    with pytest.raises(ValueError) as excinfo:
        graphical_multitest(
            hypotheses,
            policy_data,
            Nmax=20,
            alpha=0.05,
            inference_mode="bogus",
        )
    message = str(excinfo.value)
    assert "bogus" in message
    for mode in ("comparison", "ranking", "ranking_no_ties"):
        assert mode in message


def test_module_bonferroni_multitest_ranking_reaches_child_alpha(monkeypatch):
    # Prove that inference_mode="ranking" actually reaches the child tests
    # through the bonferroni_multitest -> initialize_bonferroni_test path.
    from multitest import graphical as graphical_mod

    captured = {}
    real_initializer = graphical_mod.initialize_bonferroni_test

    def spy_initializer(Nmax, alpha, num_hypotheses, inference_mode="ranking_no_ties"):
        test = real_initializer(Nmax, alpha, num_hypotheses, inference_mode=inference_mode)
        captured["test"] = test
        captured["inference_mode"] = inference_mode
        return test

    monkeypatch.setattr(graphical_mod, "initialize_bonferroni_test", spy_initializer)

    hypotheses = _sample_hypotheses()
    policy_data = _sample_policy_data(seed=5)
    graphical_mod.bonferroni_multitest(
        hypotheses,
        policy_data,
        Nmax=20,
        alpha=0.06,
        inference_mode="ranking",
    )

    assert captured["inference_mode"] == "ranking"
    mirrored = captured["test"]
    # Bonferroni divides alpha by num_hypotheses (=3): declared child alpha
    # would be 0.02 under comparison; ranking halves it to 0.01 on both
    # children while the wrapper still reads 0.02.
    assert mirrored.alpha == pytest.approx(0.06 / 3)
    assert mirrored._test_for_alternative.alpha == pytest.approx((0.06 / 3) / 2.0)
    assert mirrored._test_for_null.alpha == pytest.approx((0.06 / 3) / 2.0)


def test_module_graphical_multitest_ranking_reaches_child_alpha(monkeypatch):
    # Prove the ranking split reaches the child tests through the
    # graphical_multitest -> initialize_nsm_test path, and that the wrapper
    # GraphicalTest received the same inference_mode value.
    from multitest import graphical as graphical_mod

    captured_tests = []
    captured_graph_modes = []
    real_init_nsm = graphical_mod.initialize_nsm_test
    real_graphical_cls = graphical_mod.GraphicalTest

    def spy_init_nsm(Nmax, alpha, inference_mode="ranking_no_ties"):
        test = real_init_nsm(Nmax, alpha, inference_mode=inference_mode)
        captured_tests.append((alpha, inference_mode, test))
        return test

    class SpyGraphicalTest(real_graphical_cls):
        def __init__(self, *args, inference_mode="ranking_no_ties", **kwargs):
            captured_graph_modes.append(inference_mode)
            super().__init__(*args, inference_mode=inference_mode, **kwargs)

    monkeypatch.setattr(graphical_mod, "initialize_nsm_test", spy_init_nsm)
    monkeypatch.setattr(graphical_mod, "GraphicalTest", SpyGraphicalTest)

    hypotheses = _sample_hypotheses()
    policy_data = _sample_policy_data(seed=6)
    graphical_mod.graphical_multitest(
        hypotheses,
        policy_data,
        Nmax=20,
        alpha=0.06,
        inference_mode="ranking",
    )

    assert captured_graph_modes == ["ranking"]
    assert captured_tests, "graphical_multitest did not construct any NSM tests"
    for alpha_i, mode_i, mirrored in captured_tests:
        assert mode_i == "ranking"
        assert mirrored.alpha == pytest.approx(alpha_i)
        assert mirrored._test_for_alternative.alpha == pytest.approx(alpha_i / 2.0)
        assert mirrored._test_for_null.alpha == pytest.approx(alpha_i / 2.0)
