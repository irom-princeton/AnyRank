"""
Compare AnyRank (the ``graphical_active`` sequential graphical multitest) against
the STEP procedure with Bonferroni correction, on the same RoboArena eval data.

Both methods are driven from a single common dataset (the multitest ``data2``
per-trial progress CSV, via :func:`get_real_evals`) so the comparison is
controlled: the two pipelines differ *only* in how the per-pair time-to-decision
(TTD) and pairwise decisions are computed, and share the exact same rendering /
compact-letter-display (CLD) backend.

Per the agreed setup:
  * STEP runs on **binarized** (success / failure) data with a **Bonferroni**
    corrected per-comparison alpha, using ``MirroredStepTest``.
  * AnyRank (``graphical_active``) runs on **continuous task-progress** data.
  * AnyRank's priors (``sim_means`` used to order hypotheses / allocate alpha)
    come from an ``n_prior`` split of the same eval data.
  * Output is two per-method PNGs, rendered through the shared
    ``make_violin_plots_from_ttds`` / ``plot_model_comparison`` backend.

Both methods control the family-wise error rate at ``GLOBAL_ALPHA``:
  * STEP:    individual_alpha = alpha / n_pairs   (Bonferroni union bound)
  * AnyRank: total alpha budget = alpha, recycled across the pair graph.

Note on STEP policy synthesis: ``MirroredStepTest`` loads a precomputed decision
policy keyed by ``(n_max, individual_alpha)``; if none exists it synthesizes one
via a subprocess, which can be slow. We cap every policy to one common ``n_max``
so only a single policy is required per run.
"""

import argparse
import os
import shutil
from itertools import combinations

import numpy as np
import pandas as pd

from multitest.individual_test import (
    ExperimentConfig,
    split_eval,
    get_policy_summary,
    main as anyrank_main,
)
from multitest.plot_cld import make_violin_plots_from_ttds

from sequentialized_barnard_tests.base import Decision, Hypothesis
from sequentialized_barnard_tests.step import MirroredStepTest


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
GLOBAL_ALPHA = 0.1          # family-wise error rate for both methods
DATA_BINS = ["data2"]        # which data bin(s) of the CSV to load
SUCCESS_THRESHOLD = 1.0      # progress >= threshold counts as a STEP "success"

ALL_POLICIES = [
    "pi0_droid",
    "paligemma_vq_droid",
    "paligemma_fast_specialist_droid",
    "pi0_fast_droid",
    "paligemma_fast_droid",
    "paligemma_diffusion_droid",
    "paligemma_binning_droid",
]

PLOT_POLICY_NAMES = {
    "paligemma_binning_droid":         "PG-Bin",
    "paligemma_diffusion_droid":       "PG-Diff",
    "paligemma_vq_droid":              "PG-FSQ",
    "paligemma_fast_droid":            "PG-Fast",
    "paligemma_fast_specialist_droid": "PG-Fast-Spec",
    "pi0_droid":                       r"$\pi_0$",
    "pi0_fast_droid":                  r"$\pi_0$-Fast",
}

FOUR_POLICIES = [
    "paligemma_binning_droid",
    "pi0_droid",
    "paligemma_diffusion_droid",
    "pi0_fast_droid",
]

SEVEN_POLICIES = [
    "paligemma_binning_droid",
    "pi0_droid",
    "paligemma_diffusion_droid",
    "paligemma_vq_droid",
    "paligemma_fast_specialist_droid",
    "paligemma_fast_droid",
    "pi0_fast_droid",
]

POLICY_SETS = {"four": FOUR_POLICIES, "seven": SEVEN_POLICIES}


# ---------------------------------------------------------------------------
# Data loading / preprocessing
# ---------------------------------------------------------------------------
def get_real_evals(data_bins=None, selected_policies=None):
    """Load per-trial task-progress values for each policy from the CSV.

    Returns a dict ``{policy_name: 1-D np.ndarray of progress in [0, 1]}``.
    Policies generally have different numbers of evaluations (the CSV is sparse),
    so array lengths differ across policies.
    """
    if data_bins is None:
        data_bins = DATA_BINS
    filename = "per_trial_progress_data.csv"
    keys = selected_policies if selected_policies else list(ALL_POLICIES)
    eval_results = {k: np.array([]) for k in keys}
    for data_bin in data_bins:
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "roboarena", data_bin, filename
        )
        df = pd.read_csv(data_path)
        for k in keys:
            eval_results[k] = np.concatenate(
                [eval_results[k], df[k].dropna().to_numpy()]
            )
    return eval_results


def binarize(arr, threshold=SUCCESS_THRESHOLD):
    """Map continuous task progress to Bernoulli success/failure (0/1)."""
    return (np.asarray(arr) >= threshold).astype(int)


def labels_for(policies):
    """Display labels aligned to ``policies`` (list, same order)."""
    return [PLOT_POLICY_NAMES[p] for p in policies]


# ---------------------------------------------------------------------------
# STEP + Bonferroni
# ---------------------------------------------------------------------------
def compute_step_ttds(binary_data, policies, individual_alpha, n_max, random_seed=42):
    """Run pairwise ``MirroredStepTest`` on binarized data.

    Returns ``{(name_i, name_j): ttd}`` where ``ttd`` is the time-to-decision
    (an int) for decided pairs, or the string ``"N/A"`` for pairs that never
    reach a decision within ``n_max`` samples. This is exactly the format that
    :func:`make_violin_plots_from_ttds` / :func:`get_cld_letters_from_ttds`
    expect.
    """
    step_ttd = {}
    for alg1, alg2 in combinations(policies, 2):
        pi0 = binary_data[alg1][:n_max]
        pi1 = binary_data[alg2][:n_max]
        m = min(len(pi0), len(pi1))
        pi0, pi1 = pi0[:m], pi1[:m]
        test = MirroredStepTest(
            alternative=Hypothesis.P0LessThanP1,
            n_max=n_max,
            alpha=individual_alpha,
            random_seed=random_seed,
        )
        result = test.run_on_sequence(pi0, pi1)
        if result.decision == Decision.FailToDecide:
            step_ttd[(alg1, alg2)] = "N/A"
        else:
            step_ttd[(alg1, alg2)] = int(result.info["Time"])
        print(
            f"  STEP {alg1} vs {alg2}: decision={result.decision.name}, "
            f"ttd={step_ttd[(alg1, alg2)]}"
        )
    return step_ttd


def step_and_bernoulli(
    policy_data,
    policies,
    out_dir,
    alpha=GLOBAL_ALPHA,
    threshold=SUCCESS_THRESHOLD,
    max_sample_size=None,
):
    """STEP procedure with Bonferroni correction on binarized data.

    Produces ``<out_dir>/violin_step.png`` (success-rate violins with CLD letters
    and per-policy time-to-decision on the top axis).
    """
    os.makedirs(out_dir, exist_ok=True)
    n_pairs = len(policies) * (len(policies) - 1) // 2
    individual_alpha = alpha / n_pairs

    # Binarize, then cap every policy to a single common n_max so only one STEP
    # decision policy needs to be loaded/synthesized.
    binary_data = {p: binarize(policy_data[p], threshold) for p in policies}
    available = min(len(binary_data[p]) for p in policies)
    n_max = int(available if max_sample_size is None else min(max_sample_size, available))
    binary_data = {p: v[:n_max] for p, v in binary_data.items()}
    max_sample_size_per_model = {p: n_max for p in policies}

    print(
        f"\n[STEP] policies={policies}\n"
        f"[STEP] n_pairs={n_pairs}, global_alpha={alpha}, "
        f"individual_alpha={individual_alpha:.6g}, n_max={n_max}, "
        f"success_threshold={threshold}"
    )
    for p in policies:
        print(f"[STEP] {p}: n={n_max}, success_rate={np.mean(binary_data[p]):.3f}")

    step_ttd = compute_step_ttds(binary_data, policies, individual_alpha, n_max)

    make_violin_plots_from_ttds(
        progress_array_dict=binary_data,
        ttd_dicts={"step": step_ttd},
        policies=policies,
        labels=labels_for(policies),
        max_sample_size_per_model=max_sample_size_per_model,
        output_dir=out_dir,
        mode="success_rate",
    )
    return os.path.join(out_dir, "violin_step.png")


# ---------------------------------------------------------------------------
# AnyRank (graphical_active) on continuous task progress
# ---------------------------------------------------------------------------
def graphical_active(
    policy_data,
    policies,
    out_dir,
    alpha=GLOBAL_ALPHA,
    n_prior=20,
    beta=1.0,
    bernoulli=False,    
    graph_type="fully_connected",
    allow_transitive=True,
):
    """AnyRank ``graphical_active`` multitest on continuous task-progress data.

    Priors (``sim_means``) are taken from an ``n_prior`` split of the same eval
    data; ``real_means`` come from the held-out remainder. Delegates to the real
    AnyRank ``main`` (with ``methods=['graphical_active']``) so the statistics
    match the ``roboarena_individual_test`` pipeline exactly. That call renders
    the violin via the same ``make_violin_plots_from_ttds`` backend used by STEP.

    Produces ``<out_dir>/cld_violins/beta{beta}/violin_graphical_active.png``.
    """
    os.makedirs(out_dir, exist_ok=True)
    get_policy_summary({p: policy_data[p] for p in policies})

    # n_prior split of the same eval data -> priors for hypothesis ordering.
    Nmax_csv = min(len(policy_data[p]) for p in policies)
    prior_evals, real_evals = split_eval(policy_data, Nmax_csv, nruns=n_prior)
    sim_means = {k: float(np.mean(prior_evals[k])) for k in policies}
    real_means = {k: float(np.mean(real_evals[k])) for k in policies}
    binary_data = {p: binarize(policy_data[p], 1) for p in policies}
    n_max = min(len(binary_data[p]) for p in policies)
    binary_data = {p: v[:n_max] for p, v in binary_data.items()}

    if bernoulli:
        policy_data = binary_data

    print(
        f"\n[AnyRank] policies={policies}\n"
        f"[AnyRank] global_alpha={alpha}, n_prior={n_prior}, beta={beta}, "
        f"graph_type={graph_type}, Nmax_csv={Nmax_csv}"
    )
    for p in policies:
        print(f"[AnyRank] {p}: sim_mean={sim_means[p]:.3f}, real_mean={real_means[p]:.3f}")

    cfg = ExperimentConfig(
        alpha=alpha,
        n_prior=n_prior,
        beta=beta,
        results_dir=out_dir,
        graph_type=graph_type,
    )
    anyrank_main(
        policy_data,
        policies=policies,
        sim_means=sim_means,
        real_means=real_means,
        bernoulli=bernoulli,  # continuous task progress
        cfg=cfg,
        labels=PLOT_POLICY_NAMES,
        methods=["graphical_active"],
        allow_transitive=allow_transitive,
    )
    return os.path.join(out_dir, "cld_violins", f"beta{beta}", "violin_graphical_active.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare AnyRank (graphical_active) vs STEP+Bonferroni."
    )
    parser.add_argument(
        "--policies", choices=list(POLICY_SETS), default="seven",
        help="Which RoboArena policy set to compare (default: four).",
    )
    parser.add_argument("--alpha", type=float, default=GLOBAL_ALPHA,
                        help="Family-wise error rate for both methods.")
    parser.add_argument("--threshold", type=float, default=SUCCESS_THRESHOLD,
                        help="Progress >= threshold counts as STEP success.")
    parser.add_argument("--n_prior", type=int, default=20,
                        help="Number of prior evals for the AnyRank hypothesis split.")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="AnyRank alpha-allocation tuning parameter.")
    parser.add_argument("--bernoulli", type=bool, default=False,
                        help="AnyRank parameter for NSCORE.")
    parser.add_argument("--graph_type", choices=["soft_masked", "fully_connected"],
                        default="fully_connected",
                        help="AnyRank alpha-transfer graph type.")
    parser.add_argument("--max_sample_size", type=int, default=None,
                        help="Optional cap on STEP n_max (default: min available).")
    parser.add_argument("--which", choices=["both", "step", "anyrank"], default="both",
                        help="Which procedure(s) to run.")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join("outputs", "compare_to_step_seven"),
                        help="Output directory for the comparison plots.")
    args = parser.parse_args()

    policies = POLICY_SETS[args.policies]
    policy_data_all = get_real_evals(data_bins=DATA_BINS, selected_policies=policies)
    os.makedirs(args.out_dir, exist_ok=True)

    for Nmax in [100, 200, 300, 400, 500]:    
        policy_data = {p: policy_data_all[p][:Nmax] for p in policies}
        sim_means = {p: float(np.mean(policy_data_all[p])) for p in policies}
        real_means = {p: float(np.mean(policy_data_all[p])) for p in policies}
        print(f"\n[Main] policies={policies}, n_max={Nmax}")

        produced = {}
        if args.which in ("both", "step"):
            step_png = step_and_bernoulli(
                policy_data, policies,
                out_dir=os.path.join(args.out_dir, str(Nmax), "step"),
                alpha=args.alpha, threshold=args.threshold,
                max_sample_size=args.max_sample_size,
            )
            produced["step"] = step_png

        if args.which in ("both", "anyrank"):
            anyrank_png = graphical_active(
                policy_data, policies,
                out_dir=os.path.join(args.out_dir, str(Nmax), "anyrank"),
                alpha=args.alpha, n_prior=args.n_prior, beta=args.beta,
                graph_type=args.graph_type,
                bernoulli=args.bernoulli
            )
            produced["graphical_active"] = anyrank_png

        # Collect the per-method PNGs into the top-level out_dir with clear names.
        print("\n" + "=" * 60)
        print("Comparison plots:")
        for method, path in produced.items():
            if path and os.path.exists(path):
                dest = os.path.join(args.out_dir, str(Nmax), f"violin_{method}.png")
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if os.path.abspath(dest) != os.path.abspath(path):
                    shutil.copyfile(path, dest)
                print(f"  {method:16s} -> {dest}")
            else:
                print(f"  {method:16s} -> (not produced; expected at {path})")
        print("=" * 60)


if __name__ == "__main__":
    main()
