import argparse
import numpy as np
import os
import copy
import pandas as pd
from multitest.individual_test import (
    ExperimentConfig,
    allocate_alpha,
    allocate_weights,
    animate,
    plot_heatmap,
    _accumulate_samples,
    _average_dict,
    _fill_exchange_matrix,
    _to_policy_matrix,
    split_eval,
    get_policy_summary,
    get_partial_ranking,
    run_experiments,
    main,
)

np.random.seed(42)

ALL_METHODS = ['fixed', 'bonferroni', 'weighted_bonferroni', 'graphical_active']

def get_real_evals(data_bins=["roboarena"], selected_policies=[]):
    filename = "per_trial_progress_data.csv"
    if selected_policies:
        keys = selected_policies
        eval_results = {k: np.array([]) for k in keys}
    else:
        keys = None
        eval_results = {}
    for data_bin in data_bins:
        data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "roboarena", data_bin, filename)
        df = pd.read_csv(data_path)
        if keys is None:
            keys = df.columns.tolist()
            # keys = [k for k in keys if "floppy" not in k]
            eval_results = {k: np.array([]) for k in keys}
        for k in keys:
            eval_results[k] = np.concatenate([eval_results[k], df[k].dropna().to_numpy()])
    return eval_results


def roboarena_wm_experiment(selected_policies=None, wm_prior_path=None, cfg=None):
    """Load roboarena data and return the generic inputs needed by main()."""
    if cfg is None:
        cfg = ExperimentConfig()
    bernoulli = False
    data_bins = ["roboarena"]
    if not selected_policies:
        selected_policies = ["paligemma_binning_droid", "pi0_droid", "paligemma_diffusion_droid", "pi0_fast_droid"]
    policy_data = get_real_evals(data_bins=data_bins, selected_policies=selected_policies)
    get_policy_summary(policy_data)
    Nmax_csv = min(len(v) for v in policy_data.values())
    _, real_evals = split_eval(policy_data, Nmax_csv, nruns=cfg.n_prior)
    policies = list(real_evals.keys())

    df_wm = pd.read_csv(wm_prior_path)
    wm_evals = {k: df_wm[df_wm['wm_id'] == k]['score'].to_numpy() for k in policies}
    sim_means = {k: np.mean(wm_evals[k]) for k in policies}
    real_means = {k: np.mean(real_evals[k]) for k in policies}
    return policy_data, policies, sim_means, real_means, bernoulli


def roboarena_experiment(selected_policies=None, cfg=None):
    """Load roboarena data and return the generic inputs needed by main()."""
    if cfg is None:
        cfg = ExperimentConfig()
    bernoulli = False
    data_bins = ["roboarena"]
    perfect_sim = False
    # if not selected_policies:
    #     selected_policies = ["paligemma_binning_droid", "pi0_droid", "paligemma_diffusion_droid", "pi0_fast_droid"]
    policy_data = get_real_evals(data_bins=data_bins, selected_policies=None)

    get_policy_summary(policy_data)
    
    Nmax_csv = min(len(v) for v in policy_data.values())
    policies = list(policy_data.keys())

    if perfect_sim:
        sim_means = {p: np.mean(policy_data[p]) for p in policies}
        real_means = {p: np.mean(policy_data[p]) for p in policies}
    else:
        prior_evals, real_evals = split_eval(policy_data, Nmax_csv, nruns=cfg.n_prior)
        sim_means = {k: np.mean(prior_evals[k]) for k in policies}
        real_means = {k: np.mean(real_evals[k]) for k in policies}
    return policy_data, policies, sim_means, real_means, bernoulli


def synthetic_experiment(n_policies=10, n_sim=100, n_real=500, std=0.15, seed=0, cfg=None):
    """Generate synthetic policies with true means spread over [0, 1].

    Ten (by default) true means are drawn uniformly on [0, 1]. For each policy,
    per-trial "progress" scores are sampled from a Gaussian centered at that mean
    with standard deviation ``std`` and clipped to [0, 1] (progress is bounded).
    ``n_sim`` samples build the simulation prior and ``n_real`` samples build the
    real evaluations, mirroring the (sim prior, real evals) split that main()
    expects. No files are read; everything is generated in-memory.

    Returns the generic inputs needed by main().
    """
    if cfg is None:
        cfg = ExperimentConfig()
    bernoulli = False
    rng = np.random.default_rng(seed)

    # True means spread (roughly) evenly across [0, 1]; sorted worst -> best.
    true_means = np.sort(rng.uniform(0.0, 1.0, size=n_policies))
    policies = [f"policy_{i:02d}" for i in range(n_policies)]

    policy_data = {}
    sim_means = {}
    real_means = {}
    for i, p in enumerate(policies):
        mu = true_means[i]
        sim_samples = np.clip(rng.normal(mu, std, size=n_sim), 0.0, 1.0)
        real_samples = np.clip(rng.normal(mu, std, size=n_real), 0.0, 1.0)
        policy_data[p] = real_samples
        sim_means[p] = float(np.mean(sim_samples))   # simulation prior mean
        real_means[p] = float(np.mean(real_samples))  # empirical real mean

    print("Synthetic true means (sorted): ",
          np.array2string(true_means, precision=3, suppress_small=True))
    get_policy_summary(policy_data)
    return policy_data, policies, sim_means, real_means, bernoulli


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subfolder", type=str, default=None, help="Optional subfolder under outputs/<exp_name>/")
    parser.add_argument("--exp_name", type=str, default="roboarena7_wm_prior",
                        help="Experiment to run")
    parser.add_argument("--transitive", type=bool, default=True,
                        choices=[True, False],
                        help="Whether to use transitive policy evaluation")
    parser.add_argument("--graph_type", type=str, default="fully_connected",
                        choices=["soft_masked", "fully_connected"],
                        help="Graph type for alpha transfer weights")
    parser.add_argument("--methods", type=str, nargs='+', default=ALL_METHODS,
                        choices=ALL_METHODS,
                        metavar="METHOD",
                        help=f"Methods to run (default: all). Choices: {', '.join(ALL_METHODS)}")
    # Synthetic-experiment knobs (used only when --exp_name synthetic)
    parser.add_argument("--n_policies", type=int, default=10, help="Number of synthetic policies")
    parser.add_argument("--n_sim", type=int, default=100, help="Samples for the synthetic sim prior")
    parser.add_argument("--n_real", type=int, default=500, help="Samples for the synthetic real evals")
    parser.add_argument("--std", type=float, default=0.15, help="Std of the synthetic Gaussian scores")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for synthetic data")
    args = parser.parse_args()
    methods = args.methods  # None means all

    beta_range = [1]
    four_policies = ["paligemma_binning_droid", "pi0_droid", "paligemma_diffusion_droid", "pi0_fast_droid"]
    seven_policies = ["paligemma_binning_droid",
            "pi0_droid",
            "paligemma_diffusion_droid",
            "paligemma_vq_droid",
            "paligemma_fast_specialist_droid",
            "paligemma_fast_droid",
            "pi0_fast_droid"]
    plot_policy_names = {
        "paligemma_binning_droid":        "PG-Bin",
        "paligemma_diffusion_droid":      "PG-Diff",
        "paligemma_vq_droid":             "PG-FSQ",
        "paligemma_fast_droid":           "PG-Fast",
        "paligemma_fast_specialist_droid": "PG-Fast-Spec",
        "pi0_droid":                      r"$\pi_0$",
        "pi0_fast_droid":                 r"$\pi_0$-Fast",
    }
    

    exp_name = args.exp_name

    def results_dir(name):
        return os.path.join('roboarena_outputs_graphical_active', args.subfolder, name) if args.subfolder else os.path.join('outputs', name)

    ## RoboArena 4 with 20 heldout evals per policy
    if exp_name == "roboarena4":
        for beta in beta_range:
            cfg = ExperimentConfig(beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_experiment(cfg=cfg)
            main(policy_data, policies, sim_means, real_means, bernoulli, cfg=cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive)

    ## RoboArena 7 with 20 heldout evals per policy
    if exp_name == "roboarena7":
        for beta in beta_range:
            seven_cfg = ExperimentConfig(beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_experiment(selected_policies=seven_policies, cfg=seven_cfg)
            main(policy_data, policies, sim_means, real_means, bernoulli, cfg=seven_cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive)

    ## RoboArena WM experiment with 4 policies
    if exp_name == "roboarena4_wm_prior":
        for beta in beta_range:
            WM_cfg = ExperimentConfig(n_prior=0, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=four_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=WM_cfg)
            main(policy_data, policies, sim_means, real_means, bernoulli, cfg=WM_cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive)

    ## RoboArena WM experiment with 7 policies
    if exp_name == "roboarena7_wm_prior":
        for beta in beta_range:
            WM_cfg = ExperimentConfig(n_prior=0, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=seven_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=WM_cfg)
            main(policy_data, policies, sim_means, real_means, bernoulli, cfg=WM_cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive)

    ## Synthetic experiment: many policies with true means spread over [0, 1],
    ## sim prior and real evals both generated from the same Gaussians.
    if exp_name == "synthetic":
        for beta in beta_range:
            # n_prior=0: the sim prior is generated separately (not split from real evals).
            syn_cfg = ExperimentConfig(n_prior=0, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = synthetic_experiment(
                n_policies=args.n_policies, n_sim=args.n_sim, n_real=args.n_real,
                std=args.std, seed=args.seed, cfg=syn_cfg)
            syn_labels = {p: f"P{i}" for i, p in enumerate(policies)}
            main(policy_data, policies, sim_means, real_means, bernoulli, cfg=syn_cfg, labels=syn_labels, methods=methods, allow_transitive=args.transitive)

    elif exp_name == "all_policies_fixed":
        for beta in beta_range:
            all_cfg = ExperimentConfig(beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_experiment(cfg=all_cfg)
            main(policy_data, policies, sim_means, real_means, bernoulli, cfg=all_cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive)