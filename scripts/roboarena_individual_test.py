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
from datetime import datetime

date = datetime.now().strftime("%Y-%m-%d")

np.random.seed(42)

ALL_METHODS = ['fixed', 'graphical', 'bonferroni', 'weighted_bonferroni', 'graphical_active']


def get_real_evals(data_bins=["data1"], selected_policies=[]):
    filename = "per_trial_progress_data.csv"
    if selected_policies:
        keys = selected_policies
    else:
        keys = [
            "paligemma_binning_droid",
            "pi0_droid",
            "paligemma_vq_droid",
            "paligemma_fast_specialist_droid",
            "paligemma_fast_droid",
            "paligemma_diffusion_droid",
            "pi0_fast_droid",
        ]
    eval_results = {k: np.array([]) for k in keys}
    for data_bin in data_bins:
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "roboarena", data_bin, filename)
        df = pd.read_csv(data_path)
        for k in keys:
            eval_results[k] = np.concatenate([eval_results[k], df[k].dropna().to_numpy()])
    return eval_results


def roboarena_wm_experiment(selected_policies=None, wm_prior_path=None, cfg=None):
    """Load roboarena data and return the generic inputs needed by main()."""
    if cfg is None:
        cfg = ExperimentConfig()
    bernoulli = False
    data_bins = ["data2"]
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
    data_bins = ["data2"]
    perfect_sim = False
    if not selected_policies:
        selected_policies = ["paligemma_binning_droid", "pi0_droid", "paligemma_diffusion_droid", "pi0_fast_droid"]
    policy_data = get_real_evals(data_bins=data_bins, selected_policies=selected_policies)

    get_policy_summary(policy_data)
    
    Nmax_csv = min(len(v) for v in policy_data.values())
    policies = list(policy_data.keys())

    if perfect_sim:
        sim_means = {p: np.mean(policy_data[p]) for p in policies}
        real_means = {p: np.mean(policy_data[p]) for p in policies}
        real_evals = {p: policy_data[p] for p in policies}
    else:
        prior_evals, real_evals = split_eval(policy_data, Nmax_csv, nruns=cfg.n_prior)
        sim_means = {k: np.mean(prior_evals[k]) for k in policies}
        real_means = {k: np.mean(real_evals[k]) for k in policies}
    return real_evals, policies, sim_means, real_means, bernoulli


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subfolder", type=str, default=None, help="Optional subfolder under outputs/<exp_name>/")
    parser.add_argument("--exp_name", type=str, default="roboarena7_wm_prior",
                        help="Experiment to run")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Significance level for multiple testing")
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
    args = parser.parse_args()
    if args.subfolder is None:
        args.subfolder = date
    else:
        args.subfolder = f"{args.subfolder}_{date}"

    methods = args.methods  # None means all

    beta_range = [0,1,10]
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
            cfg = ExperimentConfig(alpha=args.alpha, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_experiment(cfg=cfg)
            main(policy_data, policies, sim_means, real_means, bernoulli, cfg=cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive)
    
    if exp_name == "roboarena4_wm_prior_perfect":
            for beta in beta_range:
                cfg = ExperimentConfig(alpha=args.alpha, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
                policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=four_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=cfg)
                sim_means = real_means.copy()  # Use perfect simulation means
                main(policy_data, policies, sim_means, real_means, bernoulli, cfg=cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive)

    if exp_name == "roboarena4_wm_prior_imperfect":
        for beta in beta_range:
            cfg = ExperimentConfig(alpha=args.alpha, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=four_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=cfg)
            real_mean_values = list(real_means.values())
            sim_means["paligemma_binning_droid"]= real_means["pi0_fast_droid"]
            sim_means["pi0_droid"] = real_means["paligemma_diffusion_droid"]
            sim_means["paligemma_diffusion_droid"] = real_means["pi0_droid"]
            sim_means["pi0_fast_droid"] = real_means["paligemma_binning_droid"]
            main(policy_data, policies, sim_means, real_means, bernoulli, cfg=cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive)

    if exp_name == "roboarena4_wm_prior_perfect_reshuffle":
        nruns = 50
        results_dir = results_dir(exp_name)
        for beta in beta_range:
            for run in range(nruns):
                results_dir_run = os.path.join(results_dir, f"run_{run}")
                cfg = ExperimentConfig(alpha=args.alpha, beta=beta, results_dir=results_dir_run, graph_type=args.graph_type)
                policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=four_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=cfg)
                sim_means = real_means.copy()  # Use perfect simulation means
                reshuffled_policy_data = {k: np.random.permutation(v) for k, v in policy_data.items()}
                main(reshuffled_policy_data, policies, sim_means, real_means, bernoulli, cfg=cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive)

    if exp_name == "roboarena4_wm_prior_imperfect_reshuffle":
        nruns = 50
        results_dir = results_dir(exp_name)
        for beta in beta_range:
            for run in range(nruns):
                results_dir_run = os.path.join(results_dir, f"run_{run}")
                cfg = ExperimentConfig(alpha=args.alpha, beta=beta, results_dir=results_dir_run, graph_type=args.graph_type)
                policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=four_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=cfg)
                # Reshuffle the policy data to simulate a different evaluation order
                sim_means["paligemma_binning_droid"]= real_means["pi0_fast_droid"] - 0.1
                sim_means["pi0_droid"] = real_means["paligemma_diffusion_droid"]
                sim_means["paligemma_diffusion_droid"] = real_means["pi0_droid"]
                sim_means["pi0_fast_droid"] = real_means["paligemma_binning_droid"] + 0.3
                reshuffled_policy_data = {k: np.random.permutation(v) for k, v in policy_data.items()}
                main(reshuffled_policy_data, policies, sim_means, real_means, bernoulli, cfg=cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive)

    if exp_name == "roboarena4_reshuffle":
        nruns = 20
        results_dir = results_dir(exp_name)
        for beta in beta_range:
            for run in range(nruns):
                results_dir_run = os.path.join(results_dir, f"run_{run}")
                cfg = ExperimentConfig(alpha=args.alpha, beta=beta, results_dir=results_dir_run, graph_type=args.graph_type)
                policy_data, policies, sim_means, real_means, bernoulli = roboarena_experiment(cfg=cfg)
                # Reshuffle the policy data to simulate a different evaluation order
                reshuffled_policy_data = {k: np.random.permutation(v) for k, v in policy_data.items()}
                main(reshuffled_policy_data, policies, sim_means, real_means, bernoulli, cfg=cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive)
    
    ## RoboArena 7 with 20 heldout evals per policy
    if exp_name == "roboarena7":
        for beta in beta_range:
            seven_cfg = ExperimentConfig(alpha=args.alpha, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_experiment(selected_policies=seven_policies, cfg=seven_cfg)
            main(policy_data, policies, sim_means, real_means, bernoulli, cfg=seven_cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive)

    if exp_name == "roboarena7_reshuffle":
        nruns = 20
        results_dir = results_dir(exp_name)
        for beta in beta_range:
            for run in range(nruns):
                results_dir_run = os.path.join(results_dir, f"run_{run}")
                seven_cfg = ExperimentConfig(alpha=args.alpha, beta=beta, results_dir=results_dir_run, graph_type=args.graph_type)
                policy_data, policies, sim_means, real_means, bernoulli = roboarena_experiment(selected_policies=seven_policies, cfg=seven_cfg)
                # Reshuffle the policy data to simulate a different evaluation order
                reshuffled_policy_data = {k: np.random.permutation(v) for k, v in policy_data.items()}
                main(reshuffled_policy_data, policies, sim_means, real_means, bernoulli, cfg=seven_cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive)

    ## RoboArena WM experiment with 4 policies
    if exp_name == "roboarena4_wm_prior":
        for beta in beta_range:
            WM_cfg = ExperimentConfig(n_prior=0, alpha = args.alpha, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=four_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=WM_cfg)
            main(policy_data, policies, sim_means, real_means, bernoulli, cfg=WM_cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive)

    if exp_name == "roboarena7_bradley_terry_max":
        # Nmax_total = 2450
        Nmax_total = 2553*2
        for beta in beta_range:
            seven_cfg = ExperimentConfig(alpha=args.alpha, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_experiment(selected_policies=seven_policies,cfg=seven_cfg)
            main(policy_data, policies, sim_means, real_means, bernoulli, cfg=seven_cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive, Nmax_total=Nmax_total)

    if exp_name == "roboarena7_bradley_terry_half":
        Nmax_total = 2450
        # Nmax_total = 2553*2
        for beta in beta_range:
            seven_cfg = ExperimentConfig(alpha=args.alpha, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_experiment(selected_policies=seven_policies,cfg=seven_cfg)
            main(cap_samples, policies, sim_means, real_means, bernoulli, cfg=seven_cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive, Nmax_total=Nmax_total)

    if exp_name == "roboarena4_bradley_terry_max":
        Nmax_total = 1200
        for beta in beta_range:
            four_cfg = ExperimentConfig(alpha=args.alpha, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_experiment(selected_policies=four_policies,cfg=four_cfg)
            main(policy_data, policies, sim_means, real_means, bernoulli, cfg=four_cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive, Nmax_total=Nmax_total)
    
    if exp_name == "roboarena4_wm_prior_bradley_terry_max":
        Nmax_total = 1200
        for beta in beta_range:
            WM_cfg = ExperimentConfig(n_prior=0, alpha = args.alpha, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=four_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=WM_cfg)
            main(policy_data, policies, sim_means, real_means, bernoulli, cfg=WM_cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive, Nmax_total=Nmax_total)
    
    if exp_name == "roboarena7_wm_prior_bradley_terry_max":
        # Nmax_total = 2450
        Nmax_total = 2553*2
        for beta in beta_range:
            WM_cfg = ExperimentConfig(n_prior=0, alpha = args.alpha, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=seven_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=WM_cfg)
            main(policy_data, policies, sim_means, real_means, bernoulli, cfg=WM_cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive, Nmax_total=Nmax_total)

    if exp_name == "roboarena7_wm_prior_bradley_terry_half":
        Nmax_total = 2450
        # Nmax_total = 2553*2
        for beta in beta_range:
            WM_cfg = ExperimentConfig(n_prior=0, alpha = args.alpha, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=seven_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=WM_cfg)
            main(cap_samples, policies, sim_means, real_means, bernoulli, cfg=WM_cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive, Nmax_total=Nmax_total)
            
    if exp_name == "roboarena4_wm_prior_reshuffle":
        nruns = 50
        results_dir = results_dir(exp_name)
        for beta in beta_range:
            for run in range(nruns):
                results_dir_run = os.path.join(results_dir, f"run_{run}")
                WM_cfg = ExperimentConfig(n_prior=0, alpha = args.alpha, beta=beta, results_dir=results_dir_run, graph_type=args.graph_type)
                policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=four_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=WM_cfg)
                reshuffled_policy_data = {k: np.random.permutation(v) for k, v in policy_data.items()}
                main(reshuffled_policy_data, policies, sim_means, real_means, bernoulli, cfg=WM_cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive)

    ## RoboArena WM experiment with 7 policies
    if exp_name == "roboarena7_wm_prior":
        for beta in beta_range:
            WM_cfg = ExperimentConfig(n_prior=0, alpha=args.alpha, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=seven_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=WM_cfg)
            main(policy_data, policies, sim_means, real_means, bernoulli, cfg=WM_cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive)

    if exp_name == "roboarena7_wm_prior_0.9":
        for beta in beta_range:
            WM_cfg = ExperimentConfig(alpha=0.9, n_prior=0, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=seven_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=WM_cfg)
            main(policy_data, policies, sim_means, real_means, bernoulli, cfg=WM_cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive)

    if exp_name == "roboarena7_wm_prior_0.1_half_new_run":
        for beta in beta_range:
            WM_cfg = ExperimentConfig(alpha=0.1, n_prior=0, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=seven_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=WM_cfg)
            cap_samples = {k: v[:len(v)//2] for k, v in policy_data.items()}
            # cap_samples = {k: v[:50] for k, v in policy_data.items()}
            main(cap_samples, policies, sim_means, real_means, bernoulli, cfg=WM_cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive)

    if exp_name == "roboarena7_wm_prior_0.1_half_plus_150":
        for beta in beta_range:
            WM_cfg = ExperimentConfig(alpha=0.1, n_prior=0, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=seven_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=WM_cfg)
            cap_samples = {k: v[:len(v)//2+150] for k, v in policy_data.items()}
            # cap_samples = {k: v[:50] for k, v in policy_data.items()}
            main(cap_samples, policies, sim_means, real_means, bernoulli, cfg=WM_cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive)

    if exp_name == "roboarena7_wm_prior_0.1_half_plus_100":
        for beta in beta_range:
            WM_cfg = ExperimentConfig(alpha=0.1, n_prior=0, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=seven_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=WM_cfg)
            cap_samples = {k: v[:len(v)//2+100] for k, v in policy_data.items()}
            # cap_samples = {k: v[:50] for k, v in policy_data.items()}
            main(cap_samples, policies, sim_means, real_means, bernoulli, cfg=WM_cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive)

    if exp_name == "roboarena7_wm_prior_0.1_50":
        for beta in beta_range:
            WM_cfg = ExperimentConfig(alpha=0.1, n_prior=0, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=seven_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=WM_cfg)
            cap_samples = {k: v[:50] for k, v in policy_data.items()}
            # cap_samples = {k: v[:50] for k, v in policy_data.items()}
            main(cap_samples, policies, sim_means, real_means, bernoulli, cfg=WM_cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive)

    if exp_name == "test_fixed_sequence":
        beta = 100
        test_cfg = ExperimentConfig(n_prior=0, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
        policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=four_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=test_cfg)
        main(policy_data, policies, sim_means, real_means, bernoulli, cfg=test_cfg, labels=plot_policy_names, methods=methods, allow_transitive=args.transitive)

