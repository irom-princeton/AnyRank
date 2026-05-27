'''
Apurva Badithela, Mar 16, 2026

To find the upper bound on expected efficiency gains from fixed sequence testing
'''

import argparse
import numpy as np
import os
import copy
import pandas as pd
from multitest.run_graphical_test import (
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

def get_real_evals(data_bins=["data1"], selected_policies=[]):
    """
    Returns the real and simulated means for a given task and policy from PlayWorld simulation
    """
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


def load_all_data(data_bin="data1"):
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data_progress = np.genfromtxt(
        f"{data_dir}/../data/roboarena/{data_bin}/per_trial_progress_data.csv",
        delimiter=",",
        dtype=float,
        skip_header=1,
        filling_values=-1,
    )
    data_preference = np.genfromtxt(
        f"{data_dir}/../data/roboarena/{data_bin}/policy_preferences.csv",
        delimiter=",",
        dtype=str,
        skip_header=1,
        filling_values="missing",
    )

    print()
    print("First 5 rows: ")
    print()
    print(data_progress[:5, :])
    print()

    N0 = data_progress.shape[0]
    n_policies = data_progress.shape[1]
    assert data_preference.shape[0] == N0

    print("N0: ", N0)
    print("N policies: ", n_policies)

    counter = 0
    data_progress_filtered = np.zeros((N0, n_policies))
    for i in range(N0):
        is_complete = True
        if np.min(data_progress[i, :]) < 0:
            is_complete = False
        if is_complete:
            data_progress_filtered[counter, :] = copy.deepcopy(data_progress[i, :])
            counter += 1

    data_progress_filtered_truncated = copy.deepcopy(
        data_progress_filtered[:counter, :]
    )
    return data_progress_filtered_truncated


def roboarena_wm_experiment(selected_policies = None, wm_prior_path=None,cfg=None):
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

    # Load wm_evals from csv: long-format with columns task_id, wm_id, instruction, score
    df_wm = pd.read_csv(wm_prior_path)
    wm_evals = {k: df_wm[df_wm['wm_id'] == k]['score'].to_numpy() for k in policies}
    sim_means = {k: np.mean(wm_evals[k]) for k in policies}
    real_means = {k: np.mean(real_evals[k]) for k in policies}
    return policy_data, policies, sim_means, real_means, bernoulli

def roboarena_experiment(selected_policies = None, cfg=None):
    """Load roboarena data and return the generic inputs needed by main()."""
    if cfg is None:
        cfg = ExperimentConfig()
    bernoulli = False
    data_bins = ["data2"]
    perfect_sim = False
    if not selected_policies:
        selected_policies = ["paligemma_binning_droid", "pi0_droid", "paligemma_diffusion_droid", "pi0_fast_droid"]
    policy_data = get_real_evals(data_bins=data_bins, selected_policies=selected_policies)

    # shuffle data for each policy
    # for k in policy_data.keys():
    #     np.random.shuffle(policy_data[k])

    get_policy_summary(policy_data)
    Nmax_csv = min(len(v) for v in policy_data.values())
    policies = list(policy_data.keys())

    # data dump 1:
    # data_progress_filtered_truncated = load_all_data()
    # cols = [6, 0, 5, 3]
    # policy_data = {p: data_progress_filtered_truncated[:, cols[i]] for i, p in enumerate(policies)}
    # breakpoint()

    if perfect_sim:
        sim_means = {p: np.mean(policy_data[p]) for p in policies}
        real_means = {p: np.mean(policy_data[p]) for p in policies}
    else:
        prior_evals, real_evals = split_eval(policy_data, Nmax_csv, nruns=cfg.n_prior)
        sim_means = {k: np.mean(prior_evals[k]) for k in policies}
        real_means = {k: np.mean(real_evals[k]) for k in policies}
    return policy_data, policies, sim_means, real_means, bernoulli


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subfolder", type=str, default=None, help="Optional subfolder under outputs/<exp_name>/")
    parser.add_argument("--exp_name", type=str, default="roboarena7_wm_prior",
                        choices=["roboarena4", "roboarena7", "roboarena4_wm_prior", "roboarena7_wm_prior", "test_fixed_sequence"],
                        help="Experiment to run")
    parser.add_argument("--graph_type", type=str, default="soft_masked",
                        choices=["soft_masked", "fully_connected"],
                        help="Graph type for alpha transfer weights")
    args = parser.parse_args()

    beta_range = [0, 1, 5, 10, 25, 50]
    four_policies = ["paligemma_binning_droid", "pi0_droid", "paligemma_diffusion_droid", "pi0_fast_droid"]
    seven_policies = ["paligemma_binning_droid",
            "pi0_droid",
            "paligemma_vq_droid",
            "paligemma_fast_specialist_droid",
            "paligemma_fast_droid",
            "paligemma_diffusion_droid",
            "pi0_fast_droid"]

    plot_policy_names = {
        "paligemma_binning_droid":        "PG-Bin",
        "paligemma_diffusion_droid":      "PG-Diff",
        "paligemma_vq_droid":             "PG-VQ",
        "paligemma_fast_droid":           "PG-Fast",
        "paligemma_fast_specialist_droid": "PG-Fast-Spec",
        "pi0_droid":                      r"$\pi_0$",
        "pi0_fast_droid":                 r"$\pi_0$-Fast",
    }

    exp_name = args.exp_name

    def results_dir(name):
        return os.path.join('outputs', args.subfolder, name) if args.subfolder else os.path.join('outputs', name)
    
    ## RoboArena 4 with 20 heldout evals per policy
    if exp_name == "roboarena4":
        for beta in beta_range:
            cfg = ExperimentConfig(beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_experiment(cfg=cfg)
            main(policy_data, policies, sim_means, real_means, bernoulli, cfg=cfg, labels=plot_policy_names)

    ## RoboArena 7 with 20 heldout evals per policy
    if exp_name == "roboarena7":
        for beta in beta_range:
            seven_cfg = ExperimentConfig(beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_experiment(selected_policies=seven_policies, cfg=seven_cfg)
            main(policy_data, policies, sim_means, real_means, bernoulli, cfg=seven_cfg, labels=plot_policy_names)

    ## RoboArena WM experiment with 4 policies
    if exp_name == "roboarena4_wm_prior":
        for beta in beta_range:
            WM_cfg = ExperimentConfig(n_prior=0, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=four_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=WM_cfg)
            main(policy_data, policies, sim_means, real_means, bernoulli, cfg=WM_cfg, labels=plot_policy_names)

    ## RoboArena WM experiment with 7 policies
    if exp_name == "roboarena7_wm_prior":
        for beta in beta_range:
            WM_cfg = ExperimentConfig(n_prior=0, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=seven_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=WM_cfg)
            main(policy_data, policies, sim_means, real_means, bernoulli, cfg=WM_cfg, labels=plot_policy_names)

    if exp_name == "test_fixed_sequence":
        beta = 100
        test_cfg = ExperimentConfig(n_prior=0, beta=beta, results_dir=results_dir(exp_name), graph_type=args.graph_type)
        policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=four_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=test_cfg)
        main(policy_data, policies, sim_means, real_means, bernoulli, cfg=test_cfg, labels=plot_policy_names)
