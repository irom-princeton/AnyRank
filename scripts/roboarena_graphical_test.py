'''
Apurva Badithela, Mar 16, 2026

To find the upper bound on expected efficiency gains from fixed sequence testing
'''

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


def roboarena_experiment(cfg=None):
    """Load roboarena data and return the generic inputs needed by main()."""
    if cfg is None:
        cfg = ExperimentConfig()
    bernoulli = False
    data_bins = ["data2"]
    perfect_sim = True
    selected_policies = ["paligemma_binning_droid", "pi0_droid", "paligemma_diffusion_droid", "pi0_fast_droid"]

    eval_results = get_real_evals(data_bins=data_bins, selected_policies=selected_policies)
    get_policy_summary(eval_results)
    Nmax_csv = min(len(v) for v in eval_results.values())
    prior_evals, _ = split_eval(eval_results, Nmax_csv, nruns=cfg.n_prior)
    policies = list(prior_evals.keys())

    data_progress_filtered_truncated = load_all_data()
    cols = [6, 0, 5, 3]
    policy_data = {p: data_progress_filtered_truncated[:, cols[i]] for i, p in enumerate(policies)}

    if perfect_sim:
        sim_means = {p: np.mean(policy_data[p]) for p in policies}
        real_means = {p: np.mean(policy_data[p]) for p in policies}
    else:
        _, real_evals = split_eval(eval_results, Nmax_csv, nruns=cfg.n_prior)
        sim_means = {k: np.mean(prior_evals[k]) for k in policies}
        real_means = {k: np.mean(real_evals[k]) for k in policies}

    return policy_data, policies, sim_means, real_means, bernoulli


if __name__ == '__main__':
    cfg = ExperimentConfig()
    policy_data, policies, sim_means, real_means, bernoulli = roboarena_experiment(cfg=cfg)
    main(policy_data, policies, sim_means, real_means, bernoulli, cfg=cfg)
