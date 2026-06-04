from sequentialized_barnard_tests.nonparametric_nsm import MirroredContinuousNsmTest, ContinuousNsmTest
from sequentialized_barnard_tests.base import Decision, Hypothesis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def pairwise_tests(policy_data, alpha, p0_index, p1_index, **kwargs):
    data0 = policy_data[:, p0_index]
    data1 = policy_data[:, p1_index]

    nsm_alt = ContinuousNsmTest(
        alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=np.arange(11)/10.
    )
    nsm_null = ContinuousNsmTest(
        alternative=Hypothesis.P0MoreThanP1, alpha=alpha, c=np.arange(11)/10.
    )

    k = 0
    while True:
        d0_1, d1_1 = data0[:k], data1[:k]
        if np.isnan(d0_1).any() or np.isnan(d1_1).any():
            break
        res_alt = nsm_alt.step(d0_1, d1_1)
        p_value_alt = res_alt.info["P-Value"]
        res_null = nsm_null.step(d0_1, d1_1)
        p_value_null = res_null.info["P-Value"]
        k += 1
        
        if p_value_alt < alpha:
            print(f"Reject null hypothesis at k={k} with p-value {p_value_alt:.4f}")
            break
        if p_value_null < alpha:
            print(f"Reject alternative hypothesis at k={k} with p-value {p_value_null:.4f}")
            break

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

def roboarena_wm_experiment(selected_policies=None, wm_prior_path=None):
    """Load roboarena data and return the generic inputs needed by main()."""
    bernoulli = False
    data_bins = ["data2"]
    if not selected_policies:
        selected_policies = ["paligemma_binning_droid", "pi0_droid", "paligemma_diffusion_droid", "pi0_fast_droid"]
    policy_data = get_real_evals(data_bins=data_bins, selected_policies=selected_policies)
    Nmax_csv = min(len(v) for v in policy_data.values())
    real_means = {k: np.mean(policy_data[k]) for k in policies}
    return policy_data, policies, sim_means, real_means, bernoulli

if __name__ == "__main__":
    # Example usage
    alpha = 0.5
    p0_index = 3
    p1_index = 4

    seven_policies = ["paligemma_binning_droid",
            "pi0_droid",
            "paligemma_diffusion_droid",
            "paligemma_vq_droid",
            "paligemma_fast_specialist_droid",
            "paligemma_fast_droid",
            "pi0_fast_droid"]
    
    policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=seven_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=WM_cfg)
    pairwise_tests(policy_data, alpha, p0_index, p1_index)