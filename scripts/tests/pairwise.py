from sequentialized_barnard_tests.nsm_graphical import MirroredContinuousNsmTest_AlphaAdaptive, ContinuousNsmTest
from sequentialized_barnard_tests.base import Decision, Hypothesis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
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

def pairwise_tests_w_plot(policy_data, alpha, p0_index, p1_index, **kwargs):
    data0 = policy_data[p0_index]
    data1 = policy_data[p1_index]

    # test_for_alternative alternative is P0<P1
    # test_for_null alternative is P0>P1
    mirrored_nsm_test = MirroredContinuousNsmTest_AlphaAdaptive(
        alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=np.arange(11)/10
    )

    nsm_alt = ContinuousNsmTest(
        alternative=Hypothesis.P0LessThanP1,
        alpha=alpha,
        c=np.arange(11) / 10,
    )
    nsm_null = ContinuousNsmTest(
        alternative=Hypothesis.P0MoreThanP1,
        alpha=alpha,
        c=np.arange(11) / 10,
    )

    p_values_alt_mirrored = []
    p_values_null_mirrored = []
    p_values_less_mirrored = []
    p_values_more_mirrored = []
    p_values_less_null_mirrored = []
    p_values_more_null_mirrored = []

    p_values_alt_attr_mirrored = []
    p_values_null_attr_mirrored = []

    p_values_alt = []
    p_values_null = []
    p_values_null_attr = []
    p_values_alt_attr = []
    ks = []

    k = 0
    while True:

        d0_1, d1_1 = data0[k], data1[k]
        if np.isnan(d0_1).any() or np.isnan(d1_1).any():
            break

        res_mirrored = mirrored_nsm_test.step(d0_1, d1_1)
        p_value_alt_mirrored = res_mirrored.info['result_for_alternative'].info['P-Value']
        p_value_null_mirrored = res_mirrored.info['result_for_null'].info['P-Value']
        p_value_alt_attr_mirrored = mirrored_nsm_test._test_for_alternative._p_value
        p_value_null_attr_mirrored = mirrored_nsm_test._test_for_null._p_value

        # p_value_less corresponds to P0<P1 happens to be the test for alternative 
        p_value_less_alt_mirrored = mirrored_nsm_test._test_for_alternative._p_value_less
        p_value_more_alt_mirrored = mirrored_nsm_test._test_for_alternative._p_value_more
        p_value_less_null_mirrored = mirrored_nsm_test._test_for_null._p_value_less
        p_value_more_null_mirrored = mirrored_nsm_test._test_for_null._p_value_more

        p_values_alt_mirrored.append(p_value_alt_mirrored)
        p_values_null_mirrored.append(p_value_null_mirrored)
        p_values_alt_attr_mirrored.append(p_value_alt_attr_mirrored)
        p_values_null_attr_mirrored.append(p_value_null_attr_mirrored)
        p_values_less_mirrored.append(p_value_less_alt_mirrored)
        p_values_more_mirrored.append(p_value_more_alt_mirrored)
        p_values_less_null_mirrored.append(p_value_less_null_mirrored)
        p_values_more_null_mirrored.append(p_value_more_null_mirrored)

        # res_alt = nsm_alt.step(d0_1, d1_1)
        # p_value_alt = res_alt.info["P-Value"]
        # p_value_attr = nsm_alt._p_value

        # res_null = nsm_null.step(d0_1, d1_1)
        # p_value_null = res_null.info["P-Value"]
        # p_value_null_attr = nsm_null._p_value

        ks.append(k)
        # p_values_alt.append(p_value_alt)
        # p_values_null.append(p_value_null)
        # p_values_null_attr.append(p_value_null_attr)
        # p_values_alt_attr.append(p_value_attr)

        k += 1

        if p_value_alt_mirrored < alpha or p_value_null_mirrored < alpha:
            print(f"Reject null hypothesis at k={k} with p-value {p_value_alt_mirrored:.4f}")
            
            break

        # if p_value_alt < alpha:
        #     print(f"Reject null hypothesis at k={k} with p-value {p_value_alt:.4f}")
        #     break
        # if p_value_null < alpha:
        #     print(f"Reject alternative hypothesis at k={k} with p-value {p_value_null:.4f}")
        #     break
    # Plot p-values
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    # Null hypothesis test
    # axs[0].plot(ks, p_values_less_null_mirrored, label="Attribute 1", lw=2)
    axs[0].plot(ks, p_values_more_null_mirrored, label="Attribute", lw=2)
    axs[0].axhline(alpha, color="red", ls=":")
    axs[0].set_ylabel("P-value")
    axs[0].set_title(r"$H_1: P_0 > P_1$")
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    # Alternative hypothesis test
    axs[1].plot(ks, p_values_less_mirrored, label="Attribute", lw=2)
    # axs[1].plot(ks, p_values_more_mirrored, label="Attribute 2", lw=2, ls="--")
    axs[1].axhline(alpha, color="red", ls=":")
    axs[1].set_ylabel("P-value")
    axs[1].set_xlabel("Number of Samples")
    axs[1].set_title(r"$H_1: P_0 < P_1$")
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("p_values_separate.png", dpi=300)
    plt.close()

    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axs[0].plot(ks, p_values_null_mirrored, label="Mirrored (Null)", lw=2, ls="-.")
    axs[0].plot(ks, p_values_null_attr_mirrored, label="Mirrored (Null; Attribute)", lw=2, ls="-.")
    # Null hypothesis test
    axs[0].axhline(alpha, color="red", ls=":")
    axs[0].set_ylabel("P-value")
    axs[0].set_title(r"$H_1: P_0 < P_1$")
    axs[0].legend()
    axs[0].grid(alpha=0.3)
    # Null hypothesis test
    

    # Alternative hypothesis test
    axs[1].plot(ks, p_values_alt_mirrored, label="Mirrored (Alternative)", lw=2)
    axs[1].plot(ks, p_values_alt_attr_mirrored, label="Mirrored (Alt; Attribute)", lw=2, ls="--")
    axs[1].axhline(alpha, color="red", ls=":")
    axs[1].set_ylabel("P-value")
    axs[1].set_xlabel("Number of Samples")
    axs[1].set_title(r"$H_1: P_0 > P_1$")
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("p_values_mirrored.png", dpi=300)
    plt.close()
    

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
        data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "roboarena", data_bin, filename)
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
    real_means = {k: np.mean(policy_data[k]) for k in selected_policies}
    return policy_data, selected_policies, real_means

if __name__ == "__main__":
    # Example usage
    alpha = 0.05
    p0_index = 1
    p1_index = 2

    seven_policies = ["paligemma_binning_droid",
            "pi0_droid",
            "paligemma_diffusion_droid",
            "paligemma_vq_droid",
            "paligemma_fast_specialist_droid",
            "paligemma_fast_droid",
            "pi0_fast_droid"]
    
    policy_data, policies, real_means = roboarena_wm_experiment(selected_policies=seven_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv')
    pairwise_tests_w_plot(policy_data, alpha, policies[p0_index], policies[p1_index])