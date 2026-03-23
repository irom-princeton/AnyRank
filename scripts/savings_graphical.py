'''
Apurva Badithela, Mar 16, 2026

To find the upper bound on expected efficiency gains from fixed sequence testing
'''

import numpy as np
import os
import sys
from sequentialized_barnard_tests import StepTest
from sequentialized_barnard_tests.base import Decision, Hypothesis
from multitest.step_ttd_toy_data import get_step_ttd_toy_data, make_toy_data
from multitest.graphical import graphical_multitest, bonferroni_multitest
import matplotlib.pyplot as plt
import tqdm

#############################################
# Setting up parameters and data
#############################################
Nmax = 500
n_runs = 20
alpha = 0.1
FIGSIZE = (12, 10)
ptune = 0.0  # tuning parameter for alpha allocation in graphical test
plot_from_saved = False  # set to True to plot from saved data
run_new_experiment = True  # set to False to plot from saved data
results_dir = 'graphical_results_weighted_alpha'
os.makedirs(results_dir, exist_ok=True)
assert not (plot_from_saved and run_new_experiment), "Cannot both plot from saved and run new experiment"
assert plot_from_saved or run_new_experiment, "Either plot from saved or run new experiment must be True"

def allocate_alpha(diffs: list[float], total_alpha: float = 0.05, p=1) -> np.ndarray:
    """
    Allocate a total alpha budget across hypotheses in inverse proportion
    to their mean differences |mu0 - mu1|.

    Hypotheses with smaller differences receive more alpha (harder to detect).

    Parameters
    ----------
    diffs        : list of absolute differences |mu0 - mu1| for each hypothesis
    total_alpha  : total alpha budget to distribute (default 0.05)
    Weight for hypothesis i:  w_i = 1 / diff_i ** p

      p = 0   → uniform allocation
      p = 1   → standard inverse weighting (default)
      p > 1   → aggressive: small differences absorb much more alpha
      p < 1   → soft: allocation closer to uniform

    Returns
    -------
    alphas : np.ndarray of allocated alpha values, one per hypothesis
    """
    diffs = np.array(diffs, dtype=float)

    if np.any(diffs < 0):
        raise ValueError("All differences must be non-negative.")
    if np.any(diffs == 0):
        raise ValueError(
            "At least one difference is exactly 0, making inverse weights undefined. "
            "Consider replacing zeros with a small epsilon."
        )
    if total_alpha <= 0 or total_alpha > 1:
        raise ValueError("total_alpha must be in (0, 1].")

    weights = 1.0 / diffs ** p
    alphas = total_alpha * (weights / weights.sum())
    return alphas

#############################################

n_policies = 10
policy_index = {i: (0.1*i, 0.1*i + 0.1) for i in range(n_policies)}  # mapping policy index to mean reward
num_hypotheses = int(n_policies * (n_policies - 1) / 2)
policy_ttd_data = get_step_ttd_toy_data()

# Oracle ordering for fixed sequence testing
# We will represent a hypothesis as a policy pair (p0, p1) and by default use it to indicate H0: p0 <= p1 vs H1: p0 > p1
null_hypotheses = policy_ttd_data.keys() 
ordered_hypotheses = sorted(null_hypotheses, key=lambda x: policy_ttd_data[x])
ordered_hypotheses_policy_indices = []
for hyp in ordered_hypotheses:
    for index, range_means in policy_index.items():
        if range_means[0]< hyp[0] <= range_means[1]:
            p0_index = index
        if range_means[0]< hyp[1] <= range_means[1]:
            p1_index = index
    ordered_hypotheses_policy_indices.append((p0_index, p1_index))

print("Ordered policy pairs (by time to decision on null hypothesis mu0 < mu1): ")
bins = {"10": [], "20": [], "30": [], "40": [], "50": [], "60": [], "70": [], "80": [], "90": []}
diffs = []
for hyp in ordered_hypotheses:
    mu0 = hyp[0]
    mu1 = hyp[1]
    diffs.append(abs(mu1 - mu0))
    if abs(mu1 - mu0) <= 0.1:
        bins["10"].append(hyp)
    elif abs(mu1 - mu0) <= 0.2:
        bins["20"].append(hyp)
    elif abs(mu1 - mu0) <= 0.3:
        bins["30"].append(hyp)
    elif abs(mu1 - mu0) <= 0.4:
        bins["40"].append(hyp)
    elif abs(mu1 - mu0) <= 0.5:
        bins["50"].append(hyp)
    elif abs(mu1 - mu0) <= 0.6:
        bins["60"].append(hyp)
    elif abs(mu1 - mu0) <= 0.7:
        bins["70"].append(hyp)
    elif abs(mu1 - mu0) <= 0.8:
        bins["80"].append(hyp)
    elif abs(mu1 - mu0) <= 0.9:
        bins["90"].append(hyp)
    print(f"Pair mu0 = {mu0:.2f}, mu1 = {mu1:.2f}: mean TTD = {policy_ttd_data[hyp]}")
# total hypotheses:
print(sum([len(bins[key]) for key in bins.keys()]))
alpha_per_hypothesis = allocate_alpha(diffs, alpha,p=ptune) # example alpha allocation for graphical test

# plot histogram of bins
fig, ax = plt.subplots()
ax.bar(bins.keys(), [len(bins[key]) for key in bins.keys()])
# add hypothesis fraction allocated on top of bars:
# for i, (key, value) in enumerate(bins.items()):
#     ax.text(key, value + 0.5, f"{alpha_hyp_unique[::-1][i]:.4f}", ha='center')
ax.set_xlabel("Difference in means (binned)")
ax.set_ylabel("Number of hypotheses")
ax.set_title("Distribution of hypotheses by difference in means")
plt.savefig(f'{results_dir}/hypothesis_distribution.png', dpi=500, bbox_inches='tight')
print("------------------------")

#####################################################
# Looping over multiple runs to generate policy data
#####################################################

if run_new_experiment:
    average_times_to_decision = {}
    average_times_to_decision_bonferroni = {}

    for run in tqdm.tqdm(range(n_runs)):
        # Generate toy data for policies
        policy_data = make_toy_data(num_policies=n_policies, Nmax=Nmax)

        # Verify policy mean
        for i in range(n_policies):
            mean_policy_i = np.mean(policy_data[:,i])
            print(f"Policy {i} mean: ", mean_policy_i)
        print("------------------------")

        # Running graphical multitest
        print("Running graphical multitest...")
        rejected_hypotheses, decision_times, p_values = graphical_multitest(ordered_hypotheses_policy_indices, policy_data, Nmax, alpha, alpha_per_hypothesis=alpha_per_hypothesis)
        for key, value in decision_times.items():
            average_times_to_decision[key] = average_times_to_decision.get(key, 0) + value

        print("Rejected hypotheses (policy pairs): ", rejected_hypotheses)
        print("Decision times: ", decision_times, "\n")

        print("Running Bonferroni corrected individual tests for comparison...")
        rejected_hypotheses_bonferroni, decision_times_bonferroni = bonferroni_multitest(ordered_hypotheses_policy_indices, policy_data, Nmax, alpha)
        for key, value in decision_times_bonferroni.items():
            average_times_to_decision_bonferroni[key] = average_times_to_decision_bonferroni.get(key, 0) + value

        print("Rejected hypotheses (policy pairs): ", rejected_hypotheses_bonferroni)
        print("Decision times: ", decision_times_bonferroni, "\n")

    # Averaging over runs
    for key in average_times_to_decision.keys():
        average_times_to_decision[key] /= n_runs
    for key in average_times_to_decision_bonferroni.keys():
        average_times_to_decision_bonferroni[key] /= n_runs

    #############################################
    # Summarizing results
    #############################################
    # PLot exchange matrix showing pvalue and alpha:
    exchange_pvalue_matrix = np.zeros((n_policies, n_policies))
    exchange_alpha_matrix = np.zeros((n_policies, n_policies))
    for i, (p0_index, p1_index) in enumerate(ordered_hypotheses_policy_indices):
        p1_index_ex = n_policies - 1 - p1_index  # to align with earlier indexing
        exchange_pvalue_matrix[p0_index, p1_index_ex] = p_values[i]
        exchange_alpha_matrix[p0_index, p1_index_ex] = alpha_per_hypothesis[i]
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    cax = ax.matshow(exchange_pvalue_matrix, cmap='viridis')
    for i in range(exchange_pvalue_matrix.shape[0]):
        for j in range(exchange_pvalue_matrix.shape[1]):
            val = exchange_pvalue_matrix[i, j]
            # Use white text on dark cells, black on light cells
            norm_val = (val - exchange_pvalue_matrix.min()) / (exchange_pvalue_matrix.max() - exchange_pvalue_matrix.min())
            text_color = 'white' if norm_val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2e}', ha='center', va='center', 
                    color=text_color, fontsize=int(min(FIGSIZE)))
    plt.colorbar(cax)
    plt.xlabel('mu0')
    plt.ylabel('mu1')
    plt.title(f'Graphical Multitest: p-values for each hypothesis')
    plt.xticks(range(n_policies), [f"{0.1*i + 0.05:.2f}" for i in range(n_policies)], rotation=90)
    plt.yticks(range(n_policies), [f"{0.1*i + 0.05:.2f}" for i in range(n_policies)][::-1])
    plt.savefig(f'{results_dir}/graphical_multitest_pvalues_N{Nmax}_n{n_runs}_alpha{alpha}_p{ptune}.png', dpi=500, bbox_inches='tight')  
    plt.close()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    cax = ax.matshow(exchange_alpha_matrix, cmap='viridis')
    for i in range(exchange_alpha_matrix.shape[0]):
        for j in range(exchange_alpha_matrix.shape[1]):
            val = exchange_alpha_matrix[i, j]
            # Use white text on dark cells, black on light cells
            norm_val = (val - exchange_alpha_matrix.min()) / (exchange_alpha_matrix.max() - exchange_alpha_matrix.min())
            text_color = 'white' if norm_val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2e}', ha='center', va='center',
                    color=text_color, fontsize=int(min(FIGSIZE)))
    plt.colorbar(cax)
    plt.xlabel('mu0')
    plt.ylabel('mu1')
    plt.title(f'Graphical Multitest: alpha allocated for each hypothesis')
    plt.xticks(range(n_policies), [f"{0.1*i + 0.05:.2f}" for i in range(n_policies)], rotation=90)
    plt.yticks(range(n_policies), [f"{0.1*i + 0.05:.2f}" for i in range(n_policies)][::-1])
    plt.savefig(f'{results_dir}/graphical_multitest_alpha_allocation_N{Nmax}_n{n_runs}_alpha{alpha}_p{ptune}.png', dpi=500, bbox_inches='tight')
    plt.close()

# Print an exchange matrix showing time to decision savings compared to Bonferroni correction:
    exchange_matrix_fs = np.zeros((n_policies, n_policies))
    exchange_matrix_saved = np.zeros((n_policies, n_policies))
    exchange_matrix_bonferroni = np.zeros((n_policies, n_policies))

    for (p0_index, p1_index), ttd in average_times_to_decision.items():
        p1_index_ex = n_policies - 1 - p1_index  # to align with earlier indexing
        exchange_matrix_fs[p0_index, p1_index_ex] = ttd
        print(f"Policy pair ({p0_index}, {p1_index}): FST TTD = {ttd}, Bonferroni TTD = {average_times_to_decision_bonferroni.get((p0_index, p1_index), Nmax)}")
        print("------------------------")


    exchange_matrix_bonferroni = np.zeros((n_policies, n_policies))
    for (p0_index, p1_index), ttd in average_times_to_decision_bonferroni.items():
        p1_index_ex = n_policies - 1 - p1_index  # to align with earlier indexing
        exchange_matrix_bonferroni[p0_index, p1_index_ex] = ttd

    total_trials_saved_fs = 0
    for (p0_index, p1_index), ttd in average_times_to_decision.items():
        avg_bonferroni_ttd = average_times_to_decision_bonferroni.get((p0_index, p1_index), Nmax)
        trials_saved = avg_bonferroni_ttd - ttd  # savings in
        total_trials_saved_fs += trials_saved
        p1_index_ex = n_policies - 1 - p1_index  # to align with earlier indexing
        exchange_matrix_saved[p0_index, p1_index_ex] = trials_saved

    np.save(f'{results_dir}/graphical_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_p{ptune}.npy', exchange_matrix_fs)
    np.save(f'{results_dir}/bonferroni_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_p{ptune}.npy', exchange_matrix_bonferroni)
    np.save(f'{results_dir}/graphical_multitest_exchange_matrix_saved_N{Nmax}_n{n_runs}_alpha{alpha}_p{ptune}.npy', exchange_matrix_saved)

if plot_from_saved:
    exchange_matrix_saved = np.load(f'{results_dir}/graphical_multitest_exchange_matrix_saved_N{Nmax}_n{n_runs}_alpha{alpha}_p{ptune}.npy')
    exchange_matrix_fs = np.load(f'{results_dir}/graphical_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_p{ptune}.npy')
    exchange_matrix_bonferroni = np.load(f'{results_dir}/bonferroni_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_p{ptune}.npy')
    total_trials_saved_fs = np.sum(exchange_matrix_saved)

print("Exchange matrix for graphical multitest (time to decision): ")
print(exchange_matrix_fs)
print("------------------------")

print("Exchange matrix for Bonferroni corrected individual tests (time to decision): ")
print(exchange_matrix_bonferroni)
print("------------------------")

print("Exchange matrix for graphical multitest (trials saved): ")
print(exchange_matrix_saved)
print(f"Total trials saved (graphical multitest): {total_trials_saved_fs}")
print("------------------------")

#############################################
# Plotting exchange matrix with heatmap
xlabels = [f"{0.1*i + 0.05:.2f}" for i in range(n_policies)]
ylabels = [f"{0.1*i + 0.05:.2f}" for i in range(n_policies)][::-1]

fig, ax = plt.subplots(figsize=FIGSIZE)
cax = ax.matshow(exchange_matrix_saved, cmap='viridis')
for i in range(exchange_matrix_saved.shape[0]):
    for j in range(exchange_matrix_saved.shape[1]):
        val = exchange_matrix_saved[i, j]
        # Use white text on dark cells, black on light cells
        norm_val = (val - exchange_matrix_saved.min()) / (exchange_matrix_saved.max() - exchange_matrix_saved.min())
        text_color = 'white' if norm_val < 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                color=text_color, fontsize=int(min(FIGSIZE)))
fig.colorbar(cax)
ax.set_xticks(range(n_policies))
ax.set_yticks(range(n_policies))
ax.set_xlabel('mu0')
ax.set_ylabel('mu1')
ax.set_title(f'Graphical Multitest: Trials Saved ({total_trials_saved_fs:.2f}) Compared to Bonferroni')
ax.set_xticklabels(xlabels)
ax.set_yticklabels(ylabels)
plt.savefig(f'{results_dir}/graphical_multitest_saved_N{Nmax}_n{n_runs}_alpha{alpha}_p{ptune}.png',dpi=500, bbox_inches='tight')
plt.close()

# Plot exchange matrix for graphical multitest time to decision
fig, ax = plt.subplots(figsize=FIGSIZE)
cax = ax.matshow(exchange_matrix_fs, cmap='viridis')
for i in range(exchange_matrix_fs.shape[0]):
    for j in range(exchange_matrix_fs.shape[1]):
        val = exchange_matrix_fs[i, j]
        # Use white text on dark cells, black on light cells
        norm_val = (val - exchange_matrix_fs.min()) / (exchange_matrix_fs.max() - exchange_matrix_fs.min())
        text_color = 'white' if norm_val < 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                color=text_color, fontsize=int(min(FIGSIZE)))
fig.colorbar(cax)
ax.set_xlabel('mu0')
ax.set_ylabel('mu1')
ax.set_title(f'Graphical Multitest: Time to Decision')
ax.set_xticks(range(n_policies))
ax.set_yticks(range(n_policies))
ax.set_xticklabels(xlabels)
ax.set_yticklabels(ylabels)
plt.savefig(f'{results_dir}/graphical_multitest_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_p{ptune}.png',dpi=500, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=FIGSIZE)
cax = ax.matshow(exchange_matrix_bonferroni, cmap='viridis')
for i in range(exchange_matrix_bonferroni.shape[0]):
    for j in range(exchange_matrix_bonferroni.shape[1]):
        val = exchange_matrix_bonferroni[i, j]
        # Use white text on dark cells, black on light cells
        norm_val = (val - exchange_matrix_bonferroni.min()) / (exchange_matrix_bonferroni.max() - exchange_matrix_bonferroni.min())
        text_color = 'white' if norm_val < 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                color=text_color, fontsize=int(min(FIGSIZE)))
fig.colorbar(cax)
ax.set_xlabel('mu0')
ax.set_ylabel('mu1')
ax.set_title(f'Bonferroni: Time to Decision')
ax.set_xticks(range(n_policies))
ax.set_yticks(range(n_policies))
ax.set_xticklabels(xlabels)
ax.set_yticklabels(ylabels)
plt.savefig(f'{results_dir}/bonferroni_multitest_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_p{ptune}.png',dpi=500, bbox_inches='tight')
plt.close()
#############################################