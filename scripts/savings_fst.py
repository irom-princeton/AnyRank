'''
Apurva Badithela, Jan 10, 2026

To find the upper bound on expected efficiency gains from fixed sequence testing
'''

import numpy as np
import os
import sys
from sequentialized_barnard_tests import StepTest
from sequentialized_barnard_tests.base import Decision, Hypothesis
from multitest.step_ttd_toy_data import get_step_ttd_toy_data, make_toy_data
from multitest.fixed_sequence import fixed_sequence_multitest_step, bonferroni_multitest
import matplotlib.pyplot as plt
import tqdm

#############################################
# Setting up parameters and data
#############################################
Nmax = 250
n_runs = 10
alpha = 0.05
plot_from_saved = False  # set to True to plot from saved data
run_new_experiment = True  # set to False to plot from saved data
os.makedirs('results', exist_ok=True)
assert not (plot_from_saved and run_new_experiment), "Cannot both plot from saved and run new experiment"
assert plot_from_saved or run_new_experiment, "Either plot from saved or run new experiment must be True"
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
for hyp in ordered_hypotheses:
    mu0 = hyp[0]
    mu1 = hyp[1]
    print(f"Pair mu0 = {mu0:.2f}, mu1 = {mu1:.2f}: mean TTD = {policy_ttd_data[hyp]}")
print("------------------------")

#############################################
# Looping over multiple runs to generate policy data
#############################################

if run_new_experiment:
    average_times_to_decision_fs = {}
    average_times_to_decision_bonferroni = {}

    for run in tqdm.tqdm(range(n_runs)):
        # Generate toy data for policies
        policy_data = make_toy_data(num_policies=n_policies, Nmax=Nmax)

        # Verify policy mean:
        for i in range(n_policies):
            mean_policy_i = np.mean(policy_data[:,i])
            print(f"Policy {i} mean: ", mean_policy_i)
        print("------------------------")

        # Running fixed sequence multitest
        print("Running fixed sequence multitest...")
        rejected_hypotheses, decision_times = fixed_sequence_multitest_step(ordered_hypotheses_policy_indices, policy_data, Nmax, alpha)
        for key, value in decision_times.items():
            average_times_to_decision_fs[key] = average_times_to_decision_fs.get(key, 0) + value

        print("Rejected hypotheses (policy pairs): ", rejected_hypotheses)
        print("Decision times: ", decision_times, "\n")

        print("Running Bonferroni corrected individual tests for comparison...")
        rejected_hypotheses_bonferroni, decision_times_bonferroni = bonferroni_multitest(ordered_hypotheses_policy_indices, policy_data, Nmax, alpha)
        for key, value in decision_times_bonferroni.items():
            average_times_to_decision_bonferroni[key] = average_times_to_decision_bonferroni.get(key, 0) + value

        print("Rejected hypotheses (policy pairs): ", rejected_hypotheses_bonferroni)
        print("Decision times: ", decision_times_bonferroni, "\n")

    # Averaging over runs
    for key in average_times_to_decision_fs.keys():
        average_times_to_decision_fs[key] /= n_runs
    for key in average_times_to_decision_bonferroni.keys():
        average_times_to_decision_bonferroni[key] /= n_runs

#############################################
# Summarizing results
#############################################
# Print an exchange matrix showing time to decision savings compared to Bonferroni correction:

    exchange_matrix_fs = np.zeros((n_policies, n_policies))
    exchange_matrix_saved = np.zeros((n_policies, n_policies))
    exchange_matrix_bonferroni = np.zeros((n_policies, n_policies))

    for (p0_index, p1_index), ttd in average_times_to_decision_fs.items():
        p1_index_ex = n_policies - 1 - p1_index  # to align with earlier indexing
        exchange_matrix_fs[p0_index, p1_index_ex] = ttd
        print(f"Policy pair ({p0_index}, {p1_index}): FST TTD = {ttd}, Bonferroni TTD = {average_times_to_decision_bonferroni.get((p0_index, p1_index), Nmax)}")
        print("------------------------")


    exchange_matrix_bonferroni = np.zeros((n_policies, n_policies))
    for (p0_index, p1_index), ttd in average_times_to_decision_bonferroni.items():
        p1_index_ex = n_policies - 1 - p1_index  # to align with earlier indexing
        exchange_matrix_bonferroni[p0_index, p1_index_ex] = ttd

    total_trials_saved_fs = 0
    for (p0_index, p1_index), ttd in average_times_to_decision_fs.items():
        avg_bonferroni_ttd = average_times_to_decision_bonferroni.get((p0_index, p1_index), Nmax)
        trials_saved = avg_bonferroni_ttd - ttd  # savings in
        total_trials_saved_fs += trials_saved
        p1_index_ex = n_policies - 1 - p1_index  # to align with earlier indexing
        exchange_matrix_saved[p0_index, p1_index_ex] = trials_saved

    np.save(f'results/fixed_sequence_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}.npy', exchange_matrix_fs)
    np.save(f'results/bonferroni_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}.npy', exchange_matrix_bonferroni)
    np.save(f'results/fixed_sequence_multitest_exchange_matrix_saved_N{Nmax}_n{n_runs}_alpha{alpha}.npy', exchange_matrix_saved)

if plot_from_saved:
    exchange_matrix_saved = np.load(f'results/fixed_sequence_multitest_exchange_matrix_saved_N{Nmax}_n{n_runs}_alpha{alpha}.npy')
    exchange_matrix_fs = np.load(f'results/fixed_sequence_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}.npy')
    exchange_matrix_bonferroni = np.load(f'results/bonferroni_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}.npy')
    total_trials_saved_fs = np.sum(exchange_matrix_saved)
    
print("Exchange matrix for fixed sequence multitest (time to decision): ")
print(exchange_matrix_fs)
print("------------------------")

print("Exchange matrix for Bonferroni corrected individual tests (time to decision): ")
print(exchange_matrix_bonferroni)
print("------------------------")

print("Exchange matrix for fixed sequence multitest (trials saved): ")
print(exchange_matrix_saved)
print(f"Total trials saved (fixed sequence multitest): {total_trials_saved_fs}")
print("------------------------")

#############################################
# Plotting exchange matrix with heatmap
xlabels = [f"{0.1*i + 0.05:.2f}" for i in range(n_policies)]
ylabels = [f"{0.1*i + 0.05:.2f}" for i in range(n_policies)][::-1]

fig, ax = plt.subplots()
cax = ax.matshow(exchange_matrix_saved, cmap='viridis')
fig.colorbar(cax)
ax.set_xticks(range(n_policies))
ax.set_yticks(range(n_policies))
ax.set_xlabel('mu0')
ax.set_ylabel('mu1')
ax.set_title(f'FST Multitest: Trials Saved ({total_trials_saved_fs:.2f}) Compared to Bonferroni')
ax.set_xticklabels(xlabels)
ax.set_yticklabels(ylabels)
plt.savefig(f'results/fst_saved_N{Nmax}_n{n_runs}_alpha{alpha}.png',dpi=500, bbox_inches='tight')

# Plot exchange matrix for fixed sequence multitest time to decision
fig, ax = plt.subplots()
cax = ax.matshow(exchange_matrix_fs, cmap='viridis')
fig.colorbar(cax)
ax.set_xlabel('mu0')
ax.set_ylabel('mu1')
ax.set_title(f'FST Multitest: Time to Decision')
ax.set_xticks(range(n_policies))
ax.set_yticks(range(n_policies))
ax.set_xticklabels(xlabels)
ax.set_yticklabels(ylabels)
plt.savefig(f'results/fixed_sequence_multitest_ttd_N{Nmax}_n{n_runs}_alpha{alpha}.png',dpi=500, bbox_inches='tight')

fig, ax = plt.subplots()
cax = ax.matshow(exchange_matrix_bonferroni, cmap='viridis')
fig.colorbar(cax)
ax.set_xlabel('mu0')
ax.set_ylabel('mu1')
ax.set_title(f'Bonferroni: Time to Decision')
ax.set_xticks(range(n_policies))
ax.set_yticks(range(n_policies))
ax.set_xticklabels(xlabels)
ax.set_yticklabels(ylabels)
plt.savefig(f'results/bonferroni_multitest_ttd_N{Nmax}_n{n_runs}_alpha{alpha}.png',dpi=500, bbox_inches='tight')
#############################################


