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
from multitest.sequential_graphical import SequentialGraphicalTest
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import tempfile, tqdm

#############################################
# Setting up parameters and data
#############################################
Nmax = 500
n_runs = 20
alpha = 0.1
FIGSIZE = (12, 10)
beta = 1.0  # tuning parameter for alpha allocation in graphical test
plot_from_saved = False  # set to True to plot from saved data
run_new_experiment = True  # set to False to plot from saved data
results_dir = 'sequential_results_exp_weights_wm'
os.makedirs(results_dir, exist_ok=True)
assert not (plot_from_saved and run_new_experiment), "Cannot both plot from saved and run new experiment"
assert plot_from_saved or run_new_experiment, "Either plot from saved or run new experiment must be True"

# Playworld setting:
task = 1
policy = "DP_test"

def get_real_and_sim_means(task=1, policy="pi0"):
    """
    Returns the real and simulated means for a given task and policy from PlayWorld simulation
    """
    if task==1:
        if policy == "pi0":
            real_sim = [(0.15, 0.19), (0.6, 0.25)]
        elif policy == "DP":
            real_sim = [(0.05, 0.13), (0.3, 0.15), (0.45, 0.22), (0.8, 0.7), (0.8, 0.83)]
        elif policy == "DP_test":
            real_sim = [(0.05, 0.13), (0.3, 0.15), (0.45, 0.22), (0.8, 0.7)]
        elif policy == "all":
            real_sim = [(0.05, 0.13), (0.15, 0.19), (0.3, 0.15), (0.45, 0.22), (0.6, 0.25), (0.8, 0.7), (0.8, 0.83)]

    elif task==2:
        if policy == "pi0":
            real_sim = [(0.05, 0), (0.45, 0)]
        elif policy == "DP":
            real_sim = [(0.15, 0.2), (0.5, 0.45), (0.65, 0.42), (0.3, 0.2)]
        elif policy == "all":
            real_sim = [(0.05, 0), (0.15, 0.2), (0.3, 0.2), (0.45, 0), (0.5, 0.45), (0.65, 0.42), (0.8, 0.7)]
    elif task==3:
        if policy == "pi0":
            real_sim = [ (0.3, 0.2), (0.85, 0.9)]
        elif policy == "DP":
            real_sim = [(0, 0.1), (0.55, 0.62), (0.63, 0.61), (1, 0.95)]
        elif policy == "all":
            real_sim = [(0, 0.1), (0.3, 0.2), (0.55, 0.62), (0.63, 0.61), (0.85, 0.9), (1, 0.95)]
    return real_sim

def synthetic_real_data(real_means, Nmax=200):
    """
    Generate synthetic real data for each policy based on the provided real means.

    Each policy's data is generated as Bernoulli samples with the corresponding mean.

    Parameters
    ----------
    real_means : list of tuples (real_mean) for each policy
    Nmax       : maximum number of samples to generate per policy

    Returns
    -------
    policy_data : np.ndarray of shape (Nmax, n_policies) containing the synthetic real data
    """
    n_policies = len(real_means)
    policy_data = np.zeros((Nmax, n_policies))

    for i, real_mean in enumerate(real_means):
        policy_data[:, i] = np.random.binomial(1, real_mean, size=Nmax)

    return policy_data

def get_ranking_from_rejections(rejected_hypotheses, null_hypotheses, policy_index):
    """
    Derive a ranking of policies based on which null hypotheses were rejected.
    Null hypothesis is mu0 <= mu1, so if it is rejected, it means mu1 > mu0, so policy 1 is better than policy 0.
    Parameters
    ----------
    rejected_hypotheses : list of tuples (p0_index, p1_index) for rejected hypotheses
    null_hypotheses     : list of tuples (mu0, mu1) for all null hypotheses

    Returns
    -------
    policy_ranking : list of policy indices ranked from best to worst based on rejections
    """
    # Count wins for each policy based on rejected hypotheses
    n_policies = len(policy_index)
    rejected_hypotheses_pairs = [null_hypotheses[k] for k in rejected_hypotheses]
    print("Rejected hypothesis (mu0, mu1): ", rejected_hypotheses_pairs)
    wins = {i: 0 for i in range(n_policies)}
    for p0_sim, p1_sim in rejected_hypotheses_pairs:
        p0_index = None
        p1_index = None
        for idx, (real, sim) in policy_index.items():
            if sim == p0_sim:
                p0_index = idx
            if sim == p1_sim:
                p1_index = idx
        if p0_index is not None and p1_index is not None:
            wins[p1_index] += 1  # policy 1 wins over policy 0
    # Rank policies based on wins
    policy_ranking = sorted(wins.keys(), key=lambda x: wins[x], reverse=True)
    return policy_ranking, wins

def allocate_alpha(diffs: list[float], total_alpha: float = 0.05, beta=1) -> np.ndarray:
    """
    Allocate a total alpha budget across hypotheses in inverse proportion
    to their mean differences |mu0 - mu1|.

    Hypotheses with smaller differences receive more alpha (harder to detect).

    Parameters
    ----------
    diffs        : list of absolute differences |mu0 - mu1| for each hypothesis
    total_alpha  : total alpha budget to distribute (default 0.05)
    Weight for hypothesis i:  w_i = exp(-beta * diff_i)

      beta = 0   → uniform allocation
      beta > 0   → proportional weighting; large differences get exponentially more alpha
      beta < 0   → small difference get exponentially more alpha; as beta → -inf, all alpha goes to the smallest difference hypothesis

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

    weights = np.exp(diffs * beta)
    alphas = total_alpha * (weights / weights.sum())
    return alphas

def allocate_weights(diffs: list[float], beta=1) -> np.ndarray:
    """
    Allocate edge weights in the graphical test in proportion to the differences |mu0 - mu1|.

    Hypotheses with larger differences receive more weight, so that rejecting them frees up more alpha for the remaining hypotheses.

    Parameters
    ----------
    diffs : list of absolute differences |mu0 - mu1| for each hypothesis
    beta  : tuning parameter for weighting; higher beta means more weight on larger differences

    Returns
    -------
    weights : np.ndarray of allocated weights for each hypothesis
    """
    diffs = np.array(diffs, dtype=float)
    if np.any(diffs < 0):
        raise ValueError("All differences must be non-negative.")

    weights = np.exp(diffs * beta)
    weights = weights / weights.sum()  # normalize to sum to 1

    return weights

def animate(graph_axes: list):
    """
    Animate a sequence of matplotlib Axes containing networkx graphs.

    Renders each ax to an image array and saves an animated GIF showing
    the graphical test state evolving as hypotheses are rejected.

    Parameters
    ----------
    graph_axes : list of matplotlib.axes.Axes
        Axes returned by graphical_multitest, one per rejection step.
    """
    if not graph_axes:
        return

    # Save each frame directly at high DPI, then assemble with PIL
    pil_frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, ax in enumerate(graph_axes):
            fig = ax.get_figure()
            frame_path = os.path.join(tmpdir, f"frame_{idx:03d}.png")
            fig.savefig(frame_path, dpi=150, bbox_inches="tight")
            pil_frames.append(PILImage.open(frame_path).copy())
            plt.close(fig)

        save_path = os.path.join(results_dir, f"graphical_test_animation_N{Nmax}_n{n_runs}.gif")
        pil_frames[0].save(
            save_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=1000,
            loop=0,
        )
    print(f"Animation saved to {save_path}")

def plot_heatmap(matrix, title, save_path, fmt='.2f'):
    """Render a labelled heatmap and save it to disk."""
    n = matrix.shape[0]
    xlabels = [f"{real_sim_means[i][1]:.2f}" for i in range(n)]
    ylabels = [f"{real_sim_means[i][1]:.2f}" for i in range(n)][::-1]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    cax = ax.matshow(matrix, cmap='viridis')
    vmin, vmax = matrix.min(), matrix.max()
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            norm_val = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
            text_color = 'white' if norm_val < 0.5 else 'black'
            ax.text(j, i, f'{val:{fmt}}', ha='center', va='center',
                    color=text_color, fontsize=int(min(FIGSIZE)))
    fig.colorbar(cax)
    ax.set_xlabel('mu0')
    ax.set_ylabel('mu1')
    ax.set_title(title)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(xlabels, rotation=90)
    ax.set_yticklabels(ylabels)
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()

#############################################
if isinstance(task, int):
    real_sim_means = get_real_and_sim_means(task=task, policy=policy)
elif task == "all":
    real_sim_means = get_real_and_sim_means(task=1, policy=policy) + get_real_and_sim_means(task=2, policy=policy) + get_real_and_sim_means(task=3, policy=policy)

n_policies = len(real_sim_means)
num_hypotheses = int(n_policies * (n_policies - 1) / 2)
sim_means = [sim for _, sim in real_sim_means]
real_means = [real for real, _ in real_sim_means]
policy_index = {i: (real_sim_means[i][0], real_sim_means[i][1]) for i in range(n_policies)}

# The numbering for the following two must always be consistent
null_hypotheses = [(real_sim_means[i][1], real_sim_means[j][1]) for i in range(n_policies) for j in range(i+1, n_policies)]
null_hypotheses_policy_indices = [(i,j) for i in range(n_policies) for j in range(i+1, n_policies)]

# Here we can define arbitrary graphs where some hypotheses are not connected to others
successor_neighbors = {(hyp_idx, hyp): [(other_hyp_idx, other_hyp) for other_hyp_idx, other_hyp in enumerate(null_hypotheses) if (hyp_idx != other_hyp_idx)] for hyp_idx, hyp in enumerate(null_hypotheses)}


# Print successor neighbors to file:
with open(os.path.join(results_dir, 'successor_neighbors.txt'), 'w') as f:
    for key, neighbors in successor_neighbors.items():
        f.write(f"{key}: {neighbors}\n")

print("Ordered policy pairs (by time to decision on null hypothesis mu0 < mu1): ")
diffs = []
for hyp in null_hypotheses:
    mu0 = hyp[0]
    mu1 = hyp[1]
    diffs.append(abs(mu1 - mu0))

# Allocate alpha according to simulation differences:
alpha_per_hypothesis = allocate_alpha(diffs, alpha,beta=beta) # example alpha allocation for graphical test
alpha_per_hypothesis_weighted_bonferroni = allocate_alpha(diffs, alpha,beta=-1*beta) # example alpha allocation for graphical test

# Allocate weights according to simulation differences:
weighted_G = np.zeros((num_hypotheses, num_hypotheses))
for k1, hyp1 in enumerate(null_hypotheses):
    neighboring_hypotheses = [neighbor[1] for neighbor in successor_neighbors[(k1, hyp1)]]
    diffs = [abs(mu1-mu0) for (mu0, mu1) in neighboring_hypotheses]
    weights_hyp1 = allocate_weights(diffs, beta=beta)
    for idx, (k2, hyp2) in enumerate(successor_neighbors[(k1, hyp1)]):
        weighted_G[k1, k2] = weights_hyp1[idx]

for i in range(len(weighted_G)):
    assert abs(sum(weighted_G[i]) - 1)<=1e-3

# Log to file:
print("Alpha allocated to each hypotheses")
with open(f'{results_dir}/alpha_allocation_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.txt', 'w') as f:
    for i, null in enumerate(null_hypotheses):
        line = f"Hypothesis {i} (mu0={null[0]:.2f}, mu1={null[1]:.2f}): alpha = {alpha_per_hypothesis[i]:.4f}\n"
        print(line.strip())
        f.write(line)

#####################################################
# Looping over multiple runs to generate policy data
#####################################################
if run_new_experiment:
    average_times_to_decision = {}
    average_times_to_decision_bonferroni = {}
    average_times_to_decision_weighted_bonferroni = {}
    average_times_to_decision_fixed = {}
    samples_per_policy = {k: {} for k in range(n_runs)}
    samples_per_policy_fixed = {k: {} for k in range(n_runs)}
    samples_per_policy_bonferroni = {k: {} for k in range(n_runs)}
    samples_per_policy_weighted_bonferroni = {k: {} for k in range(n_runs)}

    for run in tqdm.tqdm(range(n_runs)):
        # Generate toy data for policies
        policy_data = synthetic_real_data(real_means = real_means, Nmax=Nmax)

        # Verify policy mean
        for i in range(n_policies):
            mean_policy_i = np.mean(policy_data[:,i])
            print(f"Policy {i} mean: ", mean_policy_i)
        print("------------------------")

        #######################  Running graphical multitest #######################
        print("Running graphical multitest...")
        graphical_test = SequentialGraphicalTest(num_policies=policy_data.shape[1], null_hypotheses = null_hypotheses_policy_indices, total_alpha=alpha)
        # rejected_hypotheses, decision_times, p_values, G, graphs_over_time = sequential_graphical_multitest(null_hypotheses_policy_indices, policy_data, Nmax, alpha, alpha_per_hypothesis=alpha_per_hypothesis, weighted_G=weighted_G)
        rejected_hypotheses, rejected_hypotheses_indices, decision_times, p_values, G, graphs_over_time = graphical_test.sequential_graphical_multitest(null_hypotheses_policy_indices, policy_data, Nmax, alpha_per_hypothesis=alpha_per_hypothesis, weighted_G=weighted_G, verbose=True)

        for hyp, ttd in decision_times.items():
            pi1, pi2 = hyp
            samples_per_policy[run][pi1] = max(ttd, samples_per_policy[run].get(pi1, 0))
            samples_per_policy[run][pi2] = max(ttd, samples_per_policy[run].get(pi2, 0))


        animate(graphs_over_time)

        for key, value in decision_times.items():
            average_times_to_decision[key] = average_times_to_decision.get(key, 0) + value
        print("Rejected hypotheses (policy pairs): ", rejected_hypotheses)
        print("Decision times: ", decision_times, "\n")

        #######################  Running simple Bonferroni #######################
        print("Running Bonferroni corrected individual tests for comparison...")
        rejected_hypotheses_bonferroni, decision_times_bonferroni = graphical_test.bonferroni_multitest(null_hypotheses_policy_indices, policy_data, Nmax, alpha)
        for key, value in decision_times_bonferroni.items():
            average_times_to_decision_bonferroni[key] = average_times_to_decision_bonferroni.get(key, 0) + value

        for hyp, ttd in decision_times_bonferroni.items():
            pi1, pi2 = hyp
            samples_per_policy_bonferroni[run][pi1] = max(ttd, samples_per_policy_bonferroni[run].get(pi1, 0))
            samples_per_policy_bonferroni[run][pi2] = max(ttd, samples_per_policy_bonferroni[run].get(pi2, 0))

        print("Rejected hypotheses (policy pairs): ", rejected_hypotheses_bonferroni)
        print("Decision times: ", decision_times_bonferroni, "\n")

        #######################  Running weighted Bonferroni #######################
        print("Running weighted Bonferroni corrected individual tests for comparison...")

        rejected_hypotheses_weighted_bonferroni, decision_times_weighted_bonferroni = graphical_test.weighted_bonferroni_multitest(null_hypotheses_policy_indices, policy_data, Nmax, alpha_per_hypothesis_weighted_bonferroni)
        for key, value in decision_times_weighted_bonferroni.items():
            average_times_to_decision_weighted_bonferroni[key] = average_times_to_decision_weighted_bonferroni.get(key, 0) + value

        for hyp, ttd in decision_times_weighted_bonferroni.items():
            pi1, pi2 = hyp
            samples_per_policy_weighted_bonferroni[run][pi1] = max(ttd, samples_per_policy_weighted_bonferroni[run].get(pi1, 0))
            samples_per_policy_weighted_bonferroni[run][pi2] = max(ttd, samples_per_policy_weighted_bonferroni[run].get(pi2, 0))

        print("Rejected hypotheses (policy pairs): ", rejected_hypotheses_weighted_bonferroni)
        print("Decision times: ", decision_times_weighted_bonferroni, "\n")

        #######################  Running fixed sequence testing #######################
        print("Running Fixed sequence individual tests for comparison...")
        rejected_hypotheses_fixed, decision_times_fixed = graphical_test.fixed_multitest(null_hypotheses_policy_indices, policy_data, Nmax, alpha)
        for key, value in decision_times_fixed.items():
            average_times_to_decision_fixed[key] = average_times_to_decision_fixed.get(key, 0) + value

        for hyp, ttd in decision_times_fixed.items():
            pi1, pi2 = hyp
            samples_per_policy_fixed[run][pi1] = max(ttd, samples_per_policy_fixed[run].get(pi1, 0))
            samples_per_policy_fixed[run][pi2] = max(ttd, samples_per_policy_fixed[run].get(pi2, 0))

        print("Rejected hypotheses (policy pairs): ", rejected_hypotheses_fixed)
        print("Decision times: ", decision_times_fixed, "\n")

    policy_ranking, wins = get_ranking_from_rejections(rejected_hypotheses_indices, null_hypotheses, policy_index)
    print("Policy ranking based on graphical multitest rejections: ", policy_ranking)
    rejected_hypotheses_bonferroni_indices = []
    for (p0_index, p1_index) in rejected_hypotheses_bonferroni:
        null_hypothesis = (policy_index[p0_index][1], policy_index[p1_index][1])
        if null_hypothesis in null_hypotheses:
            rejected_hypotheses_bonferroni_indices.append(null_hypotheses.index(null_hypothesis))
    policy_ranking_bonferroni, wins_bonferroni = get_ranking_from_rejections(rejected_hypotheses_bonferroni_indices, null_hypotheses, policy_index)
    print("Policy ranking based on Bonferroni corrected test rejections: ", policy_ranking_bonferroni)

    # Log to file empirical real means and policy rankings:
    with open(f'{results_dir}/empirical_real_means_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.txt', 'w') as f:
        f.write("Real-sim pairs: \n")
        for real_mean, sim_mean in real_sim_means:
            f.write(f"Real mean: {real_mean:.4f}, Sim mean: {sim_mean:.4f}\n")
        f.write("==============================\n")
        f.write("Empirical real means: " + str(np.mean(policy_data, axis=0)) + "\n")
        f.write("True real means: " + str(real_means) + "\n")
        f.write("==============================\n")
        f.write("Policy ranking based on graphical multitest rejections: " + str(policy_ranking) + "\n")
        print("Policy ranking based on: ")
        for key, value in wins.items():
            f.write(f"{key}: {value}\n")
        f.write("Policy ranking based on Bonferroni corrected test rejections: " + str(policy_ranking_bonferroni) + "\n")
        print("Policy ranking based on: ")
        for key, value in wins_bonferroni.items():
            f.write(f"{key}: {value}\n")

    # Averaging over runs
    for key in average_times_to_decision.keys():
        average_times_to_decision[key] /= n_runs
    for key in average_times_to_decision_bonferroni.keys():
        average_times_to_decision_bonferroni[key] /= n_runs
    for key in average_times_to_decision_fixed.keys():
        average_times_to_decision_fixed[key] /= n_runs
    for key in average_times_to_decision_weighted_bonferroni.keys():
        average_times_to_decision_weighted_bonferroni[key] /= n_runs
    # Log average times to decision per policy:
    with open(f'{results_dir}/sample_complexity_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.txt', 'w') as f:
        for i in range(n_policies):
            policy_samples_average = np.mean([samples_per_policy[run].get(i, 0) for run in range(n_runs)])
            policy_samples_average_bonferroni = np.mean([samples_per_policy_bonferroni[run].get(i, 0) for run in range(n_runs)])
            policy_samples_average_fixed = np.mean([samples_per_policy_fixed[run].get(i, 0) for run in range(n_runs)])
            policy_samples_average_weighted_bonferroni = np.mean([samples_per_policy_weighted_bonferroni[run].get(i, 0) for run in range(n_runs)])
            f.write(f"Policy {i}: Graphical={policy_samples_average:.2f}, Bonferroni={policy_samples_average_bonferroni:.2f}, Fixed={policy_samples_average_fixed:.2f}, Weighted_Bonferroni={policy_samples_average_weighted_bonferroni:.2f}\n")

    #############################################
    # Summarizing results
    #############################################
    # PLot exchange matrix showing pvalue and alpha:
    exchange_pvalue_matrix = np.zeros((n_policies, n_policies))
    exchange_alpha_matrix = np.zeros((n_policies, n_policies))
    for i, (p0_index, p1_index) in enumerate(null_hypotheses_policy_indices):
        p1_index_ex = n_policies - 1 - p1_index  # to align with earlier indexing
        exchange_pvalue_matrix[p0_index, p1_index_ex] = p_values[i]
        exchange_alpha_matrix[p0_index, p1_index_ex] = alpha_per_hypothesis[i]

    plot_heatmap(exchange_pvalue_matrix,
                 title='Graphical Multitest: p-values for each hypothesis',
                 save_path=f'{results_dir}/graphical_multitest_pvalues_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.png',
                 fmt='.2e')
    plot_heatmap(exchange_alpha_matrix,
                 title='Graphical Multitest: alpha allocated for each hypothesis',
                 save_path=f'{results_dir}/graphical_multitest_alpha_allocation_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.png',
                 fmt='.2e')

# Save the average sample complexity for policies:
    np.save(f'{results_dir}/graphical_multitest_samples_per_policy_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy', samples_per_policy)
    np.save(f'{results_dir}/bonferroni_multitest_samples_per_policy_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy', samples_per_policy_bonferroni)

# Print an exchange matrix showing time to decision savings compared to Bonferroni correction:
    exchange_matrix_fs = np.zeros((n_policies, n_policies))
    exchange_matrix_saved = np.zeros((n_policies, n_policies))
    exchange_matrix_fixed = np.zeros((n_policies, n_policies))
    exchange_matrix_bonferroni = np.zeros((n_policies, n_policies))
    exchange_matrix_weighted_bonferroni = np.zeros((n_policies, n_policies))

    for (p0_index, p1_index), ttd in average_times_to_decision.items():
        p1_index_ex = n_policies - 1 - p1_index  # to align with earlier indexing
        exchange_matrix_fs[p0_index, p1_index_ex] = ttd
        print(f"Policy pair ({p0_index}, {p1_index}): FST TTD = {ttd}, Bonferroni TTD = {average_times_to_decision_bonferroni.get((p0_index, p1_index), Nmax)}")
        print("------------------------")


    for (p0_index, p1_index), ttd in average_times_to_decision_bonferroni.items():
        p1_index_ex = n_policies - 1 - p1_index  # to align with earlier indexing
        exchange_matrix_bonferroni[p0_index, p1_index_ex] = ttd

    for (p0_index, p1_index), ttd in average_times_to_decision_fixed.items():
        p1_index_ex = n_policies - 1 - p1_index  # to align with earlier indexing
        exchange_matrix_fixed[p0_index, p1_index_ex] = ttd

    for (p0_index, p1_index), ttd in average_times_to_decision_weighted_bonferroni.items():
        p1_index_ex = n_policies - 1 - p1_index  # to align with earlier indexing
        exchange_matrix_weighted_bonferroni[p0_index, p1_index_ex] = ttd

    total_trials_saved_fs = 0
    for (p0_index, p1_index), ttd in average_times_to_decision.items():
        avg_bonferroni_ttd = average_times_to_decision_bonferroni.get((p0_index, p1_index), Nmax)
        trials_saved = avg_bonferroni_ttd - ttd  # savings in
        total_trials_saved_fs += trials_saved
        p1_index_ex = n_policies - 1 - p1_index  # to align with earlier indexing
        exchange_matrix_saved[p0_index, p1_index_ex] = trials_saved

    np.save(f'{results_dir}/graphical_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy', exchange_matrix_fs)
    np.save(f'{results_dir}/bonferroni_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy', exchange_matrix_bonferroni)
    np.save(f'{results_dir}/weighted_bonferroni_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy', exchange_matrix_weighted_bonferroni)
    np.save(f'{results_dir}/graphical_multitest_exchange_matrix_saved_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy', exchange_matrix_saved)

if plot_from_saved:
    exchange_matrix_saved = np.load(f'{results_dir}/graphical_multitest_exchange_matrix_saved_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy')
    exchange_matrix_fs = np.load(f'{results_dir}/graphical_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy')
    exchange_matrix_bonferroni = np.load(f'{results_dir}/bonferroni_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy')
    exchange_matrix_weighted_bonferroni = np.load(f'{results_dir}/weighted_bonferroni_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy')
    total_trials_saved_fs = np.sum(exchange_matrix_saved)

print("Exchange matrix for graphical multitest (time to decision): ")
print(exchange_matrix_fs)
print("------------------------")

print("Exchange matrix for Bonferroni corrected individual tests (time to decision): ")
print(exchange_matrix_bonferroni)
print("------------------------")

print("Exchange matrix for Weighted Bonferroni corrected individual tests (time to decision): ")
print(exchange_matrix_weighted_bonferroni)
print("------------------------")

print("Exchange matrix for graphical multitest (trials saved): ")
print(exchange_matrix_saved)
print(f"Total trials saved (graphical multitest): {total_trials_saved_fs}")
print("------------------------")

#############################################
# Plotting exchange matrices as heatmaps
#############################################
plot_heatmap(exchange_matrix_saved,
             title=f'Graphical Multitest: Trials Saved ({total_trials_saved_fs:.2f}) Compared to Bonferroni',
             save_path=f'{results_dir}/graphical_multitest_saved_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.png')
plot_heatmap(exchange_matrix_fs,
             title='Graphical Multitest: Time to Decision',
             save_path=f'{results_dir}/graphical_multitest_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.png')
plot_heatmap(exchange_matrix_bonferroni,
             title='Bonferroni: Time to Decision',
             save_path=f'{results_dir}/bonferroni_multitest_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.png')
plot_heatmap(exchange_matrix_weighted_bonferroni,
             title='Weighted Bonferroni: Time to Decision',
             save_path=f'{results_dir}/weighted_bonferroni_multitest_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.png')
plot_heatmap(exchange_matrix_fixed,
             title='Fixed: Time to Decision',
             save_path=f'{results_dir}/fixed_multitest_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.png')
#############################################
