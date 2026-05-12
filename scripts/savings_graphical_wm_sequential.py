'''
Apurva Badithela, Mar 16, 2026

To find the upper bound on expected efficiency gains from fixed sequence testing
'''

import numpy as np
import os
import tqdm
import tempfile
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from multitest.sequential_graphical import SequentialGraphicalTest
from multitest.sequential_graphical_evalue import SequentialGraphicalTest as ESequentialGraphicalTest

#############################################
# Parameters
#############################################
Nmax = 500
n_runs = 10
alpha = 0.1
FIGSIZE = (12, 10)
# beta = -1.0: higher diffs: larger weight, beta = 1: in proportion to diffs.
beta = 1.0  # tuning parameter for alpha allocation in graphical test
plot_from_saved = False  # set to True to plot from saved data
run_new_experiment = True  # set to False to plot from saved data
results_dir = 'incorrect_order'
os.makedirs(results_dir, exist_ok=True)
assert not (plot_from_saved and run_new_experiment), "Cannot both plot from saved and run new experiment"
assert plot_from_saved or run_new_experiment, "Either plot from saved or run new experiment must be True"

task = [1]
policy = "DP_test"


def get_real_and_sim_means(tasks=[1], policy="pi0"):
    """
    Returns the real and simulated means for a given task and policy from PlayWorld simulation
    """
    real_sims = []
    for task in tasks:
        if task == 1:
            if policy == "pi0":
                real_sim = [(0.15, 0.19), (0.6, 0.25)]
            elif policy == "DP":
                # real_sim = [(0.05, 0.13), (0.3, 0.15), (0.45, 0.22), (0.8, 0.7), (0.81, 0.83)]
                real_sim = [(0.05, 0.13), (0.3, 0.15), (0.45, 0.22), (0.8, 0.7)]
            elif policy == "DP_test":
                real_sim = [(0.05, 0.22), (0.3, 0.15), (0.45, 0.13), (0.8, 0.7)]
                # real_sim = [(0.05, 0.7), (0.3, 0.22), (0.45, 0.15), (0.8, 0.13)]
            elif policy == "all":
                real_sim = [(0.05, 0.13), (0.15, 0.19), (0.3, 0.15), (0.45, 0.22), (0.6, 0.25), (0.8, 0.7), (0.8, 0.83)]
            real_sims.extend(real_sim)
        elif task == 2:
            if policy == "pi0":
                real_sim = [(0.05, 0), (0.45, 0)]
            elif policy == "DP":
                # real_sim = [(0.15, 0.2), (0.5, 0.45), (0.65, 0.42), (0.3, 0.2)]
                real_sim = [(0.15, 0.2), (0.5, 0.45), (0.65, 0.42), (0.35, 0.2)]
            elif policy == "all":
                real_sim = [(0.05, 0), (0.15, 0.2), (0.3, 0.2), (0.45, 0), (0.5, 0.45), (0.65, 0.42), (0.8, 0.7)]
            real_sims.extend(real_sim)
        elif task == 3:
            if policy == "pi0":
                real_sim = [(0.3, 0.2), (0.85, 0.9)]
            elif policy == "DP":
                real_sim = [(0, 0.1), (0.55, 0.62), (0.63, 0.61), (1, 0.95)]
            elif policy == "all":
                real_sim = [(0, 0.1), (0.3, 0.2), (0.55, 0.62), (0.63, 0.61), (0.85, 0.9), (1, 0.95)]
            real_sims.extend(real_sim)
    return real_sims


def synthetic_real_data(real_means, Nmax=200):
    """
    Generate synthetic real data for each policy based on the provided real means.

    Each policy's data is generated as Bernoulli samples with the corresponding mean.

    Parameters
    ----------
    real_means : list of floats, one per policy
    Nmax       : maximum number of samples to generate per policy

    Returns
    -------
    policy_data : np.ndarray of shape (Nmax, n_policies)
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
    rejected_hypotheses : list of indices into null_hypotheses that were rejected
    null_hypotheses     : list of tuples (mu0, mu1) for all null hypotheses
    policy_index        : dict mapping policy index to (real_mean, sim_mean)

    Returns
    -------
    policy_ranking : list of policy indices ranked from best to worst based on rejections
    wins           : dict of win counts per policy
    """
    n_policies = len(policy_index)
    rejected_hypotheses_pairs = [null_hypotheses[k] for k in rejected_hypotheses]
    print("Rejected hypothesis (mu0, mu1): ", rejected_hypotheses_pairs)
    wins = {i: 0 for i in range(n_policies)}
    for p0_sim, p1_sim in rejected_hypotheses_pairs:
        p0_index = next((idx for idx, (_, sim) in policy_index.items() if sim == p0_sim), None)
        p1_index = next((idx for idx, (_, sim) in policy_index.items() if sim == p1_sim), None)
        if p0_index is not None and p1_index is not None:
            wins[p1_index] += 1
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
    beta         : weighting exponent; 0=uniform, >0 favors large diffs, <0 favors small diffs

    Returns
    -------
    alphas : np.ndarray of allocated alpha values, one per hypothesis
    """
    diffs = np.array(diffs, dtype=float)
    if np.any(diffs < 0):
        raise ValueError("All differences must be non-negative.")
    diffs[diffs == 0] = 1e-2  # replace zeros to make exponential weights well-defined
    if total_alpha <= 0 or total_alpha > 1:
        raise ValueError("total_alpha must be in (0, 1].")
    weights = np.exp(diffs * beta)
    return total_alpha * (weights / weights.sum())


def allocate_weights(diffs: list[float], beta=1) -> np.ndarray:
    """
    Allocate edge weights in the graphical test in proportion to the differences |mu0 - mu1|.

    Hypotheses with larger differences receive more weight, so that rejecting them frees up
    more alpha for the remaining hypotheses.

    Parameters
    ----------
    diffs : list of absolute differences |mu0 - mu1| for each hypothesis
    beta  : tuning parameter for weighting; higher beta means more weight on larger differences

    Returns
    -------
    weights : np.ndarray of allocated weights, sums to 1
    """
    diffs = np.array(diffs, dtype=float)
    if np.any(diffs < 0):
        raise ValueError("All differences must be non-negative.")
    weights = np.exp(diffs * beta)
    return weights / weights.sum()


def animate(graph_axes: list, results_dir: str, Nmax: int, n_runs: int):
    """
    Animate a sequence of matplotlib Axes containing networkx graphs.

    Renders each ax to an image array and saves an animated GIF showing
    the graphical test state evolving as hypotheses are rejected.

    Parameters
    ----------
    graph_axes  : list of matplotlib.axes.Axes, one per rejection step
    results_dir : directory to save the GIF
    Nmax        : max samples (used in filename)
    n_runs      : number of runs (used in filename)
    """
    if not graph_axes:
        return
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


def plot_heatmap(matrix, title, save_path, real_sim_means, fmt='.2f'):
    """Render a labelled heatmap and save it to disk."""
    n = matrix.shape[0]
    xlabels = [f"{real_sim_means[i][1]:.2f}" for i in range(n)]
    ylabels = xlabels[::-1]
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


def _accumulate_samples(samples_dict, run, decision_times):
    for hyp, ttd in decision_times.items():
        pi1, pi2 = hyp
        samples_dict[run][pi1] = max(ttd, samples_dict[run].get(pi1, 0))
        samples_dict[run][pi2] = max(ttd, samples_dict[run].get(pi2, 0))


def _average_dict(d, n):
    for key in d:
        d[key] /= n


def _fill_exchange_matrix(avg_ttd_dict, n_policies):
    matrix = np.zeros((n_policies, n_policies))
    for (p0_index, p1_index), ttd in avg_ttd_dict.items():
        p1_index_ex = n_policies - 1 - p1_index
        matrix[p0_index, p1_index_ex] = ttd
    return matrix


def main():
    if isinstance(task, list):
        real_sim_means = get_real_and_sim_means(tasks=task, policy=policy)
    elif task == "all":
        real_sim_means = (
            get_real_and_sim_means(tasks=task, policy=policy)
            + get_real_and_sim_means(task=2, policy=policy)
            + get_real_and_sim_means(task=3, policy=policy)
        )

    n_policies = len(real_sim_means)
    num_hypotheses = n_policies * (n_policies - 1) // 2
    real_means = [real for real, _ in real_sim_means]
    policy_index = {i: real_sim_means[i] for i in range(n_policies)}

    null_hypotheses = [
        (real_sim_means[i][1], real_sim_means[j][1])
        for i in range(n_policies) for j in range(i + 1, n_policies)
    ]
    null_hypotheses_policy_indices = [
        (i, j) for i in range(n_policies) for j in range(i + 1, n_policies)
    ]

    successor_neighbors = {
        (hyp_idx, hyp): [
            (other_hyp_idx, other_hyp)
            for other_hyp_idx, other_hyp in enumerate(null_hypotheses)
            if hyp_idx != other_hyp_idx
        ]
        for hyp_idx, hyp in enumerate(null_hypotheses)
    }

    with open(os.path.join(results_dir, 'successor_neighbors.txt'), 'w') as f:
        for key, neighbors in successor_neighbors.items():
            f.write(f"{key}: {neighbors}\n")

    print("Ordered policy pairs (by time to decision on null hypothesis mu0 < mu1): ")
    hyp_diffs = [abs(mu1 - mu0) for (mu0, mu1) in null_hypotheses]

    alpha_per_hypothesis = allocate_alpha(hyp_diffs, alpha, beta=beta)
    alpha_per_hypothesis_weighted_bonferroni = allocate_alpha(hyp_diffs, alpha, beta=-beta)

    weighted_G = np.zeros((num_hypotheses, num_hypotheses))
    for k1, hyp1 in enumerate(null_hypotheses):
        neighboring_hypotheses = [neighbor[1] for neighbor in successor_neighbors[(k1, hyp1)]]
        neighbor_diffs = [abs(mu1 - mu0) for (mu0, mu1) in neighboring_hypotheses]
        weights_hyp1 = allocate_weights(neighbor_diffs, beta=beta)
        for idx, (k2, _) in enumerate(successor_neighbors[(k1, hyp1)]):
            weighted_G[k1, k2] = weights_hyp1[idx]

    for i in range(len(weighted_G)):
        assert abs(sum(weighted_G[i]) - 1) <= 1e-3

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
        avg_ttd = {}
        avg_ttd_evalues = {}
        avg_ttd_bonferroni = {}
        avg_ttd_weighted_bonferroni = {}
        avg_ttd_fixed = {}
        samples = {k: {} for k in range(n_runs)}
        samples_evalues = {k: {} for k in range(n_runs)}
        samples_fixed = {k: {} for k in range(n_runs)}
        samples_bonferroni = {k: {} for k in range(n_runs)}
        samples_weighted_bonferroni = {k: {} for k in range(n_runs)}

        for run in tqdm.tqdm(range(n_runs)):
            policy_data = synthetic_real_data(real_means=real_means, Nmax=Nmax)

            for i in range(n_policies):
                print(f"Policy {i} mean: ", np.mean(policy_data[:, i]))
            print("------------------------")

            # Graphical multitest
            print("Running graphical multitest...")
            graphical_test = SequentialGraphicalTest(
                num_policies=policy_data.shape[1],
                null_hypotheses=null_hypotheses_policy_indices,
                total_alpha=alpha,
            )
            rejected_hypotheses, rejected_hypotheses_indices, decision_times, p_values, G, graphs_over_time = (
                graphical_test.sequential_graphical_multitest(
                    null_hypotheses_policy_indices, policy_data, Nmax,
                    alpha_per_hypothesis=alpha_per_hypothesis,
                    weighted_G=weighted_G, verbose=True,
                )
            )
            _accumulate_samples(samples, run, decision_times)
            animate(graphs_over_time, results_dir, Nmax, n_runs)
            for key, value in decision_times.items():
                avg_ttd[key] = avg_ttd.get(key, 0) + value
            print("Rejected hypotheses (policy pairs): ", rejected_hypotheses)
            print("Decision times: ", decision_times, "\n")

            # E-value graphical multitest
            print("Running e-value graphical multitest...")
            egraphical_test = ESequentialGraphicalTest(
                num_policies=policy_data.shape[1],
                null_hypotheses=null_hypotheses_policy_indices,
                total_alpha=alpha,
            )
            _, _, decision_times_evalues, _, _, graphs_over_time_evalues = (
                egraphical_test.sequential_graphical_multitest(
                    null_hypotheses_policy_indices, policy_data, Nmax,
                    alpha_per_hypothesis=alpha_per_hypothesis,
                    weighted_G=weighted_G, verbose=True,
                )
            )
            _accumulate_samples(samples_evalues, run, decision_times_evalues)
            animate(graphs_over_time_evalues, results_dir, Nmax, n_runs)
            for key, value in decision_times_evalues.items():
                avg_ttd_evalues[key] = avg_ttd_evalues.get(key, 0) + value
            print("Decision times: ", decision_times_evalues, "\n")

            # Bonferroni
            print("Running Bonferroni corrected individual tests for comparison...")
            rejected_hypotheses_bonferroni, decision_times_bonferroni = graphical_test.bonferroni_multitest(
                null_hypotheses_policy_indices, policy_data, Nmax, alpha
            )
            _accumulate_samples(samples_bonferroni, run, decision_times_bonferroni)
            for key, value in decision_times_bonferroni.items():
                avg_ttd_bonferroni[key] = avg_ttd_bonferroni.get(key, 0) + value
            print("Rejected hypotheses (policy pairs): ", rejected_hypotheses_bonferroni)
            print("Decision times: ", decision_times_bonferroni, "\n")

            # Weighted Bonferroni
            print("Running weighted Bonferroni corrected individual tests for comparison...")
            rejected_hypotheses_weighted_bonferroni, decision_times_weighted_bonferroni = (
                graphical_test.weighted_bonferroni_multitest(
                    null_hypotheses_policy_indices, policy_data, Nmax,
                    alpha_per_hypothesis_weighted_bonferroni,
                )
            )
            _accumulate_samples(samples_weighted_bonferroni, run, decision_times_weighted_bonferroni)
            for key, value in decision_times_weighted_bonferroni.items():
                avg_ttd_weighted_bonferroni[key] = avg_ttd_weighted_bonferroni.get(key, 0) + value
            print("Rejected hypotheses (policy pairs): ", rejected_hypotheses_weighted_bonferroni)
            print("Decision times: ", decision_times_weighted_bonferroni, "\n")

            # Fixed sequence
            print("Running Fixed sequence individual tests for comparison...")
            rejected_hypotheses_fixed, decision_times_fixed = graphical_test.fixed_multitest(
                null_hypotheses_policy_indices, policy_data, Nmax, alpha
            )
            _accumulate_samples(samples_fixed, run, decision_times_fixed)
            for key, value in decision_times_fixed.items():
                avg_ttd_fixed[key] = avg_ttd_fixed.get(key, 0) + value
            print("Rejected hypotheses (policy pairs): ", rejected_hypotheses_fixed)
            print("Decision times: ", decision_times_fixed, "\n")

        policy_ranking, wins = get_ranking_from_rejections(
            rejected_hypotheses_indices, null_hypotheses, policy_index
        )
        print("Policy ranking based on graphical multitest rejections: ", policy_ranking)

        rejected_hypotheses_bonferroni_indices = [
            null_hypotheses.index((policy_index[p0][1], policy_index[p1][1]))
            for (p0, p1) in rejected_hypotheses_bonferroni
            if (policy_index[p0][1], policy_index[p1][1]) in null_hypotheses
        ]
        policy_ranking_bonferroni, wins_bonferroni = get_ranking_from_rejections(
            rejected_hypotheses_bonferroni_indices, null_hypotheses, policy_index
        )
        print("Policy ranking based on Bonferroni corrected test rejections: ", policy_ranking_bonferroni)

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

        _average_dict(avg_ttd, n_runs)
        _average_dict(avg_ttd_evalues, n_runs)
        _average_dict(avg_ttd_bonferroni, n_runs)
        _average_dict(avg_ttd_fixed, n_runs)
        _average_dict(avg_ttd_weighted_bonferroni, n_runs)

        with open(f'{results_dir}/sample_complexity_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.txt', 'w') as f:
            for i in range(n_policies):
                g = np.mean([samples[run].get(i, 0) for run in range(n_runs)])
                eg = np.mean([samples_evalues[run].get(i, 0) for run in range(n_runs)])
                b = np.mean([samples_bonferroni[run].get(i, 0) for run in range(n_runs)])
                fx = np.mean([samples_fixed[run].get(i, 0) for run in range(n_runs)])
                wb = np.mean([samples_weighted_bonferroni[run].get(i, 0) for run in range(n_runs)])
                f.write(f"Policy {i}: Graphical={g:.2f}, EGraphical={eg:.2f}, Bonferroni={b:.2f}, Fixed={fx:.2f}, Weighted_Bonferroni={wb:.2f}\n")

        #############################################
        # Summarizing results
        #############################################
        exchange_pvalue_matrix = np.zeros((n_policies, n_policies))
        exchange_alpha_matrix = np.zeros((n_policies, n_policies))
        for i, (p0_index, p1_index) in enumerate(null_hypotheses_policy_indices):
            p1_index_ex = n_policies - 1 - p1_index
            exchange_pvalue_matrix[p0_index, p1_index_ex] = p_values[i]
            exchange_alpha_matrix[p0_index, p1_index_ex] = alpha_per_hypothesis[i]

        plot_heatmap(
            exchange_pvalue_matrix,
            title='Graphical Multitest: p-values for each hypothesis',
            save_path=f'{results_dir}/graphical_multitest_pvalues_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.png',
            real_sim_means=real_sim_means, fmt='.2e',
        )
        plot_heatmap(
            exchange_alpha_matrix,
            title='Graphical Multitest: alpha allocated for each hypothesis',
            save_path=f'{results_dir}/graphical_multitest_alpha_allocation_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.png',
            real_sim_means=real_sim_means, fmt='.2e',
        )

        np.save(f'{results_dir}/graphical_multitest_samples_per_policy_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy', samples)
        np.save(f'{results_dir}/bonferroni_multitest_samples_per_policy_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy', samples_bonferroni)

        exchange_matrix_fs = _fill_exchange_matrix(avg_ttd, n_policies)
        exchange_matrix_evalues = _fill_exchange_matrix(avg_ttd_evalues, n_policies)
        exchange_matrix_bonferroni = _fill_exchange_matrix(avg_ttd_bonferroni, n_policies)
        exchange_matrix_fixed = _fill_exchange_matrix(avg_ttd_fixed, n_policies)
        exchange_matrix_weighted_bonferroni = _fill_exchange_matrix(avg_ttd_weighted_bonferroni, n_policies)

        for (p0_index, p1_index), ttd in avg_ttd.items():
            print(f"Policy pair ({p0_index}, {p1_index}): FST TTD = {ttd}, Bonferroni TTD = {avg_ttd_bonferroni.get((p0_index, p1_index), Nmax)}")
            print("------------------------")

        exchange_matrix_saved = np.zeros((n_policies, n_policies))
        total_trials_saved_fs = 0
        for (p0_index, p1_index), ttd in avg_ttd.items():
            avg_bonferroni_ttd = avg_ttd_bonferroni.get((p0_index, p1_index), Nmax)
            trials_saved = avg_bonferroni_ttd - ttd
            total_trials_saved_fs += trials_saved
            p1_index_ex = n_policies - 1 - p1_index
            exchange_matrix_saved[p0_index, p1_index_ex] = trials_saved

        np.save(f'{results_dir}/graphical_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy', exchange_matrix_fs)
        np.save(f'{results_dir}/bonferroni_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy', exchange_matrix_bonferroni)
        np.save(f'{results_dir}/weighted_bonferroni_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy', exchange_matrix_weighted_bonferroni)
        np.save(f'{results_dir}/graphical_multitest_exchange_matrix_saved_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy', exchange_matrix_saved)
        np.save(f'{results_dir}/fixed_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy', exchange_matrix_fixed)
        np.save(f'{results_dir}/evalues_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy', exchange_matrix_evalues)

    if plot_from_saved:
        exchange_matrix_saved = np.load(f'{results_dir}/graphical_multitest_exchange_matrix_saved_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy')
        exchange_matrix_fs = np.load(f'{results_dir}/graphical_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy')
        exchange_matrix_bonferroni = np.load(f'{results_dir}/bonferroni_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy')
        exchange_matrix_weighted_bonferroni = np.load(f'{results_dir}/weighted_bonferroni_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy')
        exchange_matrix_fixed = np.load(f'{results_dir}/fixed_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy')
        exchange_matrix_evalues = np.load(f'{results_dir}/evalues_multitest_exchange_matrix_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.npy')
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
    plot_heatmap(
        exchange_matrix_saved,
        title=f'Graphical Multitest: Trials Saved ({total_trials_saved_fs:.2f}) Compared to Bonferroni',
        save_path=f'{results_dir}/graphical_multitest_saved_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.png',
        real_sim_means=real_sim_means,
    )
    plot_heatmap(
        exchange_matrix_fs,
        title='Graphical Multitest: Time to Decision',
        save_path=f'{results_dir}/graphical_multitest_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.png',
        real_sim_means=real_sim_means,
    )
    plot_heatmap(
        exchange_matrix_bonferroni,
        title='Bonferroni: Time to Decision',
        save_path=f'{results_dir}/bonferroni_multitest_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.png',
        real_sim_means=real_sim_means,
    )
    plot_heatmap(
        exchange_matrix_weighted_bonferroni,
        title='Weighted Bonferroni: Time to Decision',
        save_path=f'{results_dir}/weighted_bonferroni_multitest_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.png',
        real_sim_means=real_sim_means,
    )
    plot_heatmap(
        exchange_matrix_fixed,
        title='Fixed: Time to Decision',
        save_path=f'{results_dir}/fixed_multitest_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.png',
        real_sim_means=real_sim_means,
    )
    plot_heatmap(
        exchange_matrix_evalues,
        title='E-Values Graphical: Time to Decision',
        save_path=f'{results_dir}/graphical_evalues_multitest_ttd_N{Nmax}_n{n_runs}_alpha{alpha}_beta{beta}.png',
        real_sim_means=real_sim_means,
    )


if __name__ == '__main__':
    main()
