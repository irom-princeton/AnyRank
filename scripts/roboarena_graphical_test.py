'''
Apurva Badithela, Mar 16, 2026

To find the upper bound on expected efficiency gains from fixed sequence testing
'''

import numpy as np
import os
import copy
import tqdm
import tempfile
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from multitest.sequential_graphical import SequentialGraphicalTest
from multitest.sequential_graphical_evalue import SequentialGraphicalTest as ESequentialGraphicalTest
import pandas as pd

#############################################
# Parameters
#############################################
Nmax = 500
n_runs = 1
n_prior = 20
alpha = 0.1
FIGSIZE = (12, 10)
beta = 1.0  # tuning parameter for alpha allocation in graphical test
plot_from_saved = False  # set to True to plot from saved data
run_new_experiment = True  # set to False to plot from saved data
results_dir = 'outputs/roboarena_subset4'
os.makedirs(results_dir, exist_ok=True)
assert not (plot_from_saved and run_new_experiment), "Cannot both plot from saved and run new experiment"
assert plot_from_saved or run_new_experiment, "Either plot from saved or run new experiment must be True"

def get_real_evals(data_bins=["data1"]):
    """
    Returns the real and simulated means for a given task and policy from PlayWorld simulation
    """
    filename = "per_trial_progress_data.csv"
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
    data_preference_filtered = np.zeros((N0, 3))
    binning_preferred = 0
    binning_not_preferred = 0
    binning_neutral = 0
    for i in range(N0):
        is_complete = True
        if np.min(data_progress[i, :]) < 0:
            is_complete = False
        if is_complete:
            data_progress_filtered[counter, :] = copy.deepcopy(data_progress[i, :])
            # bool0 = [policy == data_preference[i, 0] for policy in policies]
            # idx0 = np.argwhere(bool0)
            # bool1 = [policy == data_preference[i, 1] for policy in policies]
            # idx1 = np.argwhere(bool1)
            # data_preference_filtered[counter, 0] = idx0[0][0]
            # data_preference_filtered[counter, 1] = idx1[0][0]
            # data_preference_filtered[counter, 2] = float(data_preference[i, 2])

            # if idx0 == 6 and float(data_preference[i, 2]) < 0.25:
            #     binning_preferred += 1
            # elif idx0 == 6 and np.isclose(float(data_preference[i, 2]), 0.5):
            #     binning_neutral += 1
            # elif idx1 == 6 and float(data_preference[i, 2]) > 0.75:
            #     binning_not_preferred += 1
            # elif idx1 == 6 and np.isclose(float(data_preference[i, 2]), 0.5):
            #     binning_neutral += 1
            # else:
            #     pass

            # # Update counter
            counter += 1

    data_progress_filtered_truncated = copy.deepcopy(
        data_progress_filtered[:counter, :]
    )
    # data_preference_filtered_truncated = copy.deepcopy(
    #     data_preference_filtered[:counter, :]
    # )
    # return data_preference_filtered_truncated, data_progress_filtered_truncated
    return data_progress_filtered_truncated

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


def animate(graph_axes: list, results_dir: str, Nmax: int, n_prior: int):
    """
    Animate a sequence of matplotlib Axes containing networkx graphs.

    Renders each ax to an image array and saves an animated GIF showing
    the graphical test state evolving as hypotheses are rejected.

    Parameters
    ----------
    graph_axes  : list of matplotlib.axes.Axes, one per rejection step
    results_dir : directory to save the GIF
    Nmax        : max samples (used in filename)
    n_prior      : number of runs (used in filename)
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
        save_path = os.path.join(results_dir, f"graphical_test_animation_N{Nmax}_n{n_prior}.gif")
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

def split_eval(eval_results, Nmax, nruns=10):
    prior_evals = {k: np.array(v[:nruns]) for k, v in eval_results.items()}
    additional_evals = {k: np.array(v[nruns:Nmax]) for k, v in eval_results.items()}
    return prior_evals, additional_evals

def get_policy_summary(eval_results):
    Nmax = min(len(evals) for evals in eval_results.values())
    for k, evals in eval_results.items():
        print(f"Policy {k}: mean={np.mean(evals):.4f}, std={np.std(evals):.4f}, n={len(evals)}")
    return Nmax

def main():
    bernoulli=False
    data_bins = ["data1", "data2"]
    eval_results = get_real_evals(data_bins=data_bins)
    Nmax = get_policy_summary(eval_results)
    prior_evals, real_evals = split_eval(eval_results, Nmax, nruns=n_prior)
    policies = list(prior_evals.keys())
    n_policies = len(prior_evals)
    policy_data = np.zeros((n_policies, Nmax - n_prior))
    
    for i, policy in enumerate(policies):
        policy_data[i] = real_evals[policy]

    policy_data = policy_data.T  # shape (Nmax - n_prior, n_policies)
    data_progress_filtered_truncated = load_all_data()
    policy_data = data_progress_filtered_truncated[:, [6,0,1,2,4,5,3]]  # ensure we only have the policies we expect
    Nmax = min(Nmax, policy_data.shape[0])
    breakpoint()  # check data loading and shapes
    num_hypotheses = n_policies * (n_policies - 1) // 2
    sim_means = [np.mean(prior_evals[k]) for k in policies]
    policy_index = {i: policies[i] for i in range(n_policies)}
    real_means = [np.mean(real_evals[k]) for k in policies]
    real_sim_means = [(real_means[i], sim_means[i]) for i in range(n_policies)]
    
    null_hypotheses = [
        (sim_means[i], sim_means[j])
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
    with open(f'{results_dir}/alpha_allocation_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.txt', 'w') as f:
        for i, null in enumerate(null_hypotheses):
            line = f"Hypothesis {i} (mu0={null[0]:.2f}, mu1={null[1]:.2f}): alpha = {alpha_per_hypothesis[i]:.4f}\n"
            print(line.strip())
            f.write(line)

    #####################################################
    # Looping over multiple runs to generate policy data
    #####################################################
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

    for i, policy in policy_index.items():
        print(f"Policy {policy} mean: ", np.mean(policy_data[:,i]))
    print("------------------------")
    for run in range(n_runs):
        # Graphical multitest
        print("Running graphical multitest...")
        graphical_test = SequentialGraphicalTest(
            num_policies=n_policies,
            null_hypotheses=null_hypotheses_policy_indices,
            total_alpha=alpha,
        )
        rejected_hypotheses, rejected_hypotheses_indices, decision_times, p_values, G, graphs_over_time, alpha_at_rejected, hypotheses_correct = (
            graphical_test.sequential_graphical_multitest(
                null_hypotheses_policy_indices, policy_data, Nmax,
                alpha_per_hypothesis=alpha_per_hypothesis,
                weighted_G=weighted_G, verbose=True,
            )
        )

        _accumulate_samples(samples, run, decision_times)
        animate(graphs_over_time, results_dir, Nmax, n_prior)
        for key, value in decision_times.items():
            avg_ttd[key] = avg_ttd.get(key, 0) + value
        print("Rejected hypotheses (policy pairs): ", rejected_hypotheses)
        print("Decision times: ", decision_times, "\n")
        
        # E-value graphical multitest
        print("Running e-value graphical multitest...")
        egraphical_test = ESequentialGraphicalTest(
            num_policies=n_policies,
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
        animate(graphs_over_time_evalues, results_dir, Nmax, n_prior)
        for key, value in decision_times_evalues.items():
            avg_ttd_evalues[key] = avg_ttd_evalues.get(key, 0) + value
        print("Decision times: ", decision_times_evalues, "\n")

        # Bonferroni
        print("Running Bonferroni corrected individual tests for comparison...")
        rejected_hypotheses_bonferroni, decision_times_bonferroni = graphical_test.bonferroni_multitest(
            null_hypotheses_policy_indices, policy_data, Nmax, alpha, bernoulli=bernoulli
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
                alpha_per_hypothesis_weighted_bonferroni, bernoulli=bernoulli
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
            null_hypotheses_policy_indices, policy_data, Nmax, alpha, bernoulli=bernoulli
        )
        _accumulate_samples(samples_fixed, run, decision_times_fixed)
        for key, value in decision_times_fixed.items():
            avg_ttd_fixed[key] = avg_ttd_fixed.get(key, 0) + value
        print("Rejected hypotheses (policy pairs): ", rejected_hypotheses_fixed)
        print("Decision times: ", decision_times_fixed, "\n")

    # policy_ranking, wins = get_ranking_from_rejections(
    #     rejected_hypotheses_indices, null_hypotheses, policy_index
    # )
    # print("Policy ranking based on graphical multitest rejections: ", policy_ranking)

    # rejected_hypotheses_bonferroni_indices = [
    #     null_hypotheses.index((policy_index[p0][1], policy_index[p1][1]))
    #     for (p0, p1) in rejected_hypotheses_bonferroni
    #     if (policy_index[p0][1], policy_index[p1][1]) in null_hypotheses
    # ]
    # policy_ranking_bonferroni, wins_bonferroni = get_ranking_from_rejections(
    #     rejected_hypotheses_bonferroni_indices, null_hypotheses, policy_index
    # )
    # print("Policy ranking based on Bonferroni corrected test rejections: ", policy_ranking_bonferroni)
    # with open(f'{results_dir}/empirical_real_means_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.txt', 'w') as f:
    #     f.write("Real-sim pairs: \n")
    #     for real_mean, sim_mean in real_sim_means:
    #         f.write(f"Real mean: {real_mean:.4f}, Sim mean: {sim_mean:.4f}\n")
    #     f.write("==============================\n")
    #     f.write("Empirical real means: " + str(np.mean(policy_data, axis=0)) + "\n")
    #     f.write("True real means: " + str(real_means) + "\n")
    #     f.write("==============================\n")
    #     f.write("Policy ranking based on graphical multitest rejections: " + str(policy_ranking) + "\n")
    #     print("Policy ranking based on: ")
    #     for key, value in wins.items():
    #         f.write(f"{key}: {value}\n")
    #     f.write("Policy ranking based on Bonferroni corrected test rejections: " + str(policy_ranking_bonferroni) + "\n")
    #     print("Policy ranking based on: ")
    #     for key, value in wins_bonferroni.items():
    #         f.write(f"{key}: {value}\n")

    _average_dict(avg_ttd, n_runs)
    _average_dict(avg_ttd_evalues, n_runs)
    _average_dict(avg_ttd_bonferroni, n_runs)
    _average_dict(avg_ttd_fixed, n_runs)
    _average_dict(avg_ttd_weighted_bonferroni, n_runs)

    with open(f'{results_dir}/sample_complexity_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.txt', 'w') as f:
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
        save_path=f'{results_dir}/graphical_multitest_pvalues_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.png',
        real_sim_means=real_sim_means, fmt='.2e',
    )
    plot_heatmap(
        exchange_alpha_matrix,
        title='Graphical Multitest: alpha allocated for each hypothesis',
        save_path=f'{results_dir}/graphical_multitest_alpha_allocation_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.png',
        real_sim_means=real_sim_means, fmt='.2e',
    )

    np.save(f'{results_dir}/graphical_multitest_samples_per_policy_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy', samples)
    np.save(f'{results_dir}/bonferroni_multitest_samples_per_policy_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy', samples_bonferroni)

    exchange_matrix_fs = _fill_exchange_matrix(avg_ttd, n_policies)
    exchange_matrix_evalues = _fill_exchange_matrix(avg_ttd_evalues, n_policies)
    exchange_matrix_bonferroni = _fill_exchange_matrix(avg_ttd_bonferroni, n_policies)
    exchange_matrix_fixed = _fill_exchange_matrix(avg_ttd_fixed, n_policies)
    exchange_matrix_weighted_bonferroni = _fill_exchange_matrix(avg_ttd_weighted_bonferroni, n_policies)

    true_sample_complexity_fs = 0.
    true_sample_complexity_evalues = 0.
    true_sample_complexity_bonferroni = 0.
    true_sample_complexity_weighted_bonferroni = 0.
    true_sample_complexity_fixed = 0. 

    for i in range(n_policies):
        true_sample_complexity_fs += np.maximum(np.max(exchange_matrix_fs[:, i]), np.max(exchange_matrix_fs[n_policies-i-1, :]))
        true_sample_complexity_evalues += np.maximum(np.max(exchange_matrix_evalues[:, i]), np.max(exchange_matrix_evalues[n_policies-i-1, :]))
        true_sample_complexity_bonferroni += np.maximum(np.max(exchange_matrix_bonferroni[:, i]), np.max(exchange_matrix_bonferroni[n_policies-i-1, :]))
        true_sample_complexity_weighted_bonferroni += np.maximum(np.max(exchange_matrix_weighted_bonferroni[:, i]), np.max(exchange_matrix_weighted_bonferroni[n_policies-i-1, :]))
        true_sample_complexity_fixed += np.maximum(np.max(exchange_matrix_fixed[:, i]), np.max(exchange_matrix_fixed[n_policies-i-1, :]))
    

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

    np.save(f'{results_dir}/graphical_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy', exchange_matrix_fs)
    np.save(f'{results_dir}/bonferroni_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy', exchange_matrix_bonferroni)
    np.save(f'{results_dir}/weighted_bonferroni_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy', exchange_matrix_weighted_bonferroni)
    np.save(f'{results_dir}/graphical_multitest_exchange_matrix_saved_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy', exchange_matrix_saved)
    np.save(f'{results_dir}/fixed_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy', exchange_matrix_fixed)
    np.save(f'{results_dir}/evalues_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy', exchange_matrix_evalues)

    if plot_from_saved:
        exchange_matrix_saved = np.load(f'{results_dir}/graphical_multitest_exchange_matrix_saved_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy')
        exchange_matrix_fs = np.load(f'{results_dir}/graphical_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy')
        exchange_matrix_bonferroni = np.load(f'{results_dir}/bonferroni_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy')
        exchange_matrix_weighted_bonferroni = np.load(f'{results_dir}/weighted_bonferroni_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy')
        exchange_matrix_fixed = np.load(f'{results_dir}/fixed_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy')
        exchange_matrix_evalues = np.load(f'{results_dir}/evalues_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy')
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
    print()
    print("Actual Sample complexities: ")
    print(f"Complexity of Our Approach:                   {true_sample_complexity_fs}")
    print(f"Complexity of E-values Approach:              {true_sample_complexity_evalues}")
    print(f"Complexity of Bonferroni Approach:            {true_sample_complexity_bonferroni}")
    print(f"Complexity of Weighted Bonferroni Approach:   {true_sample_complexity_weighted_bonferroni}")
    print(f"Complexity of Fixed Sequence Approach:        {true_sample_complexity_fixed}")
    print()
    print("Decision times: ")
    with np.printoptions(precision=3, suppress=True):
        print(decision_times)
    print()
    print("alpha at rejected:")
    with np.printoptions(precision=3, suppress=True):
        print(alpha_at_rejected)
    print("------------------------")

    #############################################
    # Plotting exchange matrices as heatmaps
    #############################################
    plot_heatmap(
        exchange_matrix_saved,
        title=f'Graphical Multitest: Trials Saved ({total_trials_saved_fs:.2f}) Compared to Bonferroni',
        save_path=f'{results_dir}/graphical_multitest_saved_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.png',
        real_sim_means=real_sim_means,
    )
    plot_heatmap(
        exchange_matrix_fs,
        title='Graphical Multitest: Time to Decision',
        save_path=f'{results_dir}/graphical_multitest_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.png',
        real_sim_means=real_sim_means,
    )
    plot_heatmap(
        exchange_matrix_bonferroni,
        title='Bonferroni: Time to Decision',
        save_path=f'{results_dir}/bonferroni_multitest_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.png',
        real_sim_means=real_sim_means,
    )
    plot_heatmap(
        exchange_matrix_weighted_bonferroni,
        title='Weighted Bonferroni: Time to Decision',
        save_path=f'{results_dir}/weighted_bonferroni_multitest_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.png',
        real_sim_means=real_sim_means,
    )
    plot_heatmap(
        exchange_matrix_fixed,
        title='Fixed: Time to Decision',
        save_path=f'{results_dir}/fixed_multitest_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.png',
        real_sim_means=real_sim_means,
    )
    plot_heatmap(
        exchange_matrix_evalues,
        title='E-Values Graphical: Time to Decision',
        save_path=f'{results_dir}/graphical_evalues_multitest_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.png',
        real_sim_means=real_sim_means,
    )


if __name__ == '__main__':
    main()
