'''
David Snyder,  12 May 2026

Set up a large-scale experiment framework to test FPR control and empirical time-to-full-ranking for our multitest method
'''

import numpy as np
import os
import copy
from tqdm import tqdm
import tempfile
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from src.multitest.sequential_graphical import SequentialGraphicalTest
from src.multitest.sequential_graphical_evalue import SequentialGraphicalTest as ESequentialGraphicalTest
import pandas as pd

#############################################
# Parameters
#############################################
bernoulli = True
n_policies = 5
num_hypotheses = n_policies * (n_policies - 1) // 2
Nmax = 1000
n_runs = 100
n_prior = 20
alpha = 0.05
FIGSIZE = (12, 10)
beta = 5.0  # tuning parameter for alpha allocation in graphical test
plot_from_saved = False  # set to True to plot from saved data
run_new_experiment = True  # set to False to plot from saved data
results_dir = 'outputs/large_scale_graphical_test_results_v4'
os.makedirs(results_dir, exist_ok=True)
assert not (plot_from_saved and run_new_experiment), "Cannot both plot from saved and run new experiment"
assert plot_from_saved or run_new_experiment, "Either plot from saved or run new experiment must be True"

def get_real_evals():
    """
    Returns the real and simulated means for a given task and policy from PlayWorld simulation
    """
    filename = "per_trial_progress_data.csv"
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "roboarena", filename)
    df = pd.read_csv(data_path)
    eval_results = {}
    eval_results["paligemma_binning_droid"] = df["paligemma_binning_droid"].dropna().to_numpy()
    eval_results["pi0_droid"] = df["pi0_droid"].dropna().to_numpy()
    # eval_results["paligemma_vq_droid"] = df["paligemma_vq_droid"].dropna().to_numpy()
    # eval_results["paligemma_fast_specialist_droid"] = df["paligemma_fast_specialist_droid"].dropna().to_numpy()
    # eval_results["paligemma_fast_droid"] = df["paligemma_fast_droid"].dropna().to_numpy()
    eval_results["paligemma_diffusion_droid"] = df["paligemma_diffusion_droid"].dropna().to_numpy()
    eval_results["pi0_fast_droid"] = df["pi0_fast_droid"].dropna().to_numpy()
    return eval_results

def get_synthetic_bernoulli_evals(n_policies, n_max=1000, means=None, make_ranking_impossible=False):
    """
    Returns the real synthetic data
    """
    assert n_policies >= 2

    if make_ranking_impossible:
        if means is None:
            means_base = (np.arange(n_policies-1) + 0.5) / (n_policies - 1.)
            insertion_idx = n_policies // 2
            means = np.zeros(n_policies)
            for i in range(insertion_idx):
                means[i] = means_base[i]
            
            means[insertion_idx] = means[insertion_idx-1]
            for i in range(insertion_idx+1, n_policies):
                means[i] = means_base[i-1]

        else:
            assert np.array(means).shape[0] == n_policies
            assert np.isclose(np.min(np.diff(np.sort(np.array(means)))), 0.)
    else:
        if means is None:
            means = (np.arange(n_policies) + 0.5) / n_policies
        else:
            assert np.array(means).shape[0] == n_policies
    
    eval_results = {}
    for i in range(n_policies):
        p_string = "Policy_" + str(i)
        eval_results[p_string] = np.random.binomial(1, means[i], size=(n_max, ))
    
    return eval_results, means

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
    # for i in range(n_policies):
    #     for j in range(n_policies - i - 1):
    #         if matrix[i, j] <= 0.5:
    #             matrix[i, j] = Nmax
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
    make_ranking_impossible = True
    policy_true_means = None
    ALPHA_AT_REJECTION_SORTED = np.zeros((num_hypotheses, n_runs))
    CORRECT_FULL_RANKING_ALT = []
    CORRECT_PARTIAL_RANKING_NULL = []

    INCOMPLETE_FULL_RANKING_ALT = []
    INCORRECT_FULL_RANKING_ALT = []

    INCOMPLETE_PARTIAL_RANKING_NULL = []
    INCORRECT_PARTIAL_RANKING_NULL = []
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

    for run in tqdm(range(n_runs)):
        #####################
        ### GENERATE DATA ###
        #####################
        if policy_true_means is None:
            eval_results, policy_true_means = get_synthetic_bernoulli_evals(n_policies=n_policies, n_max=Nmax, make_ranking_impossible=make_ranking_impossible)
        else:
            eval_results, policy_true_means = get_synthetic_bernoulli_evals(n_policies=n_policies, n_max=Nmax, means=policy_true_means)
        
        if make_ranking_impossible:
            assert np.isclose(np.min(np.diff(np.sort(policy_true_means))), 0.)
        
        prior_evals, real_evals = split_eval(eval_results, Nmax, nruns=n_prior)
        policies = list(prior_evals.keys())
        policy_data = np.zeros((n_policies, Nmax - n_prior))

        for i, policy in enumerate(policies):
            policy_data[i] = real_evals[policy]

        policy_data = policy_data.T  # shape (Nmax - n_prior, n_policies)

        sim_means = [np.mean(prior_evals[k]) for k in policies]
        policy_index = {i: policies[i] for i in range(n_policies)}
        real_means = [np.mean(real_evals[k]) for k in policies]
        real_sim_means = [(real_means[i], sim_means[i]) for i in range(n_policies)]
        
        null_hypotheses = [
            (sim_means[i], sim_means[j])
            for i in range(n_policies) for j in range(i + 1, n_policies)
        ]
        correct_hypothesis_signs = np.zeros(num_hypotheses)
        counter = 0
        for i in range(n_policies):
            for j in range(i+1, n_policies):
                if policy_true_means[j] > policy_true_means[i]:
                    correct_hypothesis_signs[counter] = 1. 
                elif policy_true_means[j] < policy_true_means[i]:
                    correct_hypothesis_signs[counter] = -1. 
                else:
                    pass
                
                counter += 1
        
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

        # print("Ordered policy pairs (by time to decision on null hypothesis mu0 < mu1): ")
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

        # print("Alpha allocated to each hypotheses")
        with open(f'{results_dir}/alpha_allocation_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.txt', 'w') as f:
            for i, null in enumerate(null_hypotheses):
                line = f"Hypothesis {i} (mu0={null[0]:.2f}, mu1={null[1]:.2f}): alpha = {alpha_per_hypothesis[i]:.4f}\n"
                print(line.strip())
                f.write(line)

        # for i, policy in policy_index.items():
            # print(f"Policy {policy} mean: ", np.mean(policy_data[:,i]))
        
        # print()
        # print("------------------------")
        # print()
        
        # Graphical multitest
        # print("Running graphical multitest...")
        graphical_test = SequentialGraphicalTest(
            num_policies=n_policies,
            null_hypotheses=null_hypotheses_policy_indices,
            total_alpha=alpha,
        )
        rejected_hypotheses, rejected_hypotheses_indices, decision_times, p_values, G, graphs_over_time, alpha_at_rejected, hypothesis_decisions = (
            graphical_test.sequential_graphical_multitest(
                null_hypotheses_policy_indices, policy_data, Nmax,
                alpha_per_hypothesis=alpha_per_hypothesis,
                weighted_G=weighted_G, verbose=False,
            )
        )

        if np.isclose(np.linalg.norm(correct_hypothesis_signs - hypothesis_decisions), 0.) and np.min(np.abs(correct_hypothesis_signs)) > 0.5:
            # Then alternative is true (full ranking is POSSIBLE) [2nd condition]
            # AND we correctly ranked everything
            CORRECT_FULL_RANKING_ALT.append(run) # All hypotheses decided correctly
        elif np.isclose(np.linalg.norm(correct_hypothesis_signs - hypothesis_decisions), 0.) and np.min(np.abs(correct_hypothesis_signs)) < 0.5:
            # Then null is true (full ranking is IMPOSSIBLE) [2nd condition]
            # AND we did not make an erroneous rejection!
            CORRECT_PARTIAL_RANKING_NULL.append(run) # At least one hypothesis was undecidable (means were the same)
        elif (np.linalg.norm(correct_hypothesis_signs - hypothesis_decisions) >= 0.1) and np.min(np.abs(correct_hypothesis_signs)) > 0.5:
            # Then alternative is true (full ranking is POSSIBLE) [2nd condition]
            # AND we incorrectly ranked everything
            INCOMPLETE_FULL_RANKING_ALT.append(run) # At least one hypothesis not decided CORRECTLY
            if np.max(np.abs(correct_hypothesis_signs - hypothesis_decisions) > 1.5):
                INCORRECT_FULL_RANKING_ALT.append(run)
            
        elif (np.linalg.norm(correct_hypothesis_signs - hypothesis_decisions) >= 0.1) and np.min(np.abs(correct_hypothesis_signs)) < 0.5:
            # Then null is true (full ranking is IMPOSSIBLE) [2nd condition]
            # AND we still make an error somewhere
            INCOMPLETE_PARTIAL_RANKING_NULL.append(run) # Ranking is at least incomplete

            # Now check if the ranking has an explicit error [True 0, Estimate +- 1] OR [TRUE +- 1, Estimate -+ 1]
            for i in range(num_hypotheses):
                if np.isclose(correct_hypothesis_signs[i], 0.) and np.abs(hypothesis_decisions[i]) > 0.5:
                    INCORRECT_PARTIAL_RANKING_NULL.append(run)
                    break
                elif np.abs(correct_hypothesis_signs[i]) > 0.5 and np.abs(correct_hypothesis_signs[i] - hypothesis_decisions[i]) > 1.5:
                    INCORRECT_PARTIAL_RANKING_NULL.append(run)
                    break
                else:
                    pass
            
        else:
            # Should not be possible to reach this state
            raise ValueError("Outcome of graphical procedure unrecognized")
        
        ALPHA_AT_REJECTION_SORTED[:len(alpha_at_rejected), run] = np.sort(alpha_at_rejected)
        
        _accumulate_samples(samples, run, decision_times)
        animate(graphs_over_time, results_dir, Nmax, n_prior)
        # This has a bug when not everything is rejected, because decision times does not account for the non-rejected hypotheses
        for i, key_null in enumerate(null_hypotheses_policy_indices):
            found_key = False 
            for key, value in decision_times.items():
                if key == key_null:
                    found_key = True
                    avg_ttd[key] = avg_ttd.get(key, 0) + value
                    break 
            
            # Fix bug to keep time-to-decision for unrejected hypotheses
            if not found_key:
                avg_ttd[key] = avg_ttd.get(key, 0) + Nmax
        
        # print("Rejected hypotheses (policy pairs): ", rejected_hypotheses)
        # print("Decision times: ", decision_times, "\n")

        # E-value graphical multitest
        # print("Running e-value graphical multitest...")
        egraphical_test = ESequentialGraphicalTest(
            num_policies=n_policies,
            null_hypotheses=null_hypotheses_policy_indices,
            total_alpha=alpha,
        )
        _, _, decision_times_evalues, _, _, graphs_over_time_evalues = (
            egraphical_test.sequential_graphical_multitest(
                null_hypotheses_policy_indices, policy_data, Nmax,
                alpha_per_hypothesis=alpha_per_hypothesis,
                weighted_G=weighted_G, verbose=False,
            )
        )
        _accumulate_samples(samples_evalues, run, decision_times_evalues)
        animate(graphs_over_time_evalues, results_dir, Nmax, n_prior)
        for key, value in decision_times_evalues.items():
            avg_ttd_evalues[key] = avg_ttd_evalues.get(key, 0) + value
        # print("Decision times: ", decision_times_evalues, "\n")

        # Bonferroni
        # print("Running Bonferroni corrected individual tests for comparison...")
        rejected_hypotheses_bonferroni, decision_times_bonferroni = graphical_test.bonferroni_multitest(
            null_hypotheses_policy_indices, policy_data, Nmax, alpha, bernoulli=bernoulli
        )
        _accumulate_samples(samples_bonferroni, run, decision_times_bonferroni)
        for key, value in decision_times_bonferroni.items():
            avg_ttd_bonferroni[key] = avg_ttd_bonferroni.get(key, 0) + value
        # print("Rejected hypotheses (policy pairs): ", rejected_hypotheses_bonferroni)
        # print("Decision times: ", decision_times_bonferroni, "\n")

        # Weighted Bonferroni
        # print("Running weighted Bonferroni corrected individual tests for comparison...")
        rejected_hypotheses_weighted_bonferroni, decision_times_weighted_bonferroni = (
            graphical_test.weighted_bonferroni_multitest(
                null_hypotheses_policy_indices, policy_data, Nmax,
                alpha_per_hypothesis_weighted_bonferroni, bernoulli=bernoulli
            )
        )
        _accumulate_samples(samples_weighted_bonferroni, run, decision_times_weighted_bonferroni)
        for key, value in decision_times_weighted_bonferroni.items():
            avg_ttd_weighted_bonferroni[key] = avg_ttd_weighted_bonferroni.get(key, 0) + value
        # print("Rejected hypotheses (policy pairs): ", rejected_hypotheses_weighted_bonferroni)
        # print("Decision times: ", decision_times_weighted_bonferroni, "\n")

        # Fixed sequence
        # print("Running Fixed sequence individual tests for comparison...")
        rejected_hypotheses_fixed, decision_times_fixed = graphical_test.fixed_multitest(
            null_hypotheses_policy_indices, policy_data, Nmax, alpha, bernoulli=bernoulli
        )
        _accumulate_samples(samples_fixed, run, decision_times_fixed)
        for key, value in decision_times_fixed.items():
            avg_ttd_fixed[key] = avg_ttd_fixed.get(key, 0) + value
        # print("Rejected hypotheses (policy pairs): ", rejected_hypotheses_fixed)
        # print("Decision times: ", decision_times_fixed, "\n")
        
        print()
        print(f"After iteration {run+1}:")

        proportion_fully_correct_under_alt = float(len(CORRECT_FULL_RANKING_ALT)) / (run+1)
        proportion_fully_correct_under_null = float(len(CORRECT_PARTIAL_RANKING_NULL)) / (run+1)
        proportion_with_actively_erroneous_decisions = float(len(INCORRECT_FULL_RANKING_ALT) + len(INCORRECT_PARTIAL_RANKING_NULL)) / (run+1)
        proportion_that_only_fail_to_fully_rank = float(len(INCOMPLETE_FULL_RANKING_ALT) - len(INCORRECT_FULL_RANKING_ALT)) / (run+1)
        proportion_that_only_fail_to_complete_partial_ranking = float(len(INCOMPLETE_PARTIAL_RANKING_NULL) - len(INCORRECT_PARTIAL_RANKING_NULL)) / (run+1)
        
        print()
        print("Proportion of correct full rankings under alt [POWER]: ")
        print(f"{proportion_fully_correct_under_alt:0.2f}")
        print()
        print("Proportion with undecided cases under alt [TYPE-II ERROR]: ")
        print(f"{proportion_that_only_fail_to_fully_rank:0.2f}")
        print()
        print("Proportion of correct partial rankings under equality condition [POWER]: ")
        print(f"{proportion_fully_correct_under_null:0.2f}")
        print()
        print("Proportion of undecided partial rankings under equality condition [TYPE-II ERROR]: ")
        print(f"{proportion_that_only_fail_to_complete_partial_ranking:0.2f}")
        print()
        print("Proportion of erroneous rankings [TYPE-I ERROR]: ")
        print(f"{proportion_with_actively_erroneous_decisions:0.2f}")

    print()
    print("-----------------------")
    print()
    print("Finished for loop!")
    print()
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
    print("average alpha at rejected:")
    with np.printoptions(precision=3, suppress=True):
        print(np.mean(ALPHA_AT_REJECTION_SORTED, axis=1))
    
    np.save(f'{results_dir}/alpha_at_rejection_sorted_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy', ALPHA_AT_REJECTION_SORTED)
    # np.save(f'{results_dir}/correct_full_ranking_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy', CORRECT_FULL_RANKING)

    proportion_fully_correct_under_alt = float(len(CORRECT_FULL_RANKING_ALT)) / (n_runs)
    proportion_fully_correct_under_null = float(len(CORRECT_PARTIAL_RANKING_NULL)) / (n_runs)
    proportion_with_actively_erroneous_decisions = float(len(INCORRECT_FULL_RANKING_ALT) + len(INCORRECT_PARTIAL_RANKING_NULL)) / (n_runs)
    proportion_that_only_fail_to_fully_rank = float(len(INCOMPLETE_FULL_RANKING_ALT) - len(INCORRECT_FULL_RANKING_ALT)) / (n_runs)
    proportion_that_only_fail_to_complete_partial_ranking = float(len(INCOMPLETE_PARTIAL_RANKING_NULL) - len(INCORRECT_PARTIAL_RANKING_NULL)) / (n_runs)
    
    print()
    print("Proportion of correct full rankings under alt [POWER]: ")
    print(f"{proportion_fully_correct_under_alt:0.2f}")
    print()
    print("Proportion with undecided cases under alt [TYPE-II ERROR]: ")
    print(f"{proportion_that_only_fail_to_fully_rank:0.2f}")
    print()
    print("Proportion of correct partial rankings under equality condition [POWER]: ")
    print(f"{proportion_fully_correct_under_null:0.2f}")
    print()
    print("Proportion of undecided partial rankings under equality condition [TYPE-II ERROR]: ")
    print(f"{proportion_that_only_fail_to_complete_partial_ranking:0.2f}")
    print()
    print("Proportion of erroneous rankings [TYPE-I ERROR]: ")
    print(f"{proportion_with_actively_erroneous_decisions:0.2f}")
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
