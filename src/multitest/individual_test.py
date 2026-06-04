
'''
Apurva Badithela, Mar 16, 2026

To find the upper bound on expected efficiency gains from fixed sequence testing
'''

import json
import numpy as np
import os
import copy
import tqdm
import tempfile
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from PIL import Image as PILImage
from multitest.sequential_graphical import SequentialGraphicalTest
from multitest.sequential_graphical_evalue import SequentialGraphicalTest as ESequentialGraphicalTest
from multitest.sequential_graphical_evalue_active import SequentialGraphicalTest as EActiveSequentialGraphicalTest
from multitest.sequential_graphical_active import SequentialGraphicalTest as ASequentialGraphicalTest
from multitest.plot_cld import make_violin_plots_from_ttds, get_cld_letters_from_ttds
import pandas as pd

#############################################
# Hyperparameters
#############################################
@dataclass
class ExperimentConfig:
    Nmax: int = 5000
    n_runs: int = 1
    n_prior: int = 20
    alpha: float = 0.1
    figsize: tuple = (12, 10)
    beta: float = 1.0                  # tuning parameter for alpha allocation in graphical test
    plot_from_saved: bool = False      # set to True to plot from saved data
    run_new_experiment: bool = True    # set to False to plot from saved data
    results_dir: str = 'outputs/roboarena_subset4'
    plot_violin: bool = True           # set to False to skip violin plots (data is always saved)
    graph_type: str = "soft_masked"    # "soft_masked" or "fully_connected"


def get_partial_ranking(rejected_hypotheses, null_hypotheses_policy_indices, n_policies, hypotheses_correct=None):
    """
    Derive a partial ranking from decided pairwise hypotheses.

    For each decided pair (p0, p1), the direction is determined by hypotheses_correct:
      +1  — null rejected, alternative accepted  → p1 > p0 (p1 wins, p0 loses)
      -1  — null accepted                        → p0 > p1 (p0 wins, p1 loses)
    When hypotheses_correct is None every decided pair is assumed to have direction +1
    (backward-compatible with callers that only track rejections).

    Pairs with no decision are left incomparable — no ordering is claimed.

    The ranking is a linear extension of the confirmed partial order, using net score
    (wins - losses) as the primary key and raw wins as a tiebreaker.

    Parameters
    ----------
    rejected_hypotheses            : list of (p0, p1) pairs for which the test reached
                                     a decision (null rejected or accepted)
    null_hypotheses_policy_indices : full ordered list of all tested (p0, p1) pairs
    n_policies                     : total number of policies
    hypotheses_correct             : optional dict {(p0, p1): +1 or -1} recording the
                                     direction of each decision; if None, all decisions
                                     are treated as null-rejected (+1)

    Returns
    -------
    ranking      : policy indices sorted best-to-worst by net score (wins - losses),
                   tiebroken by raw wins
    wins         : dict {policy_idx: number of confirmed wins}
    losses       : dict {policy_idx: number of confirmed losses}
    incomparable : list of tested (p0, p1) pairs for which no ordering was established
    """
    wins = {i: 0 for i in range(n_policies)}
    losses = {i: 0 for i in range(n_policies)}
    rejected_set = set(map(tuple, rejected_hypotheses))
    for i, (p0, p1) in enumerate(rejected_hypotheses):
        direction = hypotheses_correct[i] if hypotheses_correct is not None else 1
        if direction == 1:
            wins[p1] += 1
            losses[p0] += 1
        else:
            wins[p0] += 1
            losses[p1] += 1
    incomparable = [pair for pair in null_hypotheses_policy_indices if tuple(pair) not in rejected_set]
    ranking = sorted(range(n_policies),
                     key=lambda x: (wins[x] - losses[x], wins[x]),
                     reverse=True)
    return ranking, wins, losses, incomparable


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
    # diffs = diffs - diffs.max()  # shift to make all diffs non-negative, preserving relative magnitudes
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


def plot_heatmap(matrix, title, save_path, real_sim_means, fmt='.2f', figsize=(12, 10)):
    """Render a labelled heatmap and save it to disk."""
    n = matrix.shape[0]
    xlabels = [
        f"R={real:.2f}\nS={sim:.2f}"
        for real, sim in real_sim_means
    ]
    ylabels = xlabels[::-1]
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(matrix, cmap='viridis')
    vmin, vmax = matrix.min(), matrix.max()
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            norm_val = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
            text_color = 'white' if norm_val < 0.5 else 'black'
            ax.text(j, i, f'{val:{fmt}}', ha='center', va='center',
                    color=text_color, fontsize=int(2*min(figsize)))

    fig.colorbar(cax)
    ax.set_xlabel("Policy 0 mean (real, sim)", fontsize=int(2*figsize[0]))
    ax.set_ylabel("Policy 1 mean (real, sim)", fontsize=int(2*figsize[1]))
    ax.set_title(title, fontsize=int(2*figsize[0]))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(xlabels, rotation=0,fontsize=16)
    ax.set_yticklabels(ylabels, fontsize=16)
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()


def _accumulate_samples(samples_dict, run, decision_times, Nmax_policies, procedure="graphical"):
    for hyp, ttd in decision_times.items():
        pi1, pi2 = hyp
        pi1_max = Nmax_policies[pi1]
        pi2_max = Nmax_policies[pi2]
        Nmax = min(pi1_max, pi2_max)
        if procedure == "fixed_sequence" and ttd == "N/A":
            ttd = 0
        else:
            ttd = Nmax if ttd == "N/A" else ttd

        pi1, pi2 = hyp
        
        samples_dict[run][pi1] = min(pi1_max, max(ttd, samples_dict[run].get(pi1, 0)))
        samples_dict[run][pi2] = min(pi2_max, max(ttd, samples_dict[run].get(pi2, 0)))

def _average_dict(d, n):
    for key in d:
        d[key] /= n

def _fill_exchange_matrix(avg_ttd_dict, n_policies):
    matrix = np.zeros((n_policies, n_policies))
    for (p0_index, p1_index), ttd in avg_ttd_dict.items():
        p1_index_ex = n_policies - 1 - p1_index
        matrix[p1_index_ex, p0_index] = ttd
    return matrix

def _to_policy_matrix(policy_data: dict, policies: list, Nmax: int) -> np.ndarray:
    """Convert dict of policy -> evals to a (Nmax, n_policies) matrix, NaN-padded for short sequences."""
    mat = np.full((Nmax, len(policies)), np.nan)
    for i, p in enumerate(policies):
        evals = np.asarray(policy_data[p])
        rows = min(len(evals), Nmax)
        mat[:rows, i] = evals[:rows]
    return mat

def split_eval(eval_results, Nmax, nruns=10, perfect_sim=False):
    prior_evals = {k: np.array(v[:nruns]) for k, v in eval_results.items()}
    additional_evals = {k: np.array(v[nruns:Nmax]) for k, v in eval_results.items()}
    if perfect_sim:
        prior_evals = copy.deepcopy(eval_results)
        additional_evals = copy.deepcopy(eval_results)
    return prior_evals, additional_evals

def get_policy_summary(eval_results):
    Nmax = min(len(evals) for evals in eval_results.values())
    for k, evals in eval_results.items():
        print(f"Policy {k}: mean={np.mean(evals):.4f}, std={np.std(evals):.4f}, n={len(evals)}")
    return Nmax

def run_experiments(
    n_policies, ordered_hypotheses,
    policy_data, Nmax, alpha_per_hypothesis,
    alpha_per_hypothesis_weighted_bonferroni, weighted_G,
    policy_index, real_sim_means, bernoulli=False,
    cfg=None, labels=None, methods=None, allow_transitive=False,
):
    if cfg is None:
        cfg = ExperimentConfig()
    n_runs = cfg.n_runs
    results_dir = cfg.results_dir
    alpha = cfg.alpha
    beta = cfg.beta
    n_prior = cfg.n_prior
    plot_from_saved = cfg.plot_from_saved
    figsize = cfg.figsize
    ALL_METHODS = ('graphical', 'evalues', 'evalues_active', 'graphical_active', 'bonferroni', 'weighted_bonferroni', 'fixed')
    if methods is None:
        methods = set(ALL_METHODS)
    else:
        methods = set(methods)
    Nmax_policies = {
        i: np.sum(~np.isnan(policy_data[:, i]))         for i,policy in policy_index.items()
    }
    max_sample_size_per_model = {
        policy: np.sum(~np.isnan(policy_data[:, i]))         for i,policy in policy_index.items()
    }
    #####################################################
    # Looping over multiple runs to generate policy data
    #####################################################
    avg_ttd = {}
    avg_ttd_evalues = {}
    avg_ttd_evalues_active = {}
    avg_ttd_active = {}
    avg_ttd_bonferroni = {}
    avg_ttd_weighted_bonferroni = {}
    avg_ttd_fixed = {}
    samples = {k: {} for k in range(n_runs)}
    samples_evalues = {k: {} for k in range(n_runs)}
    samples_evalues_active = {k: {} for k in range(n_runs)}
    samples_active = {k: {} for k in range(n_runs)}
    samples_fixed = {k: {} for k in range(n_runs)}
    samples_bonferroni = {k: {} for k in range(n_runs)}
    samples_weighted_bonferroni = {k: {} for k in range(n_runs)}
    n_rejected_per_run = {
        'graphical': [],
        'evalues': [],
        'evalues_active': [],
        'graphical_active': [],
        'bonferroni': [],
        'weighted_bonferroni': [],
        'fixed': [],
    }
    decision_times_per_run = {
        'graphical': [],
        'evalues': [],
        'evalues_active': [],
        'graphical_active': [],
        'bonferroni': [],
        'weighted_bonferroni': [],
        'fixed': [],
    }
    rejection_order_per_run = {
        'graphical': [],
        'evalues': [],
        'evalues_active': [],
        'graphical_active': [],
        'bonferroni': [],
        'weighted_bonferroni': [],
        'fixed': [],
    }
    hypotheses_correct_per_run = {
        'graphical': [],
        'evalues': [],
        'evalues_active': [],
        'graphical_active': [],
        'bonferroni': [],
        'weighted_bonferroni': [],
        'fixed': [],
    }
    null_hypotheses = [hyp for hyp, _, _ in ordered_hypotheses]
    null_hypotheses_policy_indices = [idx for _, idx, _ in ordered_hypotheses]
    
    for i, policy in policy_index.items():
        print(f"Policy {policy} mean: ", np.nanmean(policy_data[:,i]))
    print("------------------------")
    for run in range(n_runs):
        ########################  P-value Graphical multitest ########################
        if 'graphical' in methods:
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
                    allow_transitive=allow_transitive,
                )
            )
            

            _accumulate_samples(samples, run, decision_times, Nmax_policies, procedure="p-graphical")
            animate(graphs_over_time, results_dir, Nmax, n_prior)
            for key, value in decision_times.items():
                Nmax_hyp = min(Nmax_policies[key[0]], Nmax_policies[key[1]])
                avg_ttd[key] = avg_ttd.get(key, 0) + (Nmax_hyp if value == "N/A" else value)
            print("Rejected hypotheses (policy pairs): ", rejected_hypotheses)
            print("Decision times: ", decision_times, "\n")
            decision_times_per_run['graphical'].append(
                {f"{p0},{p1}": v for (p0, p1), v in decision_times.items()}
            )
            n_rejected_per_run['graphical'].append(len(rejected_hypotheses))
            rejection_order_per_run['graphical'].append(
                [[policy_index[p0], policy_index[p1]] for p0, p1 in rejected_hypotheses]
            )
            hypotheses_correct_per_run['graphical'].append({f"{p0},{p1}": v for (p0, p1), v in hypotheses_correct.items()})

        ######################## E-Value Graphical multitest ########################
        if 'evalues' in methods:
            print("Running e-value graphical multitest...")
            egraphical_test = ESequentialGraphicalTest(
                num_policies=n_policies,
                null_hypotheses=null_hypotheses_policy_indices,
                total_alpha=alpha,
            )
            rejected_hypotheses_evalues, _, decision_times_evalues, _, _, graphs_over_time_evalues, hypotheses_correct_evalues = (
                egraphical_test.sequential_graphical_multitest(
                    null_hypotheses_policy_indices, policy_data, Nmax,
                    alpha_per_hypothesis=alpha_per_hypothesis,
                    weighted_G=weighted_G, verbose=True,
                )
            )
            _accumulate_samples(samples_evalues, run, decision_times_evalues, Nmax_policies, procedure="evalues")
            # animate(graphs_over_time_evalues, results_dir, Nmax, n_prior)
            for key, value in decision_times_evalues.items():
                Nmax_hyp = min(Nmax_policies[key[0]], Nmax_policies[key[1]])
                avg_ttd_evalues[key] = avg_ttd_evalues.get(key, 0) + (Nmax_hyp   if value == "N/A" else value)
            print("Rejected hypotheses (policy pairs): ", rejected_hypotheses_evalues)
            print("Decision times: ", decision_times_evalues, "\n")
            decision_times_per_run['evalues'].append(
                {f"{p0},{p1}": v for (p0, p1), v in decision_times_evalues.items()}
            )
            n_rejected_per_run['evalues'].append(len(rejected_hypotheses_evalues))
            rejection_order_per_run['evalues'].append(
                [[policy_index[p0], policy_index[p1]] for p0, p1 in rejected_hypotheses_evalues]
            )
            hypotheses_correct_per_run['evalues'].append({f"{p0},{p1}": v for (p0, p1), v in hypotheses_correct_evalues.items()})

        ######################## E-Value Active Selection Graphical multitest ########################
        if 'evalues_active' in methods:
            print("Running e-value active graphical multitest...")
            egraphical_active_test = EActiveSequentialGraphicalTest(
                num_policies=n_policies,
                null_hypotheses=null_hypotheses_policy_indices,
                total_alpha=alpha,
            )
            rejected_hypotheses_evalues_active, _, decision_times_evalues_active, _, _, graphs_over_time_evalues_active, _, hypotheses_correct_evalues_active, _ = (
                egraphical_active_test.sequential_graphical_multitest(
                    null_hypotheses_policy_indices, policy_data, Nmax,
                    alpha_per_hypothesis=alpha_per_hypothesis,
                    weighted_G=weighted_G, verbose=True, bernoulli=bernoulli,
                )
            )
            _accumulate_samples(samples_evalues_active, run, decision_times_evalues_active, Nmax_policies, procedure="evalues_active")
            # animate(graphs_over_time_evalues_active, results_dir, Nmax, n_prior)
            for key, value in decision_times_evalues_active.items():
                Nmax_hyp = min(Nmax_policies[key[0]], Nmax_policies[key[1]])
                avg_ttd_evalues_active[key] = avg_ttd_evalues_active.get(key, 0) + (Nmax_hyp if value == "N/A" else value)
            print("Rejected hypotheses (policy pairs): ", rejected_hypotheses_evalues_active)
            print("Decision times: ", decision_times_evalues_active, "\n")
            decision_times_per_run['evalues_active'].append(
                {f"{p0},{p1}": v for (p0, p1), v in decision_times_evalues_active.items()}
            )
            n_rejected_per_run['evalues_active'].append(len(rejected_hypotheses_evalues_active))
            rejection_order_per_run['evalues_active'].append(
                [[policy_index[p0], policy_index[p1]] for p0, p1 in rejected_hypotheses_evalues_active]
            )
            hypotheses_correct_per_run['evalues_active'].append({f"{p0},{p1}": v for (p0, p1), v in hypotheses_correct_evalues_active.items()})

        ######################## P-Value Active Selection Graphical multitest ########################
        if 'graphical_active' in methods:
            print("Running active graphical multitest (p-value)...")
            agraphical_test = ASequentialGraphicalTest(
                num_policies=n_policies,
                null_hypotheses=null_hypotheses_policy_indices,
                total_alpha=alpha,
            )
            rejected_hypotheses_active, _, decision_times_active, _, _, graphs_over_time_active, _, hypotheses_correct_active, _ = (
                agraphical_test.sequential_graphical_multitest(
                    null_hypotheses_policy_indices, policy_data, Nmax,
                    alpha_per_hypothesis=alpha_per_hypothesis,
                    weighted_G=weighted_G, verbose=True, bernoulli=bernoulli,allow_transitive=allow_transitive,
                )
            )
            _accumulate_samples(samples_active, run, decision_times_active, Nmax_policies, procedure="graphical_active")
            animate(graphs_over_time_active, results_dir, Nmax, n_prior)
            for key, value in decision_times_active.items():
                Nmax_hyp = min(Nmax_policies[key[0]], Nmax_policies[key[1]])
                avg_ttd_active[key] = avg_ttd_active.get(key, 0) + (Nmax_hyp   if value == "N/A" else value)
            print("Rejected hypotheses (policy pairs): ", rejected_hypotheses_active)
            print("Decision times: ", decision_times_active, "\n")
            decision_times_per_run['graphical_active'].append(
                {f"{p0},{p1}": v for (p0, p1), v in decision_times_active.items()}
            )
            n_rejected_per_run['graphical_active'].append(len(rejected_hypotheses_active))
            rejection_order_per_run['graphical_active'].append(
                [[policy_index[p0], policy_index[p1]] for p0, p1 in rejected_hypotheses_active]
            )
            hypotheses_correct_per_run['graphical_active'].append({f"{p0},{p1}": v for (p0, p1), v in hypotheses_correct_active.items()})

        ######################## Bonferroni ########################
        if 'bonferroni' in methods:
            print("Running Bonferroni corrected individual tests for comparison...")
            _graphical_test_for_bonferroni = graphical_test if 'graphical' in methods else SequentialGraphicalTest(
                num_policies=n_policies,
                null_hypotheses=null_hypotheses_policy_indices,
                total_alpha=alpha,
            )
            rejected_hypotheses_bonferroni, decision_times_bonferroni, hypotheses_correct_bonferroni = _graphical_test_for_bonferroni.bonferroni_multitest(
                null_hypotheses_policy_indices, policy_data, Nmax, alpha, bernoulli=bernoulli,
                allow_transitive=allow_transitive,
            )
            _accumulate_samples(samples_bonferroni, run, decision_times_bonferroni, Nmax_policies, procedure="bonferroni")
            for key, value in decision_times_bonferroni.items():
                Nmax_hyp = min(Nmax_policies[key[0]], Nmax_policies[key[1]])
                avg_ttd_bonferroni[key] = avg_ttd_bonferroni.get(key, 0) + (Nmax_hyp   if value == "N/A" else value)
            print("Rejected hypotheses (policy pairs): ", rejected_hypotheses_bonferroni)
            print("Decision times: ", decision_times_bonferroni, "\n")
            decision_times_per_run['bonferroni'].append(
                {f"{p0},{p1}": v for (p0, p1), v in decision_times_bonferroni.items()}
            )
            n_rejected_per_run['bonferroni'].append(len(rejected_hypotheses_bonferroni))
            rejection_order_per_run['bonferroni'].append(
                [[policy_index[p0], policy_index[p1]] for p0, p1 in rejected_hypotheses_bonferroni]
            )
            hypotheses_correct_per_run['bonferroni'].append({f"{p0},{p1}": v for (p0, p1), v in hypotheses_correct_bonferroni.items()})

        ######################## Weighted Bonferroni ########################
        if 'weighted_bonferroni' in methods:
            print("Running weighted Bonferroni corrected individual tests for comparison...")
            _graphical_test_for_wb = graphical_test if 'graphical' in methods else SequentialGraphicalTest(
                num_policies=n_policies,
                null_hypotheses=null_hypotheses_policy_indices,
                total_alpha=alpha,
            )
            rejected_hypotheses_weighted_bonferroni, decision_times_weighted_bonferroni, hypotheses_correct_weighted_bonferroni = (
                _graphical_test_for_wb.weighted_bonferroni_multitest(
                    null_hypotheses_policy_indices, policy_data, Nmax,
                    alpha_per_hypothesis_weighted_bonferroni, bernoulli=bernoulli,
                    allow_transitive=allow_transitive,
                )
            )
            _accumulate_samples(samples_weighted_bonferroni, run, decision_times_weighted_bonferroni, Nmax_policies, procedure="weighted_bonferroni")
            for key, value in decision_times_weighted_bonferroni.items():
                Nmax_hyp = min(Nmax_policies[key[0]], Nmax_policies[key[1]])
                avg_ttd_weighted_bonferroni[key] = avg_ttd_weighted_bonferroni.get(key, 0) + (Nmax_hyp   if value == "N/A" else value)
            print("Rejected hypotheses (policy pairs): ", rejected_hypotheses_weighted_bonferroni)
            print("Decision times: ", decision_times_weighted_bonferroni, "\n")
            decision_times_per_run['weighted_bonferroni'].append(
                {f"{p0},{p1}": v for (p0, p1), v in decision_times_weighted_bonferroni.items()}
            )
            n_rejected_per_run['weighted_bonferroni'].append(len(rejected_hypotheses_weighted_bonferroni))
            rejection_order_per_run['weighted_bonferroni'].append(
                [[policy_index[p0], policy_index[p1]] for p0, p1 in rejected_hypotheses_weighted_bonferroni]
            )
            hypotheses_correct_per_run['weighted_bonferroni'].append({f"{p0},{p1}": v for (p0, p1), v in hypotheses_correct_weighted_bonferroni.items()})

        ######################## Fixed Sequence ########################
        if 'fixed' in methods:
            print("Running Fixed sequence individual tests for comparison...")
            _graphical_test_for_fixed = graphical_test if 'graphical' in methods else SequentialGraphicalTest(
                num_policies=n_policies,
                null_hypotheses=null_hypotheses_policy_indices,
                total_alpha=alpha,
            )
            rejected_hypotheses_fixed, decision_times_fixed, hypotheses_correct_fixed = _graphical_test_for_fixed.fixed_multitest(
                null_hypotheses_policy_indices, policy_data, Nmax, alpha, bernoulli=bernoulli,
                allow_transitive=allow_transitive,
            )
            _accumulate_samples(samples_fixed, run, decision_times_fixed, Nmax_policies, procedure="fixed_sequence")
            
            for key, value in decision_times_fixed.items():
                Nmax_hyp = min(Nmax_policies[key[0]], Nmax_policies[key[1]])
                avg_ttd_fixed[key] = avg_ttd_fixed.get(key, 0) + (Nmax_hyp if value == "N/A" else value)
            print("Rejected hypotheses (policy pairs): ", rejected_hypotheses_fixed)
            print("Decision times: ", decision_times_fixed, "\n")
            decision_times_per_run['fixed'].append(
                {f"{p0},{p1}": v for (p0, p1), v in decision_times_fixed.items()}
            )
            n_rejected_per_run['fixed'].append(len(rejected_hypotheses_fixed))
            rejection_order_per_run['fixed'].append(
                [[policy_index[p0], policy_index[p1]] for p0, p1 in rejected_hypotheses_fixed]
            )
            hypotheses_correct_per_run['fixed'].append({f"{p0},{p1}": v for (p0, p1), v in hypotheses_correct_fixed.items()})

    os.makedirs(f'{results_dir}/rejection_order', exist_ok=True)
    rejection_order_path = f'{results_dir}/rejection_order/rejection_order_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.json'
    with open(rejection_order_path, 'w') as f:
        json.dump(rejection_order_per_run, f, indent=2)
    print(f"Rejection order saved to {rejection_order_path}")

    decision_times_path = f'{results_dir}/rejection_order/decision_times_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.json'
    with open(decision_times_path, 'w') as f:
        json.dump({'decision_times': decision_times_per_run, 'hypotheses_correct': hypotheses_correct_per_run}, f, indent=2)
    print(f"Decision times and hypotheses correct saved to {decision_times_path}")

    with open(f'{results_dir}/empirical_real_means_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.txt', 'w') as f:
        f.write("Real-sim pairs: \n")
        for real_mean, sim_mean in real_sim_means:
            f.write(f"Real mean: {real_mean:.4f}, Sim mean: {sim_mean:.4f}\n")
        f.write("==============================\n")
        f.write("Empirical real means: " + str(np.nanmean(policy_data, axis=0)) + "\n")
        f.write("True real means: " + str([rs[0] for rs in real_sim_means]) + "\n")

    policy_meta = {
        str(i): {
            'name': policy_index[i],
            'real_mean': real_sim_means[i][0],
            'sim_mean': real_sim_means[i][1],
        }
        for i in range(n_policies)
    }
    with open(f'{results_dir}/policy_meta.json', 'w') as f:
        json.dump(policy_meta, f, indent=2)


    _average_dict(avg_ttd, n_runs)
    _average_dict(avg_ttd_evalues, n_runs)
    _average_dict(avg_ttd_evalues_active, n_runs)
    _average_dict(avg_ttd_active, n_runs)
    _average_dict(avg_ttd_bonferroni, n_runs)
    _average_dict(avg_ttd_fixed, n_runs)
    _average_dict(avg_ttd_weighted_bonferroni, n_runs)

    # Compute and save CLD-based policy rankings from averaged TTDs.
    policy_names = [policy_index[i] for i in range(n_policies)]
    progress_array_dict_cld = {
        policy_index[i]: policy_data[:, i][~np.isnan(policy_data[:, i])]
        for i in range(n_policies)
    }
    ttd_dicts_for_ranking = {
        "graphical":           {(policy_index[i], policy_index[j]): ttd for (i, j), ttd in avg_ttd.items()},
        "evalues":             {(policy_index[i], policy_index[j]): ttd for (i, j), ttd in avg_ttd_evalues.items()},
        "evalues_active":      {(policy_index[i], policy_index[j]): ttd for (i, j), ttd in avg_ttd_evalues_active.items()},
        "graphical_active":    {(policy_index[i], policy_index[j]): ttd for (i, j), ttd in avg_ttd_active.items()},
        "bonferroni":          {(policy_index[i], policy_index[j]): ttd for (i, j), ttd in avg_ttd_bonferroni.items()},
        "weighted_bonferroni": {(policy_index[i], policy_index[j]): ttd for (i, j), ttd in avg_ttd_weighted_bonferroni.items()},
        "fixed":               {(policy_index[i], policy_index[j]): ttd for (i, j), ttd in avg_ttd_fixed.items()},
    }
    cld_results_for_ranking = get_cld_letters_from_ttds(
        progress_array_dict_cld, ttd_dicts_for_ranking, max_sample_size_per_model, policy_names
    )

    ranking_path = f'{results_dir}/policy_rankings_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.txt'
    print(f"\n{'='*60}\nPolicy Rankings (CLD)\n{'='*60}")
    with open(ranking_path, 'w') as f:
        f.write(f"Policy Rankings (CLD) — N={Nmax}, n={n_prior}, alpha={alpha}, beta={beta}\n")
        f.write(f"Policies: {policy_names}\n")
        for method, (cld_dict, max_ttd) in cld_results_for_ranking.items():
            active = [(alg, cld_dict[alg]) for alg in policy_names if max_ttd.get(alg, 0) > 0]
            inactive = [alg for alg in policy_names if max_ttd.get(alg, 0) == 0]
            ranked = sorted(active, key=lambda x: (min(x[1]), len(x[1])))
            f.write(f"\n{'='*60}\n{method}\n{'='*60}\n")
            print(f"\n{method}:")
            for alg, letters in ranked:
                line = f"  {alg}: {letters}"
                f.write(line + "\n")
                print(line)
            if inactive:
                note = f"  Not evaluated: {inactive}"
                f.write(note + "\n")
                print(note)
    print(f"\nPolicy rankings written to {ranking_path}")

    os.makedirs(f'{results_dir}/sample_complexity', exist_ok=True)
    with open(f'{results_dir}/sample_complexity/sample_complexity_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.txt', 'w') as f:
        for i in range(n_policies):
            g = np.mean([samples[run].get(i, 0) for run in range(n_runs)])
            eg = np.mean([samples_evalues[run].get(i, 0) for run in range(n_runs)])
            eg_active = np.mean([samples_evalues_active[run].get(i, 0) for run in range(n_runs)])
            ag = np.mean([samples_active[run].get(i, 0) for run in range(n_runs)])
            b = np.mean([samples_bonferroni[run].get(i, 0) for run in range(n_runs)])
            fx = np.mean([samples_fixed[run].get(i, 0) for run in range(n_runs)])
            wb = np.mean([samples_weighted_bonferroni[run].get(i, 0) for run in range(n_runs)])
            f.write(f"Policy {i}: Graphical={g:.2f}, EGraphical={eg:.2f}, EGraphicalActive={eg_active:.2f}, GraphicalActive={ag:.2f}, Bonferroni={b:.2f}, Fixed={fx:.2f}, Weighted_Bonferroni={wb:.2f}\n")

    n_hypotheses = len(null_hypotheses_policy_indices)
    os.makedirs(f'{results_dir}/rejection_counts', exist_ok=True)
    with open(f'{results_dir}/rejection_counts/rejection_counts_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.txt', 'w') as f:
        f.write(f"Total hypotheses: {n_hypotheses}\n")
        f.write("==============================\n")
        for method, counts in n_rejected_per_run.items():
            mean_rejected = np.mean(counts)
            f.write(f"{method}: mean={mean_rejected:.2f} / {n_hypotheses}, per_run={counts}\n")

    #############################################
    # Summarizing results
    #############################################
    if 'graphical' in methods:
        exchange_pvalue_matrix = np.zeros((n_policies, n_policies))
        exchange_alpha_matrix = np.zeros((n_policies, n_policies))
        for i, (p0_index, p1_index) in enumerate(null_hypotheses_policy_indices):
            p1_index_ex = n_policies - 1 - p1_index
            exchange_pvalue_matrix[p1_index_ex, p0_index] = p_values[i]
            exchange_alpha_matrix[p1_index_ex, p0_index] = alpha_per_hypothesis[i]

        os.makedirs(f'{results_dir}/exchange_matrices', exist_ok=True)
        plot_heatmap(
            exchange_pvalue_matrix,
            title='Graphical Multitest: p-values for each hypothesis',
            save_path=f'{results_dir}/exchange_matrices/graphical_multitest_pvalues_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.png',
            real_sim_means=real_sim_means, fmt='.2e', figsize=figsize,
        )
        plot_heatmap(
            exchange_alpha_matrix,
            title='Graphical Multitest: alpha allocated for each hypothesis',
            save_path=f'{results_dir}/exchange_matrices/graphical_multitest_alpha_allocation_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.png',
            real_sim_means=real_sim_means, fmt='.2e', figsize=figsize,
        )

    np.save(f'{results_dir}/sample_complexity/graphical_multitest_samples_per_policy_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy', samples)
    np.save(f'{results_dir}/sample_complexity/bonferroni_multitest_samples_per_policy_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy', samples_bonferroni)

    exchange_matrix_graphical = _fill_exchange_matrix(avg_ttd, n_policies)
    exchange_matrix_evalues = _fill_exchange_matrix(avg_ttd_evalues, n_policies)
    exchange_matrix_evalues_active = _fill_exchange_matrix(avg_ttd_evalues_active, n_policies)
    exchange_matrix_active = _fill_exchange_matrix(avg_ttd_active, n_policies)
    exchange_matrix_bonferroni = _fill_exchange_matrix(avg_ttd_bonferroni, n_policies)
    exchange_matrix_fixed = _fill_exchange_matrix(avg_ttd_fixed, n_policies)
    exchange_matrix_weighted_bonferroni = _fill_exchange_matrix(avg_ttd_weighted_bonferroni, n_policies)

    true_sample_complexity_fs = 0.
    true_sample_complexity_evalues = 0.
    true_sample_complexity_evalues_active = 0.
    true_sample_complexity_active = 0.
    true_sample_complexity_bonferroni = 0.
    true_sample_complexity_weighted_bonferroni = 0.
    true_sample_complexity_fixed = 0.

    for i in range(n_policies):
        true_sample_complexity_fs += np.maximum(np.max(exchange_matrix_graphical[:, i]), np.max(exchange_matrix_graphical[n_policies-i-1, :]))
        true_sample_complexity_evalues += np.maximum(np.max(exchange_matrix_evalues[:, i]), np.max(exchange_matrix_evalues[n_policies-i-1, :]))
        true_sample_complexity_evalues_active += np.maximum(np.max(exchange_matrix_evalues_active[:, i]), np.max(exchange_matrix_evalues_active[n_policies-i-1, :]))
        true_sample_complexity_active += np.maximum(np.max(exchange_matrix_active[:, i]), np.max(exchange_matrix_active[n_policies-i-1, :]))
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
        exchange_matrix_saved[p1_index_ex, p0_index] = trials_saved
        

    os.makedirs(f'{results_dir}/exchange_matrices', exist_ok=True)
    np.save(f'{results_dir}/exchange_matrices/graphical_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy', exchange_matrix_graphical)
    np.save(f'{results_dir}/exchange_matrices/bonferroni_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy', exchange_matrix_bonferroni)
    np.save(f'{results_dir}/exchange_matrices/weighted_bonferroni_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy', exchange_matrix_weighted_bonferroni)
    np.save(f'{results_dir}/exchange_matrices/graphical_multitest_exchange_matrix_saved_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy', exchange_matrix_saved)
    np.save(f'{results_dir}/exchange_matrices/fixed_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy', exchange_matrix_fixed)
    np.save(f'{results_dir}/exchange_matrices/evalues_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy', exchange_matrix_evalues)
    np.save(f'{results_dir}/exchange_matrices/evalues_active_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy', exchange_matrix_evalues_active)
    np.save(f'{results_dir}/exchange_matrices/active_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy', exchange_matrix_active)

    if plot_from_saved:
        exchange_matrix_saved = np.load(f'{results_dir}/exchange_matrices/graphical_multitest_exchange_matrix_saved_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy')
        exchange_matrix_graphical = np.load(f'{results_dir}/exchange_matrices/graphical_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy')
        exchange_matrix_bonferroni = np.load(f'{results_dir}/exchange_matrices/bonferroni_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy')
        exchange_matrix_weighted_bonferroni = np.load(f'{results_dir}/exchange_matrices/weighted_bonferroni_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy')
        exchange_matrix_fixed = np.load(f'{results_dir}/exchange_matrices/fixed_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy')
        exchange_matrix_evalues = np.load(f'{results_dir}/exchange_matrices/evalues_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy')
        exchange_matrix_evalues_active = np.load(f'{results_dir}/exchange_matrices/evalues_active_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy')
        exchange_matrix_active = np.load(f'{results_dir}/exchange_matrices/active_multitest_exchange_matrix_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.npy')
        total_trials_saved_fs = np.sum(exchange_matrix_saved)

    print("Exchange matrix for graphical multitest (time to decision): ")
    print(exchange_matrix_graphical)
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
    n_hypotheses = len(null_hypotheses_policy_indices)
    mean_rejected = {k: np.mean(v) for k, v in n_rejected_per_run.items()}
    print("Actual Sample complexities: ")
    print(f"Complexity of Our Approach:                   {true_sample_complexity_fs}  (rejected {mean_rejected['graphical']:.1f}/{n_hypotheses})")
    print(f"Complexity of E-values Approach:              {true_sample_complexity_evalues}  (rejected {mean_rejected['evalues']:.1f}/{n_hypotheses})")
    print(f"Complexity of E-values Active Approach:       {true_sample_complexity_evalues_active}  (rejected {mean_rejected['evalues_active']:.1f}/{n_hypotheses})")
    print(f"Complexity of Active Graphical Approach:      {true_sample_complexity_active}  (rejected {mean_rejected['graphical_active']:.1f}/{n_hypotheses})")
    print(f"Complexity of Bonferroni Approach:            {true_sample_complexity_bonferroni}  (rejected {mean_rejected['bonferroni']:.1f}/{n_hypotheses})")
    print(f"Complexity of Weighted Bonferroni Approach:   {true_sample_complexity_weighted_bonferroni}  (rejected {mean_rejected['weighted_bonferroni']:.1f}/{n_hypotheses})")
    print(f"Complexity of Fixed Sequence Approach:        {true_sample_complexity_fixed}  (rejected {mean_rejected['fixed']:.1f}/{n_hypotheses})")

    sc_path = f'{results_dir}/sample_complexity/actual_sample_complexity_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.txt'
    with open(sc_path, 'w') as f:
        f.write(f"Actual Sample Complexities — N={Nmax}, n={n_prior}, alpha={alpha}, beta={beta}\n")
        f.write(f"Total hypotheses: {n_hypotheses}\n")
        f.write("==============================\n")
        f.write(f"Our Approach (Graphical):              {true_sample_complexity_fs}  (rejected {mean_rejected['graphical']:.1f}/{n_hypotheses})\n")
        f.write(f"E-values (passive):                    {true_sample_complexity_evalues}  (rejected {mean_rejected['evalues']:.1f}/{n_hypotheses})\n")
        f.write(f"E-values (active):                     {true_sample_complexity_evalues_active}  (rejected {mean_rejected['evalues_active']:.1f}/{n_hypotheses})\n")
        f.write(f"Active Graphical (p-value):             {true_sample_complexity_active}  (rejected {mean_rejected['graphical_active']:.1f}/{n_hypotheses})\n")
        f.write(f"Bonferroni:                            {true_sample_complexity_bonferroni}  (rejected {mean_rejected['bonferroni']:.1f}/{n_hypotheses})\n")
        f.write(f"Weighted Bonferroni:                   {true_sample_complexity_weighted_bonferroni}  (rejected {mean_rejected['weighted_bonferroni']:.1f}/{n_hypotheses})\n")
        f.write(f"Fixed Sequence:                        {true_sample_complexity_fixed}  (rejected {mean_rejected['fixed']:.1f}/{n_hypotheses})\n")
    print(f"Actual sample complexities written to {sc_path}")
    print()
    if 'graphical' in methods:
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
        save_path=f'{results_dir}/exchange_matrices/graphical_multitest_saved_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.png',
        real_sim_means=real_sim_means, figsize=figsize,
    )
    plot_heatmap(
        exchange_matrix_graphical,
        title='Graphical Multitest: Time to Decision',
        save_path=f'{results_dir}/exchange_matrices/graphical_multitest_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.png',
        real_sim_means=real_sim_means, figsize=figsize,
    )
    plot_heatmap(
        exchange_matrix_bonferroni,
        title='Bonferroni: Time to Decision',
        save_path=f'{results_dir}/exchange_matrices/bonferroni_multitest_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.png',
        real_sim_means=real_sim_means, figsize=figsize,
    )
    plot_heatmap(
        exchange_matrix_weighted_bonferroni,
        title='Weighted Bonferroni: Time to Decision',
        save_path=f'{results_dir}/exchange_matrices/weighted_bonferroni_multitest_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.png',
        real_sim_means=real_sim_means, figsize=figsize,
    )
    plot_heatmap(
        exchange_matrix_fixed,
        title='Fixed: Time to Decision',
        save_path=f'{results_dir}/exchange_matrices/fixed_multitest_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.png',
        real_sim_means=real_sim_means, figsize=figsize,
    )
    plot_heatmap(
        exchange_matrix_evalues,
        title='E-Values Graphical: Time to Decision',
        save_path=f'{results_dir}/exchange_matrices/evalues_multitest_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.png',
        real_sim_means=real_sim_means, figsize=figsize,
    )
    plot_heatmap(
        exchange_matrix_evalues_active,
        title='E-Values Active Graphical: Time to Decision',
        save_path=f'{results_dir}/exchange_matrices/evalues_active_multitest_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.png',
        real_sim_means=real_sim_means, figsize=figsize,
    )
    plot_heatmap(
        exchange_matrix_active,
        title='Active Graphical (p-value): Time to Decision',
        save_path=f'{results_dir}/exchange_matrices/active_multitest_ttd_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}.png',
        real_sim_means=real_sim_means, figsize=figsize,
    )

    # CLD violin plots — one per testing method
    progress_array_dict_cld = {
        policy_index[i]: policy_data[:, i][~np.isnan(policy_data[:, i])]
        for i in range(n_policies)
    }
    ttd_dicts_cld = {
        "graphical":           {(policy_index[i], policy_index[j]): ttd for (i, j), ttd in avg_ttd.items()},
        "evalues":             {(policy_index[i], policy_index[j]): ttd for (i, j), ttd in avg_ttd_evalues.items()},
        "evalues_active":      {(policy_index[i], policy_index[j]): ttd for (i, j), ttd in avg_ttd_evalues_active.items()},
        "graphical_active":    {(policy_index[i], policy_index[j]): ttd for (i, j), ttd in avg_ttd_active.items()},
        "bonferroni":          {(policy_index[i], policy_index[j]): ttd for (i, j), ttd in avg_ttd_bonferroni.items()},
        "weighted_bonferroni": {(policy_index[i], policy_index[j]): ttd for (i, j), ttd in avg_ttd_weighted_bonferroni.items()},
        "fixed":               {(policy_index[i], policy_index[j]): ttd for (i, j), ttd in avg_ttd_fixed.items()},
    }
    policy_names = [policy_index[i] for i in range(n_policies)]

    # Always persist the violin inputs so they can be re-plotted later.
    violin_dir = os.path.join(results_dir, "violin_data")
    os.makedirs(violin_dir, exist_ok=True)
    violin_base = os.path.join(violin_dir, f"violin_N{Nmax}_n{n_prior}_alpha{alpha}_beta{beta}")
    np.savez(
        violin_base + "_progress.npz",
        **{name: arr for name, arr in progress_array_dict_cld.items()},
    )
    violin_meta = {
        "policies": policy_names,
        "Nmax": Nmax,
        "ttd_dicts": {
            method: [[pi, pj, ttd] for (pi, pj), ttd in ttd_dict.items()]
            for method, ttd_dict in ttd_dicts_cld.items()
        },
    }
    with open(violin_base + "_meta.json", "w") as f:
        json.dump(violin_meta, f, indent=2)

    plot_labels = labels if labels is not None else policy_names
    if plot_from_saved or cfg.plot_violin:
        if bernoulli:
            mode="success_rate"
        else:
            mode="task_progress"

        make_violin_plots_from_ttds(
            progress_array_dict=progress_array_dict_cld,
            ttd_dicts=ttd_dicts_cld,
            policies=policy_names,
            labels=plot_labels,
            max_sample_size_per_model=max_sample_size_per_model,
            mode=mode,
            output_dir=os.path.join(results_dir, "cld_violins", f"beta{beta}"),
        )


def soft_masked_graph(ordered_null_hypotheses, cfg, forward_fraction=0.9):
    """
    Build a weight matrix that sends most alpha forward (to harder-to-detect
    hypotheses later in the ordering) and a small fraction backward.

    forward_fraction controls the total share of weight assigned to forward
    edges when both forward and backward neighbors exist.  Within each group
    weights are proportional to allocate_weights (larger mean gaps get more).
    Edge hypotheses (no forward or no backward neighbors) receive all weight
    in the available direction.
    """
    num_hypotheses = len(ordered_null_hypotheses)
    weighted_G = np.zeros((num_hypotheses, num_hypotheses))

    for k1 in range(num_hypotheses):
        forward_neighbors = [(k2, ordered_null_hypotheses[k2]) for k2 in range(k1 + 1, num_hypotheses)]
        backward_neighbors = [(k2, ordered_null_hypotheses[k2]) for k2 in range(0, k1)]

        fwd_diffs = [abs(mu1 - mu0) for _, (mu0, mu1) in forward_neighbors]
        bwd_diffs = [abs(mu1 - mu0) for _, (mu0, mu1) in backward_neighbors]

        if forward_neighbors and backward_neighbors:
            fwd_weights = np.asarray(allocate_weights(fwd_diffs, beta=cfg.beta)) * forward_fraction
            bwd_weights = np.asarray(allocate_weights(bwd_diffs, beta=cfg.beta)) * (1.0 - forward_fraction)
        elif forward_neighbors:
            fwd_weights = np.asarray(allocate_weights(fwd_diffs, beta=cfg.beta))
            bwd_weights = np.array([])
        else:
            fwd_weights = np.array([])
            bwd_weights = np.asarray(allocate_weights(bwd_diffs, beta=cfg.beta))

        for idx, (k2, _) in enumerate(forward_neighbors):
            weighted_G[k1, k2] = fwd_weights[idx]
        for idx, (k2, _) in enumerate(backward_neighbors):
            weighted_G[k1, k2] = bwd_weights[idx]

    return weighted_G

def fully_connected_graph(ordered_null_hypotheses, cfg):
    num_hypotheses = len(ordered_null_hypotheses)
    successor_neighbors = {
        (hyp_idx, hyp): [
            (other_hyp_idx, other_hyp)
            for other_hyp_idx, other_hyp in enumerate(ordered_null_hypotheses)
            if hyp_idx != other_hyp_idx
        ]
        for hyp_idx, hyp in enumerate(ordered_null_hypotheses)
    }

    with open(os.path.join(cfg.results_dir, 'successor_neighbors.txt'), 'w') as f:
        for key, neighbors in successor_neighbors.items():
            f.write(f"{key}: {neighbors}\n")

    weighted_G = np.zeros((num_hypotheses, num_hypotheses))
    for k1, hyp1 in enumerate(ordered_null_hypotheses):
        neighboring_hypotheses = [neighbor[1] for neighbor in successor_neighbors[(k1, hyp1)]]
        neighbor_diffs = [abs(mu1 - mu0) for (mu0, mu1) in neighboring_hypotheses]
        weights_hyp1 = allocate_weights(neighbor_diffs, beta=cfg.beta)
        sig_weights = len([w for w in weights_hyp1 if w > 1e-3])
        
        if sig_weights > 1:
            print(f"Hypothesis {k1} (mu0={hyp1[0]:.2f}, mu1={hyp1[1]:.2f}) has {sig_weights} significant neighbors with weights: {weights_hyp1}")
            
        print(f"Null hypothesis {k1} (mu0={hyp1[0]:.2f}, mu1={hyp1[1]:.2f}):")
        print(np.array2string(weights_hyp1, precision=3, suppress_small=True,))
        for idx, (k2, _) in enumerate(successor_neighbors[(k1, hyp1)]):
            weighted_G[k1, k2] = weights_hyp1[idx]
    return weighted_G

def visualize_weighted_graph(weighted_G, ordered_null_hypotheses, save_path=None,
                             figsize=(14, 6), weight_threshold=1e-3):
    """
    Draw weighted_G as a directed graph with hypotheses arranged left-to-right
    in their ordering (index 0 = largest mean gap, last = smallest).

    Forward edges (j > i, alpha flowing to harder hypotheses) are drawn in blue
    above the node line; backward edges are drawn in red below it.  Edge
    thickness is proportional to weight.  Only edges above weight_threshold
    are shown.
    """
    import networkx as nx

    num_h = len(ordered_null_hypotheses)
    G = nx.DiGraph()
    for i in range(num_h):
        G.add_node(i)

    forward_edges, backward_edges = [], []
    edge_weights = {}
    for i in range(num_h):
        for j in range(num_h):
            w = weighted_G[i, j]
            if w > weight_threshold:
                G.add_edge(i, j, weight=w)
                edge_weights[(i, j)] = w
                (forward_edges if j > i else backward_edges).append((i, j))

    # Linear layout: hypothesis index maps directly to x-position
    pos = {i: (i, 0.0) for i in range(num_h)}
    node_labels = {
        i: f"H{i}\n({mu0:.2f},{mu1:.2f})"
        for i, (mu0, mu1) in enumerate(ordered_null_hypotheses)
    }

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(G, pos, node_color='lightsteelblue', node_size=900, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7, ax=ax)

    max_w = max(edge_weights.values()) if edge_weights else 1.0

    if forward_edges:
        fwd_w = [edge_weights[e] / max_w * 6 for e in forward_edges]
        nx.draw_networkx_edges(
            G, pos, edgelist=forward_edges, width=fwd_w,
            edge_color='steelblue', alpha=0.75, ax=ax,
            connectionstyle='arc3,rad=-0.35', arrows=True,
            arrowsize=15, min_source_margin=20, min_target_margin=20,
        )

    if backward_edges:
        bwd_w = [edge_weights[e] / max_w * 6 for e in backward_edges]
        nx.draw_networkx_edges(
            G, pos, edgelist=backward_edges, width=bwd_w,
            edge_color='tomato', alpha=0.75, ax=ax,
            connectionstyle='arc3,rad=0.35', arrows=True,
            arrowsize=15, min_source_margin=20, min_target_margin=20,
        )

    # Label edges with weight values
    edge_label_dict = {e: f"{edge_weights[e]:.2f}" for e in edge_weights}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label_dict, font_size=6, ax=ax)

    ax.set_title(
        "Weighted Transition Graph  (blue = forward, red = backward)\n"
        "Node ordering: H0 = largest mean gap → last = smallest",
        fontsize=11,
    )
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Graph saved to {save_path}")
        matrix_path = os.path.splitext(save_path)[0] + '.npy'
        np.save(matrix_path, weighted_G)
        print(f"Weight matrix saved to {matrix_path}")

        # Heatmap of the weight matrix
        heatmap_path = os.path.splitext(save_path)[0] + '_heatmap.png'
        tick_labels = [f"H{i}\n({mu0:.2f},{mu1:.2f})" for i, (mu0, mu1) in enumerate(ordered_null_hypotheses)]
        fig2, ax2 = plt.subplots(figsize=figsize)
        im = ax2.imshow(weighted_G, cmap='Blues', vmin=0, vmax=1)
        fig2.colorbar(im, ax=ax2, label='weight')
        ax2.set_xticks(range(num_h))
        ax2.set_yticks(range(num_h))
        ax2.set_xticklabels(tick_labels, fontsize=7)
        ax2.set_yticklabels(tick_labels, fontsize=7)
        ax2.set_xlabel("Target hypothesis (j)", fontsize=10)
        ax2.set_ylabel("Source hypothesis (i)", fontsize=10)
        ax2.set_title("weighted_G: alpha transfer weights\n(row i → weight sent to column j)", fontsize=11)
        for i in range(num_h):
            for j in range(num_h):
                ax2.text(j, i, f"{weighted_G[i, j]:.2f}", ha='center', va='center', fontsize=6,
                         color='white' if weighted_G[i, j] > 0.5 else 'black')
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved to {heatmap_path}")
    else:
        plt.show()


def main(policy_data, policies=None, sim_means=None, real_means=None, bernoulli=False, cfg=None, labels=None, methods=None, allow_transitive=False):
    """
    Main function to run experiments.

    Parameters
    ----------
    policy_data : dict mapping policy name -> 1-D array of per-trial evaluations.
                  Policies may have different numbers of evaluations; the matrix
                  passed to the test methods is NaN-padded and capped at cfg.Nmax.
    policies    : list of policy name strings controlling ordering; derived from
                  policy_data.keys() if None.
    sim_means   : dict mapping policy name -> mean simulation performance
    real_means  : dict mapping policy name -> mean real-world performance
    bernoulli   : whether to treat data as Bernoulli (binary) or continuous
    cfg         : ExperimentConfig; if None, defaults are used
    """
    if cfg is None:
        cfg = ExperimentConfig()
    if policies is None:
        policies = list(policy_data.keys())

    os.makedirs(cfg.results_dir, exist_ok=True)
    assert not (cfg.plot_from_saved and cfg.run_new_experiment), "Cannot both plot from saved and run new experiment"
    assert cfg.plot_from_saved or cfg.run_new_experiment, "Either plot from saved or run new experiment must be True"

    with open(os.path.join(cfg.results_dir, 'config.json'), 'w') as f:
        json.dump(asdict(cfg), f, indent=2)

    n_policies = len(policies)
    Nmax = min(cfg.Nmax, max(len(policy_data[p]) for p in policies))
    num_hypotheses = n_policies * (n_policies - 1) // 2

    # Convert to NaN-padded matrix for the test methods
    policy_matrix = _to_policy_matrix(policy_data, policies, Nmax)

    policy_index = {i: policies[i] for i in range(n_policies)}
    real_sim_means = [(real_means[p], sim_means[p]) for p in policies]

    null_hypotheses = [
        (sim_means[policies[i]], sim_means[policies[j]])
        for i in range(n_policies) for j in range(i + 1, n_policies)
    ]

    null_hypotheses_policy_indices = [
        (i, j) for i in range(n_policies) for j in range(i + 1, n_policies)
    ]

    print("Ordered policy pairs (by time to decision on null hypothesis mu0 < mu1): ")
    hyp_diffs = [abs(mu1 - mu0) for (mu0, mu1) in null_hypotheses]

    # ordering hypotheses by absolute difference in means (largest gap = easiest to detect = most power)
    # each tuple is (null_hypothesis which is (sim0, sim1), policy_indices, mean_diff)
    ordered_hypotheses = sorted(zip(null_hypotheses, null_hypotheses_policy_indices, hyp_diffs), key=lambda x: x[2], reverse=True)
    ordered_hyp_diffs = [item[2] for item in ordered_hypotheses]
    ordered_null_hypotheses = [item[0] for item in ordered_hypotheses]
    ordered_null_hypotheses_policy_indices = [item[1] for item in ordered_hypotheses]

    alpha_per_hypothesis = allocate_alpha(ordered_hyp_diffs, cfg.alpha, beta=cfg.beta)
    alpha_per_hypothesis_weighted_bonferroni = allocate_alpha(ordered_hyp_diffs, cfg.alpha, beta=-cfg.beta)
    # Fully connected graph
    weighted_G_full = fully_connected_graph(ordered_null_hypotheses, cfg)

    # Soft-masked graph (forward edges weighted by mean gap, backward edges allowed but penalized)
    weighted_G_soft = soft_masked_graph(ordered_null_hypotheses, cfg)

    if cfg.graph_type == "fully_connected":
        weighted_G = weighted_G_full
    elif cfg.graph_type == "soft_masked":
        weighted_G = weighted_G_soft
    else:
        raise ValueError(f"Unknown graph_type {cfg.graph_type!r}; expected 'soft_masked' or 'fully_connected'")

    for i in range(len(weighted_G)):
        assert abs(sum(weighted_G[i]) - 1) <= 1e-3

    print("Alpha allocated to each hypotheses")
    with open(f'{cfg.results_dir}/alpha_allocation.txt', 'w') as f:
        for i, null in enumerate(ordered_null_hypotheses):
            line = f"Hypothesis {i} (mu0={null[0]:.2f}, mu1={null[1]:.2f}): alpha = {alpha_per_hypothesis[i]:.4f}\n"
            print(line.strip())
            f.write(line)

    plot_labels = [labels[p] for p in policies] if labels is not None else None
    run_experiments(
        n_policies, ordered_hypotheses,
        policy_matrix, Nmax, alpha_per_hypothesis,
        alpha_per_hypothesis_weighted_bonferroni, weighted_G,
        policy_index, real_sim_means, bernoulli,
        cfg=cfg, labels=plot_labels, methods=methods, allow_transitive=allow_transitive,
    )
