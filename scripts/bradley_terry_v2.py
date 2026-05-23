"""
Implementation of basic Bradley-Terry Model fitting procedure, as described in 
Equations (4) and (5) at https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model

"""

import numpy as np
import os 
import argparse 
import copy
from scipy.stats import pearsonr
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_progress(total_progress, total_counts, policy_names, save_dir):
    # Inputs:
    # total_progress : np.ndarray
    # total_counts   : np.ndarray
    # policy_names   : list[str]

    x = np.arange(len(policy_names))
    width = 0.4

    fig, ax1 = plt.subplots(figsize=(11, 5))

    # Bar plot for average progress
    bars = ax1.bar(
        x - width/2,
        total_progress,
        width,
        label="Average Progress"
    )

    ax1.set_ylabel("Average Progress")
    ax1.set_ylim(0, 1)

    # Second axis for evaluation counts
    ax2 = ax1.twinx()

    ax2.bar(
        x + width/2,
        total_counts,
        width,
        label="Evaluation Count",
        alpha=0.7
    )

    ax2.set_ylabel("Number of Evaluations")

    # Shared x-axis
    ax1.set_xticks(x)
    ax1.set_xticklabels(policy_names, rotation=20)

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper right"
    )

    ax1.set_title("Policy Progress vs Evaluation Count")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "policy_progress_and_counts.png"))

def csv_to_pairwise_data(csv_path, columns=None, seed=None):
    rng = np.random.default_rng(seed)

    # Load CSV; blanks become NaN
    df = pd.read_csv(csv_path)

    # Keep only rows with more than 1 observed numeric value
    df = df[df.notna().sum(axis=1) > 1]

    # Keep only selected columns if specified
    if columns is not None:
        df = df[columns]

    policy_progress = df.mean(axis=0, skipna=True)

    # Number of evaluations per policy
    policy_counts = (
        df.notna()
          .sum(axis=0)
          .to_numpy()
    )
    comparison_counts = {
        policy: 0 for policy in df.columns
    }
    data_rows = []

    for _, row in df.iterrows():
        # indices of policies that have values in this row
        observed = np.where(row.notna().to_numpy())[0]

        for a in range(len(observed) - 1):
            for b in range(a + 1, len(observed)):
                i = observed[a]
                j = observed[b]

                xi = row.iloc[i]
                xj = row.iloc[j]

                if np.isclose(xi, xj):
                    continue
                elif xj > xi:
                    outcome = 1.0
                    comparison_counts[df.columns[i]] += 1
                    comparison_counts[df.columns[j]] += 1
                else:
                    outcome = 0.0
                    comparison_counts[df.columns[i]] += 1
                    comparison_counts[df.columns[j]] += 1

                data_rows.append([i, j, outcome])
    data = np.asarray(data_rows, dtype=float)
    return data, list(df.columns), policy_progress, policy_counts, comparison_counts

def plot_bt_result(policy_names, total_counts, policy_oracle_performance, p_old, pearson_r, comparison_counts, save_dir,  metric = "progress"):
    # Sort by Bradley-Terry score
    idx_ranked = np.argsort(p_old)[::-1]

    policy_names_ranked = [policy_names[i]
        .replace("_droid", "")
        .replace("paligemma", "pg") for i in idx_ranked]
    p_ranked = p_old[idx_ranked]
    oracle_ranked = [policy_oracle_performance[i] for i in idx_ranked]
    counts_ranked = total_counts[idx_ranked]
    comparison_counts_ranked = [comparison_counts[policy_names[i]] for i in idx_ranked]
    
    x = np.arange(len(policy_names))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(11, 5))

    bars = ax1.bar(
        x,
        oracle_ranked,
        width=0.5,
    )

    # Rich x-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [
            (
                f"{name}\n"
                f"evals={counts_ranked[i]}\n"
                f"pairs={comparison_counts_ranked[i]}"
            )
            for i, name in enumerate(policy_names_ranked)
        ],
        fontsize=12,
    )

    # BT score above each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        x_center = bar.get_x() + bar.get_width()/2

        ax1.text(
            x_center,
            height + 0.01,
            f"BT={p_ranked[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
        )

    ax1.set_ylabel("Oracle Progress", fontsize=14)
    ax1.set_title(
        f"Bradley-Terry Scores ({metric}) vs Oracle Progress",
        fontsize=16,
    )

    # Pearson box
    ax1.text(
        0.98,
        0.80,
        f"Pearson r = {pearson_r:.3f}",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", alpha=0.15),
        fontsize=14,
    )

    ax1.set_ylim(0, max(oracle_ranked) + 0.08)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"bt_results_{metric}.png"),
        dpi=200,
    )

    

def combine_policy_progress(
    progress1,
    counts1,
    comparison_counts1,
    progress2,
    counts2,
    comparison_counts2
):
    """
    Combine two policy-progress estimates using
    weighted averaging by evaluation count.
    """

    total_counts = counts1 + counts2

    total_progress = np.divide(
        progress1 * counts1 + progress2 * counts2,
        total_counts,
        out=np.zeros_like(total_counts, dtype=float),
        where=total_counts > 0
    )

    # Combine comparison counts if provided
    comparison_counts = {
        p: comparison_counts1.get(p, 0) + comparison_counts2.get(p, 0)
        for p in policy_names
    }
    
    return total_progress, total_counts, comparison_counts

def csv_preferences_to_pairwise_data(
    csv_path,
    policy_names,
    ignore_ties=True,
    has_header=True,
):
    if has_header:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, header=None, names=["A", "B", "preference"])

    policy_to_idx = {p: i for i, p in enumerate(policy_names)}
    preference_counts = {p: 0 for p in policy_names}
    data_rows = []

    for _, row in df.iterrows():
        a_name = row["A"]
        b_name = row["B"]
        pref = float(row["preference"])

        if a_name not in policy_to_idx or b_name not in policy_to_idx:
            continue

        i = policy_to_idx[a_name]
        j = policy_to_idx[b_name]

        if np.isclose(pref, 0.5):
            if ignore_ties:
                continue
            else:
                data_rows.append([i, j, 0.5])
        elif np.isclose(pref, 1.0):
            data_rows.append([i, j, 1.0])
            preference_counts[policy_names[i]] += 1
            preference_counts[policy_names[j]] += 1
        elif np.isclose(pref, 0.0):
            data_rows.append([i, j, 0.0])
            preference_counts[policy_names[i]] += 1
            preference_counts[policy_names[j]] += 1
        else:
            raise ValueError(f"Invalid preference value: {pref}")

    return np.array(data_rows, dtype=float), preference_counts

def subselect_and_remap(selected_policies, policy_names, data):
    name_to_old_idx = {name: i for i, name in enumerate(policy_names)}

    old_to_new = {
        name_to_old_idx[name]: new_idx
        for new_idx, name in enumerate(selected_policies)
    }

    new_to_old = {
        new_idx: old_idx
        for old_idx, new_idx in old_to_new.items()
    }

    selected_old_indices = set(old_to_new.keys())

    rows = []

    for row in data:
        i_old, j_old, outcome = int(row[0]), int(row[1]), row[2]

        if i_old in selected_old_indices and j_old in selected_old_indices:
            rows.append([
                old_to_new[i_old],
                old_to_new[j_old],
                outcome
            ])

    return np.asarray(rows, dtype=float), old_to_new, new_to_old

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "This script fits a Bradley-Terry model to rank options."
        )
    )
    parser.add_argument(
        "-p",
        "--is_pref",
        type=int,
        default=0,
        help=("Whether the data is already in preference form. If false, the data is in " 
              "progress form, and preferences will be assigned by pairwise higher progress. " 
              "Defaults to True."),
    )
    parser.add_argument(
        "-t",
        "--abs_tol",
        type=float,
        default=1e-5,
        help=("Absolute tolerance of Bradley Terry score ranks at termination. " 
              "Defaults to 1e-5."),
    )

    parser.add_argument(
        "-ng",
        "--n_games",
        type=int,
        default=15,
        help=("Number of total games played by each teams / policies. " 
              "Defaults to 15."),
    )

    parser.add_argument(
        "-all",
        "--use_all_data",
        type=bool,
        default=False,
        help=("Whether to use all available data for Bradley-Terry score computation, or to use a subset of the data from n_games. "),
    )

    data_path1 = "/n/fs/irom-testing/multitest/data/roboarena/data1/per_trial_progress_data.csv"
    data_path2 = "/n/fs/irom-testing/multitest/data/roboarena/data2/per_trial_progress_data.csv"
    policy_names = np.genfromtxt(
        data_path1,
        delimiter=",",
        dtype=str,
        max_rows=1,
    )

    pref_path1 = "/n/fs/irom-testing/multitest/data/roboarena/data1/policy_preferences.csv"
    pref_path2 = "/n/fs/irom-testing/multitest/data/roboarena/data2/policy_preferences.csv"
    metric = "preference" # oracle or preference
    subselect_and_remap = False # whether to use only a subset of policies for BT ranking, and to remap indices accordingly
    save_dir = "/n/fs/irom-testing/multitest/outputs/roboarena/bradley_terry"
    os.makedirs(save_dir, exist_ok=True)

    progress_data1, policy_names1, policy_progress1, policy_counts1, comparison_counts1 = csv_to_pairwise_data(data_path1, seed=0)
    progress_data2, policy_names2, policy_progress2, policy_counts2, comparison_counts2 = csv_to_pairwise_data(data_path2, columns=policy_names, seed=0)
    total_progress_data = np.concatenate([progress_data1, progress_data2], axis=0)
    total_policy_progress, total_policy_counts, comparison_counts = combine_policy_progress(
        policy_progress1, policy_counts1, comparison_counts1,
        policy_progress2, policy_counts2, comparison_counts2
    )
    plot_progress(total_policy_progress, total_policy_counts, policy_names, save_dir)
    
    if metric == "progress":
        data = total_progress_data
    elif metric == "preference":
        pref_data1, preference_counts1 = csv_preferences_to_pairwise_data(pref_path1, policy_names)
        pref_data2, preference_counts2 = csv_preferences_to_pairwise_data(pref_path2, policy_names)
        data = np.concatenate([pref_data1, pref_data2], axis=0)
        comparison_counts = {p: preference_counts1.get(p, 0) + preference_counts2.get(p, 0) for p in policy_names}
    args = parser.parse_args()
    n_games = args.n_games
    is_pref = bool(args.is_pref)
    abs_tol = args.abs_tol

    if subselect_and_remap: 
        selected_policies = ["pi0_droid",
                        "pi0_fast_droid",
                        "paligemma_diffusion_droid",
                        "paligemma_binning_droid",
                        ]
        data, old_to_new, new_to_old = subselect_and_remap(selected_policies, policy_names, data)
    else:
        selected_policies = policy_names
    n_teams = len(selected_policies)
    n_games = data.shape[0]

    ARRAY = np.zeros((n_teams, n_teams))
    for i in range(n_games):
        if data[i, 2] > 0.75: 
            ARRAY[int(data[i, 1]), int(data[i, 0])] += 1
        elif np.isclose(data[i, 2], 0.5):
            pass
            # ARRAY[int(data[i, 1]), int(data[i, 0])] += 0.5
            # ARRAY[int(data[i, 0]), int(data[i, 1])] += 0.5
        else:
            ARRAY[int(data[i, 0]), int(data[i, 1])] += 1
    
    print("Data array of 'wins' / 'losses': ")
    print()
    print(ARRAY)

    # Define some optimization initial conditions
    current_tol = 1. 
    p_old = np.ones(n_teams)
    iteration_counts = 0 

    #
    # Run Bradley-Terry updates to convergence 
    #
    while current_tol >= abs_tol: 
        # Update to p_new
        try: 
            del p_new
        except:
            pass 
        
        # Update p --> p' (Equation 5)
        p_new = np.zeros(n_teams)
        for i in range(n_teams):
            num = 0. 
            den = 0.
            for j in range(n_teams):
                if j != i:
                    num += (ARRAY[i, j]*p_old[j])/(p_old[i] + p_old[j])
                    den += (ARRAY[j, i])/(p_old[i] + p_old[j])
            p_new[i] = num / den
        
        # Normalize p_new by geometric mean (Equation 4)
        norm_const = (np.prod(p_new))**(1 / n_teams)
        p_new /= norm_const

        current_tol = np.linalg.norm(p_new - p_old)
        p_old = copy.deepcopy(p_new)
        iteration_counts += 1
    
    # Print some summary results
    print()
    print(f"Terminated in {iteration_counts} iterations!")
    print()
    print("Power scores: ")
    print(p_old)
    print()
    print("Ranking (ascending): ")
    print(np.argsort(p_old))
    print()
    print("Ranking (strongest to weakest): ")
    print(np.argsort(p_old)[::-1])

    idx_ranked = np.argsort(p_old)[::-1]
    print()
    print("Power scores (descending): ")
    print(p_old[idx_ranked])

    print()
    print("Policy name ranking: ")
    for i in range(n_teams):
        print(f"Rank #{i+1}: ", selected_policies[int(idx_ranked[i])])

    policy_oracle_performance = [total_policy_progress[list(policy_names).index(p)] for p in selected_policies]
    np.save(os.path.join(save_dir, "roboarena_policy_performance_oracle_progress.npy"), policy_oracle_performance)
    
    pearson_stat, pearson_pvalue = pearsonr(policy_oracle_performance, p_old)
    print("Pearson R^2: ", pearson_stat)

    ranking_vector = np.zeros(n_teams)
    for i in range(n_teams):
        ranking_vector[idx_ranked[n_teams-1-i]] = i 
    
    print(ranking_vector)
    silly_pearson_stat, silly_pearson_pvalue = pearsonr(policy_oracle_performance, ranking_vector)
    print("Silly Pearson R^2: ", silly_pearson_stat)
    
    plot_bt_result(selected_policies, total_policy_counts, policy_oracle_performance, p_old, pearson_stat, comparison_counts=comparison_counts,save_dir=save_dir, metric=metric)
