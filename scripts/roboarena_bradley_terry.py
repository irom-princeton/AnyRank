'''
RoboArena Bradley-Terry model fitting.
Source: https://github.com/pranavatreya/roboarena_backend/blob/main/central_server/central_server.py
'''

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multitest.bradley_terry_davidson import fit_bt_davidson
from multitest.bradley_terry_em import em_hybrid

logging.basicConfig(level=logging.INFO)
# Leaderboard algorithm hyper-params
EXCLUDE = {"PI0", "PI0_FAST"}
NUM_RANDOM_SEEDS = 100

# Paths:
DATA_PATH = "/n/fs/irom-testing/multitest/data/roboarena/data2/per_trial_progress_data.csv"
PREF_PATH = "/n/fs/irom-testing/multitest/data/roboarena/data2/policy_preferences.csv"
SAVE_DIR = "/n/fs/irom-testing/multitest/outputs/roboarena/roboarena_bradley_terry"
os.makedirs(SAVE_DIR, exist_ok=True)


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
                data_rows.append([i, j, 1])
                preference_counts[policy_names[i]] += 1
                preference_counts[policy_names[j]] += 1
        elif np.isclose(pref, 0.0):
            data_rows.append([i, j, 2])
            preference_counts[policy_names[i]] += 1
            preference_counts[policy_names[j]] += 1
        elif np.isclose(pref, 1.0):
            data_rows.append([i, j, 0.0])
            preference_counts[policy_names[i]] += 1
            preference_counts[policy_names[j]] += 1
        else:
            raise ValueError(f"Invalid preference value: {pref}")

    return np.array(data_rows, dtype=float), preference_counts

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
                
                # track comparisons:
                if np.isclose(xi, xj):
                    outcome = 1
                elif xi > xj:
                    outcome = 2
                else:
                    outcome = 0

                comparison_counts[df.columns[i]] += 1
                comparison_counts[df.columns[j]] += 1
                data_rows.append([i, j, outcome])
    data = np.asarray(data_rows, dtype=float)
    return data, list(df.columns), policy_progress, policy_counts, comparison_counts

def subselect_and_remap_prefs(selected_policies, policy_names, data):
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

def plot_bt_result(policy_names, total_counts, policy_oracle_performance, p_old, save_dir,  metric = "progress"):
    # Sort by Bradley-Terry score
    idx_ranked = np.argsort(p_old)[::-1]

    policy_names_ranked = [policy_names[i]
        .replace("_droid", "")
        .replace("paligemma", "pg") for i in idx_ranked]
    p_ranked = p_old[idx_ranked]
    oracle_ranked = [policy_oracle_performance[i] for i in idx_ranked]
    counts_ranked = [total_counts[k] for k in idx_ranked]
    # comparison_counts_ranked = [comparison_counts[policy_names[i]] for i in idx_ranked]
    
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

    ax1.set_ylim(0, max(oracle_ranked) + 0.08)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"bt_results_{metric}.png"),
        dpi=200,
    )

def roboarena_ranking_bt_davidson(metric="preference", subselect_and_remap=True):
    policy_names = np.genfromtxt(
        DATA_PATH,
        delimiter=",",
        dtype=str,
        max_rows=1,
    )

    data, policy_names, policy_progress, policy_counts, comparison_counts = csv_to_pairwise_data(DATA_PATH, columns=policy_names, seed=0)
    if metric == "preference":
        data, preference_counts = csv_preferences_to_pairwise_data(PREF_PATH, policy_names, ignore_ties=False)
        
    
    if subselect_and_remap: 
        selected_policies = ["paligemma_binning_droid",
            "pi0_droid",
            "paligemma_vq_droid",
            "paligemma_fast_specialist_droid",
            "paligemma_fast_droid",
            "paligemma_diffusion_droid",
            "pi0_fast_droid",
                            ]
        data, old_to_new, new_to_old = subselect_and_remap_prefs(selected_policies, policy_names, data)
    
    policy_indices = list(old_to_new.keys()) if subselect_and_remap else range(len(policy_names))
    policy_names = [policy_names[k] for k in policy_indices]
    pref_df = pd.DataFrame(data, columns=["i", "j", "y"])
    if pref_df.empty:
        return []
    
    # Per-policy A/B eval counts (from filtered pref_df)
    counts_i = pref_df["i"].value_counts()
    counts_j = pref_df["j"].value_counts()
    eval_counts = (counts_i.add(counts_j, fill_value=0)).astype(int).to_dict()
    eval_counts = {int(k): v for k, v in eval_counts.items()}
    bt_board, tie_nu = fit_bt_davidson(pref_df)
    print(
            "BT-Davidson fit complete: "
            f"{len(bt_board)} policies, tie_nu={tie_nu:.4f}"
        )
    policy_mapping = {policy_names[old_to_new[k]]: old_to_new[k] for k in policy_indices}
    policy_oracle_performance = {old_to_new[k]:  policy_progress[policy_names[old_to_new[k]]] for k in policy_indices}
    p_old = bt_board["score"].to_numpy()
    stds = bt_board["std"].to_numpy()
    print("Preference counts:", preference_counts)
    print("Policies:", policy_mapping)

    id_to_policy = {v: k for k, v in policy_mapping.items()}
    bt_board["policy_name"] = bt_board["policy"].astype(int).map(id_to_policy)
    bt_board = bt_board[["policy", "policy_name", "score", "std"]]
    print(bt_board)

    # Sort by Bradley-Terry score
    plot_bt_result(policy_names, eval_counts, policy_oracle_performance, p_old, SAVE_DIR,  metric=metric)
    breakpoint()


def roboarena_ranking_em(metric='preference', use_partials=False, subselect_and_remap=True):
    policy_names = np.genfromtxt(
        DATA_PATH,
        delimiter=",",
        dtype=str,
        max_rows=1,
    )

    progress_data, policy_names, policy_progress, policy_counts, comparison_counts = csv_to_pairwise_data(DATA_PATH, columns=policy_names, seed=0)
    pref_data, preference_counts = csv_preferences_to_pairwise_data(PREF_PATH, policy_names, ignore_ties=False)
    
    if subselect_and_remap: 
        selected_policies = ["paligemma_binning_droid",
            "pi0_droid",
            # "paligemma_vq_droid",
            # "paligemma_fast_specialist_droid",
            # "paligemma_fast_droid",
            "paligemma_diffusion_droid",
            "pi0_fast_droid",
                            ]
        pref_data, old_to_new, new_to_old = subselect_and_remap_prefs(selected_policies, policy_names, pref_data)
        progress_data, _, _ = subselect_and_remap_prefs(selected_policies, policy_names, progress_data)
    
    if use_partials:
        # concatenate progress and preference data, treating progress as partial signals
        pref_df = pd.DataFrame(pref_data, columns=["i", "j", "y"])
        prog_df = pd.DataFrame(progress_data, columns=["i", "j", "y"])
        merged_df = pd.merge(pref_df, prog_df, on=["i", "j"], how="outer", suffixes=("_pref", "_prog"))
        merged_df["y"] = merged_df["y_pref"].fillna(1)  # default to tie if no preference
        merged_df["i_partial"] = merged_df["y_prog"].fillna(0)  # default to 0 partial signal if no progress data
        merged_df["j_partial"] = merged_df["y_prog"].fillna(0)
        df = merged_df[["i", "j", "y", "i_partial", "j_partial"]]
        data = df.to_numpy()
    else:
        if metric == "preference":
            data = pref_data    
        elif metric == "progress":
            data = progress_data
        df = pd.DataFrame(data, columns=["i", "j", "y"])
        if df.empty:
            return []
        
        # Per-policy A/B eval counts (from filtered pref_df)
        counts_i = df["i"].value_counts()
        counts_j = df["j"].value_counts()
        eval_counts = (counts_i.add(counts_j, fill_value=0)).astype(int).to_dict()
        eval_counts = {int(k): v for k, v in eval_counts.items()}

    bt_board = em_hybrid(df, use_partials=use_partials)
    
    policy_indices = list(old_to_new.keys()) if subselect_and_remap else range(len(policy_names))
    policy_names = [policy_names[k] for k in policy_indices]
    policy_mapping = {policy_names[old_to_new[k]]: old_to_new[k] for k in policy_indices}
    policy_oracle_performance = {old_to_new[k]:  policy_progress[policy_names[old_to_new[k]]] for k in policy_indices}
    p_old = bt_board["score"].to_numpy()
    
    print("Preference counts:", preference_counts)
    print("Policies:", policy_mapping)

    id_to_policy = {v: k for k, v in policy_mapping.items()}
    bt_board["policy_name"] = bt_board["policy"].astype(int).map(id_to_policy)
    bt_board = bt_board[["policy", "policy_name", "score"]]
    print(bt_board)

if __name__ == "__main__":
    # roboarena_ranking_bt_davidson(metric="preference")
    roboarena_ranking_em(metric='preference', use_partials=True, subselect_and_remap=True)