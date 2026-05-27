'''
RoboArena Bradley-Terry model fitting.
Source: https://github.com/pranavatreya/roboarena_backend/blob/main/central_server/central_server.py
'''

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multitest.bradley_terry_davidson import fit_bt_davidson
from multitest.bradley_terry_em import em_hybrid
from multitest.bradley_terry import build_wins_matrix, fit_bradley_terry

logging.basicConfig(level=logging.INFO)
# Leaderboard algorithm hyper-params
NUM_RANDOM_SEEDS = 100

# Paths:
DATA_PATH = "/n/fs/irom-testing/multitest/data/roboarena/data2/per_trial_progress_data.csv"
PREF_PATH = "/n/fs/irom-testing/multitest/data/roboarena/data2/policy_preferences.csv"
BASE_SAVE_DIR = "/n/fs/irom-testing/multitest/outputs/roboarena/roboarena_bradley_terry"

"""
Preferences are saved in the following format:
if preference == "A":
    preference_value = 0
elif preference == "B":
    preference_value = 1
elif preference == "TIE":
    preference_value = 0.5

"""
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
            data_rows.append([i, j, 2]) # A wins
            preference_counts[policy_names[i]] += 1
            preference_counts[policy_names[j]] += 1
        elif np.isclose(pref, 1.0):
            data_rows.append([i, j, 0]) # B wins
            preference_counts[policy_names[i]] += 1
            preference_counts[policy_names[j]] += 1
        else:
            raise ValueError(f"Invalid preference value: {pref}")

    return np.array(data_rows, dtype=float), preference_counts

def csv_preferences_to_pairwise_data_w_progress(
    csv_path,
    progress_csv_path,
    policy_names,
    ignore_ties=True,
    has_header=True,
    require_progress=False,
):
    if has_header:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, header=None, names=["A", "B", "preference"])

    progress_df = None
    if progress_csv_path is not None:
        progress_df = pd.read_csv(progress_csv_path)

        if len(progress_df) < len(df):
            raise ValueError(
                f"Progress CSV has fewer rows ({len(progress_df)}) "
                f"than preference CSV ({len(df)})."
            )
        
    policy_to_idx = {p: i for i, p in enumerate(policy_names)}
    preference_counts = {p: 0 for p in policy_names}
    data_rows = []

    for row_idx, row in df.iterrows():
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
            outcome = 1
        elif np.isclose(pref, 0.0):
            outcome = 2  # A wins
        elif np.isclose(pref, 1.0):
            outcome = 0  # B wins
        else:
            raise ValueError(f"Invalid preference value: {pref}")
        
        if progress_df is not None:
            progress_row = progress_df.iloc[row_idx]

            if a_name not in progress_df.columns or b_name not in progress_df.columns:
                raise ValueError(
                    f"Missing progress columns for row {row_idx}: "
                    f"A={a_name}, B={b_name}"
                )

            i_partial = progress_row[a_name]
            j_partial = progress_row[b_name]

            if pd.isna(i_partial) or pd.isna(j_partial):
                if require_progress:
                    raise ValueError(
                        f"Missing progress scores at row {row_idx}: "
                        f"A={a_name}, B={b_name}, "
                        f"i_partial={i_partial}, j_partial={j_partial}"
                    )
                else:
                    i_partial = 0.0 if pd.isna(i_partial) else i_partial
                    j_partial = 0.0 if pd.isna(j_partial) else j_partial

            data_rows.append([
                i,
                j,
                outcome,
                float(i_partial),
                float(j_partial),
            ])
        else:
            data_rows.append([i, j, outcome])

        preference_counts[policy_names[i]] += 1
        preference_counts[policy_names[j]] += 1

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
                if np.isclose(xi, xj): # tie
                    outcome = 1
                elif xi > xj: # A wins
                    outcome = 2
                else:         # B wins
                    outcome = 0

                comparison_counts[df.columns[i]] += 1
                comparison_counts[df.columns[j]] += 1
                data_rows.append([i, j, outcome])
    data = np.asarray(data_rows, dtype=float)
    return data, list(df.columns), policy_progress, policy_counts, comparison_counts

def csv_to_pairwise_data_with_progress_scores(csv_path, columns=None, seed=None):
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
                if np.isclose(xi, xj): # tie
                    outcome = 1
                elif xi > xj: # A wins
                    outcome = 2
                else:         # B wins
                    outcome = 0

                comparison_counts[df.columns[i]] += 1
                comparison_counts[df.columns[j]] += 1
                data_rows.append([i, j, outcome, xi, xj])
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
        i_old, j_old = int(row[0]), int(row[1])
        if i_old in selected_old_indices and j_old in selected_old_indices:
            rows.append([
                old_to_new[i_old],
                old_to_new[j_old],
                *row[2:]
            ])

    return np.asarray(rows, dtype=float), old_to_new, new_to_old


def load_roboarena_data(selected_policies=None, ignore_ties=False):
    """Load progress and preference pairwise data, subselected to selected_policies."""
    raw_names = np.genfromtxt(DATA_PATH, delimiter=",", dtype=str, max_rows=1)
    progress_data, policy_names, policy_progress, policy_counts, comparison_counts = \
        csv_to_pairwise_data(DATA_PATH, columns=raw_names, seed=0)
    
    pref_data_w_scores, _ = \
        csv_preferences_to_pairwise_data_w_progress(PREF_PATH, DATA_PATH, policy_names, ignore_ties=ignore_ties)
    
    pref_data, preference_counts = csv_preferences_to_pairwise_data(
        PREF_PATH, policy_names, ignore_ties=ignore_ties
    )
    if selected_policies is not None:
        pref_data, old_to_new, new_to_old = subselect_and_remap_prefs(
            selected_policies, policy_names, pref_data
        )
        progress_data, _, _ = subselect_and_remap_prefs(
            selected_policies, policy_names, progress_data
        )
        pref_data_w_scores, _, _ = subselect_and_remap_prefs(
            selected_policies, policy_names, pref_data_w_scores
        )
        policy_indices = list(old_to_new.keys())
        policy_names = [policy_names[k] for k in policy_indices]
    else:
        old_to_new = {i: i for i in range(len(policy_names))}
        new_to_old = {i: i for i in range(len(policy_names))}
        policy_indices = list(range(len(policy_names)))

    # Returns progress data as preferences!
    return (
        pref_data, progress_data, pref_data_w_scores, policy_names, policy_progress,
        preference_counts, old_to_new, new_to_old, policy_indices
    )


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

def save_bt_rankings(ranked_names, ranked_scores, save_dir, metric):
    path = os.path.join(save_dir, f"bt_rankings_{metric}.txt")
    with open(path, "w") as f:
        for rank, (name, score) in enumerate(zip(ranked_names, ranked_scores)):
            f.write(f"#{rank + 1}: {name}  (BT={score:.4f})\n")
    print(f"Rankings saved to {path}")


def roboarena_ranking_bt_davidson(metric="preference", subselect_and_remap=True, selected_policies=None, save_dir=BASE_SAVE_DIR):
    if subselect_and_remap and selected_policies is None:
        selected_policies = ["paligemma_binning_droid", "pi0_droid", "paligemma_vq_droid",
                             "paligemma_fast_specialist_droid", "paligemma_fast_droid",
                             "paligemma_diffusion_droid", "pi0_fast_droid"]
    pref_data, progress_data, _, policy_names, policy_progress, preference_counts, old_to_new, new_to_old, policy_indices = \
        load_roboarena_data(selected_policies if subselect_and_remap else None)
    data = pref_data if metric == "preference" else progress_data

    pref_df = pd.DataFrame(data, columns=["i", "j", "y"])
    if pref_df.empty:
        return []

    counts_i = pref_df["i"].value_counts()
    counts_j = pref_df["j"].value_counts()
    eval_counts = (counts_i.add(counts_j, fill_value=0)).astype(int).to_dict()
    eval_counts = {int(k): v for k, v in eval_counts.items()}

    bt_board, tie_nu = fit_bt_davidson(pref_df)
    print(f"BT-Davidson fit complete: {len(bt_board)} policies, tie_nu={tie_nu:.4f}")

    policy_mapping = {policy_names[old_to_new[k]]: old_to_new[k] for k in policy_indices}
    policy_oracle_performance = {old_to_new[k]: policy_progress[policy_names[old_to_new[k]]] for k in policy_indices}
    p_old = bt_board["score"].to_numpy()
    print("Preference counts:", preference_counts)
    print("Policies:", policy_mapping)

    id_to_policy = {v: k for k, v in policy_mapping.items()}
    bt_board["policy_name"] = bt_board["policy"].astype(int).map(id_to_policy)
    bt_board = bt_board[["policy", "policy_name", "score", "std"]]
    print(bt_board)

    os.makedirs(save_dir, exist_ok=True)
    save_bt_rankings(bt_board["policy_name"].tolist(), bt_board["score"].to_numpy(), save_dir, metric)
    plot_bt_result(policy_names, eval_counts, policy_oracle_performance, p_old, save_dir, metric=metric)

def roboarena_ranking_naive(metric="preference", subselect_and_remap=True, selected_policies=None, save_dir=BASE_SAVE_DIR):
    if subselect_and_remap and selected_policies is None:
        selected_policies = ["paligemma_binning_droid", "pi0_droid", "paligemma_vq_droid",
                             "paligemma_fast_specialist_droid", "paligemma_fast_droid",
                             "paligemma_diffusion_droid", "pi0_fast_droid"]
    pref_data, progress_data, _, policy_names, policy_progress, preference_counts, old_to_new, new_to_old, policy_indices = \
        load_roboarena_data(selected_policies if subselect_and_remap else None)
    data = pref_data if metric == "preference" else progress_data

    if len(data) == 0:
        return []

    n_teams = len(policy_names)
    wins_matrix = build_wins_matrix(data, n_teams)
    bt_scores = fit_bradley_terry(wins_matrix)

    print(f"Bradley-Terry fit complete: {n_teams} policies")

    policy_mapping = {policy_names[old_to_new[k]]: old_to_new[k] for k in policy_indices}
    policy_oracle_performance = {old_to_new[k]: policy_progress[policy_names[old_to_new[k]]] for k in policy_indices}

    print("Preference counts:", preference_counts)
    print("Policies:", policy_mapping)

    id_to_policy = {v: k for k, v in policy_mapping.items()}
    idx_ranked = np.argsort(bt_scores)[::-1]
    print("\nBradley-Terry scores (strongest to weakest):")
    for rank, idx in enumerate(idx_ranked):
        print(f"  #{rank + 1}: {id_to_policy.get(idx, idx)}  (BT={bt_scores[idx]:.4f})")
    os.makedirs(save_dir, exist_ok=True)
    save_bt_rankings([id_to_policy[idx] for idx in idx_ranked], bt_scores[idx_ranked], save_dir, metric)

def roboarena_ranking_em(metric='preference', use_partials=False, subselect_and_remap=True, selected_policies=None, save_dir=BASE_SAVE_DIR):
    if subselect_and_remap and selected_policies is None:
        selected_policies = ["paligemma_binning_droid", "pi0_droid",
                             "paligemma_diffusion_droid", "pi0_fast_droid"]
    pref_data, progress_data, pref_data_w_scores, policy_names, policy_progress, preference_counts, old_to_new, new_to_old, policy_indices = \
        load_roboarena_data(selected_policies if subselect_and_remap else None)

    if use_partials:
        data = pref_data_w_scores
        df = pd.DataFrame(data, columns=["i", "j", "y", "i_partial", "j_partial"])
    else:
        data = pref_data if metric == "preference" else progress_data
        df = pd.DataFrame(data, columns=["i", "j", "y"])
        if df.empty:
            return []

    bt_board = em_hybrid(df, use_partials=use_partials)

    policy_mapping = {policy_names[old_to_new[k]]: old_to_new[k] for k in policy_indices}
    policy_oracle_performance = {old_to_new[k]: policy_progress[policy_names[old_to_new[k]]] for k in policy_indices}
    p_old = bt_board["score"].to_numpy()

    print("Preference counts:", preference_counts)
    print("Policies:", policy_mapping)

    id_to_policy = {v: k for k, v in policy_mapping.items()}
    bt_board["policy_name"] = bt_board["policy"].astype(int).map(id_to_policy)
    bt_board = bt_board[["policy", "policy_name", "score"]]
    print(bt_board)
    os.makedirs(save_dir, exist_ok=True)
    save_bt_rankings(bt_board["policy_name"].tolist(), bt_board["score"].to_numpy(), save_dir, f"{metric}_partials{use_partials}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subfolder", type=str, default=None, help="Optional subfolder under outputs/<exp_name>/")
    parser.add_argument("--exp_name", type=str, default="roboarena4",
                        choices=["roboarena4", "roboarena7"],
                        help="Experiment to run")
    parser.add_argument("--method", type=str, default="naive", choices=["davidson", "em", "naive"], help="BT fitting method to use")
    args = parser.parse_args()

    four_policies = ["paligemma_binning_droid", "pi0_droid", "paligemma_diffusion_droid", "pi0_fast_droid"]
    seven_policies = ["paligemma_binning_droid",
            "pi0_droid",
            "paligemma_vq_droid",
            "paligemma_fast_specialist_droid",
            "paligemma_fast_droid",
            "paligemma_diffusion_droid",
            "pi0_fast_droid"]
    
    save_dir = os.path.join(BASE_SAVE_DIR, args.exp_name, args.method)
    if args.subfolder:
        save_dir = os.path.join(save_dir, args.subfolder)
    os.makedirs(save_dir, exist_ok=True)

    bt_method = args.method
    
    if args.exp_name == "roboarena4":
        selected_policies = four_policies
    elif args.exp_name == "roboarena7":
        selected_policies = seven_policies

    if bt_method == "davidson":
        roboarena_ranking_bt_davidson(metric="preference", subselect_and_remap=True, selected_policies=selected_policies, save_dir=save_dir)
        roboarena_ranking_bt_davidson(metric="progress", subselect_and_remap=True, selected_policies=selected_policies, save_dir=save_dir)

    elif bt_method == "em":
        roboarena_ranking_em(metric='preference', use_partials=True, subselect_and_remap=True, selected_policies=selected_policies, save_dir=save_dir)
        roboarena_ranking_em(metric='preference', use_partials=False, subselect_and_remap=True, selected_policies=selected_policies, save_dir=save_dir)

        # roboarena_ranking_em(metric='progress', use_partials=True, subselect_and_remap=True, selected_policies=selected_policies, save_dir=save_dir)
        roboarena_ranking_em(metric='progress', use_partials=False, subselect_and_remap=True, selected_policies=selected_policies, save_dir=save_dir)

    elif bt_method == "naive":
        roboarena_ranking_naive(metric="preference", subselect_and_remap=True, selected_policies=selected_policies, save_dir=save_dir)
        roboarena_ranking_naive(metric="progress", subselect_and_remap=True, selected_policies=selected_policies, save_dir=save_dir)