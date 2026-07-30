"""
Implementation of basic Bradley-Terry Model fitting procedure, as described in
Equations (4) and (5) at https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model

"""

import numpy as np
import os
import argparse
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Generic Bradley-Terry functions
# ---------------------------------------------------------------------------

def build_wins_matrix(data, n_teams):
    """
    Convert pairwise outcome data into a wins matrix.

    data : np.ndarray, shape (N, 3), columns [i, j, outcome]
        outcome > 0.75  -> j beat i
        outcome < 0.25  -> i beat j
        outcome ~ 0.5   -> tie, ignored

    Returns wins_matrix where wins_matrix[i, j] = number of times i beat j.
    """
    wins = np.zeros((n_teams, n_teams))
    for row in data:
        i, j, outcome = int(row[0]), int(row[1]), row[2]
        if outcome == 2:    # i wins
            wins[i, j] += 1
        elif outcome == 0:  # j wins
            wins[j, i] += 1
        # outcome == 1 → tie, ignored
    return wins


def fit_bradley_terry(wins_matrix, abs_tol=1e-5):
    """
    Fit Bradley-Terry scores from a wins matrix (Equations 4 and 5).

    wins_matrix[i, j] = number of times i beat j.

    Returns p : np.ndarray
        BT scores normalized so their geometric mean equals 1.
    """
    n = wins_matrix.shape[0]
    p = np.ones(n)
    tol = 1.0
    iterations = 0

    while tol >= abs_tol:
        p_new = np.zeros(n)
        for i in range(n):
            num, den = 0.0, 0.0
            for j in range(n):
                if j != i:
                    num += (wins_matrix[i, j] * p[j]) / (p[i] + p[j])
                    den += wins_matrix[j, i] / (p[i] + p[j])
            p_new[i] = num / den

        p_new /= np.prod(p_new) ** (1 / n)
        tol = np.linalg.norm(p_new - p)
        p = p_new
        iterations += 1

    print(f"Bradley-Terry converged in {iterations} iterations.")
    return p


def subselect_and_remap_policies(selected_policies, policy_names, data):
    name_to_old_idx = {name: i for i, name in enumerate(policy_names)}

    old_to_new = {
        name_to_old_idx[name]: new_idx
        for new_idx, name in enumerate(selected_policies)
    }
    new_to_old = {new_idx: old_idx for old_idx, new_idx in old_to_new.items()}
    counts = {new_idx: 0 for new_idx in new_to_old}
    selected_old_indices = set(old_to_new)

    rows = []
    for row in data:
        i_old, j_old, outcome = int(row[0]), int(row[1]), row[2]
        if i_old in selected_old_indices and j_old in selected_old_indices:
            rows.append([old_to_new[i_old], old_to_new[j_old], outcome])
            counts[old_to_new[i_old]] += 1
            counts[old_to_new[j_old]] += 1

    return np.asarray(rows, dtype=float), old_to_new, new_to_old, counts


# ---------------------------------------------------------------------------
# Roboarena data utilities
# ---------------------------------------------------------------------------

def csv_to_pairwise_data(csv_path, columns=None, seed=None):
    rng = np.random.default_rng(seed)

    df = pd.read_csv(csv_path)
    df = df[df.notna().sum(axis=1) > 1]

    if columns is not None:
        df = df[columns]

    policy_progress = df.mean(axis=0, skipna=True)
    policy_counts = df.notna().sum(axis=0).to_numpy()
    comparison_counts = {policy: 0 for policy in df.columns}
    data_rows = []

    for _, row in df.iterrows():
        observed = np.where(row.notna().to_numpy())[0]

        for a in range(len(observed) - 1):
            for b in range(a + 1, len(observed)):
                i = observed[a]
                j = observed[b]
                xi = row.iloc[i]
                xj = row.iloc[j]

                if np.isclose(xi, xj):
                    outcome = 1
                elif xi > xj:
                    outcome = 2  # i wins
                else:
                    outcome = 0  # j wins

                comparison_counts[df.columns[i]] += 1
                comparison_counts[df.columns[j]] += 1
                data_rows.append([i, j, outcome])

    data = np.asarray(data_rows, dtype=float)
    return data, list(df.columns), policy_progress, policy_counts, comparison_counts


def csv_preferences_to_pairwise_data(csv_path, policy_names, ignore_ties=True, has_header=True):
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
            if not ignore_ties:
                data_rows.append([i, j, 1])
        elif np.isclose(pref, 1.0):
            data_rows.append([i, j, 0])  # B wins
            preference_counts[policy_names[i]] += 1
            preference_counts[policy_names[j]] += 1
        elif np.isclose(pref, 0.0):
            data_rows.append([i, j, 2])  # A wins
            preference_counts[policy_names[i]] += 1
            preference_counts[policy_names[j]] += 1
        else:
            raise ValueError(f"Invalid preference value: {pref}")

    return np.array(data_rows, dtype=float), preference_counts


def combine_policy_progress(progress1, counts1, comparison_counts1,
                            progress2, counts2, comparison_counts2,
                            policy_names):
    """Combine two policy-progress estimates using weighted averaging by evaluation count."""
    total_counts = counts1 + counts2
    total_progress = np.divide(
        progress1 * counts1 + progress2 * counts2,
        total_counts,
        out=np.zeros_like(total_counts, dtype=float),
        where=total_counts > 0,
    )
    comparison_counts = {
        p: comparison_counts1.get(p, 0) + comparison_counts2.get(p, 0)
        for p in policy_names
    }
    return total_progress, total_counts, comparison_counts


# ---------------------------------------------------------------------------
# Roboarena plotting
# ---------------------------------------------------------------------------

def plot_progress(total_progress, total_counts, policy_names, save_dir):
    x = np.arange(len(policy_names))
    width = 0.4

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(x - width / 2, total_progress, width, label="Average Progress")
    ax1.set_ylabel("Average Progress")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, total_counts, width, label="Evaluation Count", alpha=0.7)
    ax2.set_ylabel("Number of Evaluations")

    ax1.set_xticks(x)
    ax1.set_xticklabels(policy_names, rotation=20)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    ax1.set_title("Policy Progress vs Evaluation Count")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "policy_progress_and_counts.png"))


def plot_bt_result(policy_names, total_counts, policy_oracle_performance,
                   bt_scores, pearson_r, comparison_counts, save_dir, metric="progress"):
    idx_ranked = np.argsort(bt_scores)[::-1]

    policy_names_ranked = [
        policy_names[i].replace("_droid", "").replace("paligemma", "pg")
        for i in idx_ranked
    ]
    bt_ranked = bt_scores[idx_ranked]
    oracle_ranked = [policy_oracle_performance[i] for i in idx_ranked]
    counts_ranked = total_counts[idx_ranked]
    comparison_counts_ranked = [comparison_counts[policy_names[i]] for i in idx_ranked]

    x = np.arange(len(policy_names))
    fig, ax1 = plt.subplots(figsize=(11, 5))
    bars = ax1.bar(x, oracle_ranked, width=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [
            f"{name}\nevals={counts_ranked[i]}\npairs={comparison_counts_ranked[i]}"
            for i, name in enumerate(policy_names_ranked)
        ],
        fontsize=12,
    )

    for i, bar in enumerate(bars):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"BT={bt_ranked[i]:.2f}",
            ha="center", va="bottom", fontsize=15, fontweight="bold",
        )

    ax1.set_ylabel("Oracle Progress", fontsize=14)
    ax1.set_title(f"Bradley-Terry Scores ({metric}) vs Oracle Progress", fontsize=16)
    ax1.text(
        0.98, 0.80, f"Pearson r = {pearson_r:.3f}",
        transform=ax1.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round", alpha=0.15), fontsize=14,
    )
    ax1.set_ylim(0, max(oracle_ranked) + 0.08)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"bt_results_{metric}.png"), dpi=200)


# ---------------------------------------------------------------------------
# Roboarena experiment
# ---------------------------------------------------------------------------

ROBOARENA_POLICIES = [
    "paligemma_binning_droid",
    "pi0_droid",
    "paligemma_vq_droid",
    "paligemma_fast_specialist_droid",
    "paligemma_fast_droid",
    "paligemma_diffusion_droid",
    "pi0_fast_droid",
]


def main():
    parser = argparse.ArgumentParser(
        description="Fit a Bradley-Terry model to rank Roboarena policies."
    )
    parser.add_argument(
        "-d", "--data_path",
        type=str,
        default="/n/fs/irom-testing/multitest/data/roboarena/data2/per_trial_progress_data.csv",
        help="Path to per-trial progress CSV.",
    )
    parser.add_argument(
        "-pp", "--pref_path",
        type=str,
        default="/n/fs/irom-testing/multitest/data/roboarena/data2/policy_preferences.csv",
        help="Path to pairwise preference CSV (used when --metric=preference).",
    )
    parser.add_argument(
        "-s", "--save_dir",
        type=str,
        default="/n/fs/irom-testing/multitest/outputs/roboarena/bradley_terry",
        help="Directory to write output figures and arrays.",
    )
    parser.add_argument(
        "-m", "--metric",
        type=str,
        default="preference",
        choices=["progress", "preference"],
        help="Source of pairwise comparisons.",
    )
    parser.add_argument(
        "--no_subselect",
        action="store_true",
        help="Use all policies instead of the default curated subset.",
    )
    parser.add_argument(
        "-t", "--abs_tol",
        type=float,
        default=1e-5,
        help="Convergence tolerance for Bradley-Terry iteration.",
    )
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Load data ---
    policy_names = np.genfromtxt(args.data_path, delimiter=",", dtype=str, max_rows=1)
    progress_data, policy_names, policy_progress, policy_counts, comparison_counts = \
        csv_to_pairwise_data(args.data_path, columns=policy_names, seed=0)

    if args.metric == "progress":
        data = progress_data
    else:
        data, comparison_counts = csv_preferences_to_pairwise_data(args.pref_path, policy_names)

    # --- Optionally restrict to the curated policy subset ---
    if not args.no_subselect:
        selected_policies = ROBOARENA_POLICIES
        data, _, _, _ = subselect_and_remap_policies(selected_policies, policy_names, data)
    else:
        selected_policies = list(policy_names)

    n_teams = len(selected_policies)

    # --- Fit Bradley-Terry model ---
    wins_matrix = build_wins_matrix(data, n_teams)
    print("Wins matrix:")
    print(wins_matrix)
    print()

    bt_scores = fit_bradley_terry(wins_matrix, abs_tol=args.abs_tol)

    # --- Report rankings ---
    idx_ranked = np.argsort(bt_scores)[::-1]
    print("\nBradley-Terry scores (strongest to weakest):")
    for rank, idx in enumerate(idx_ranked):
        print(f"  #{rank + 1}: {selected_policies[idx]}  (BT={bt_scores[idx]:.4f})")

    # --- Evaluate against oracle progress ---
    policy_oracle_performance = [
        policy_progress[list(policy_names).index(p)] for p in selected_policies
    ]
    np.save(
        os.path.join(args.save_dir, "roboarena_policy_performance_oracle_progress.npy"),
        policy_oracle_performance,
    )

    pearson_stat, _ = pearsonr(policy_oracle_performance, bt_scores)
    print(f"\nPearson r (BT score vs oracle progress): {pearson_stat:.4f}")

    ranking_vector = np.zeros(n_teams)
    for i, idx in enumerate(idx_ranked):
        ranking_vector[idx] = n_teams - 1 - i
    silly_pearson_stat, _ = pearsonr(policy_oracle_performance, ranking_vector)
    print(f"Pearson r (rank index vs oracle progress): {silly_pearson_stat:.4f}")

    # --- Plot ---
    plot_bt_result(
        selected_policies,
        policy_counts,
        policy_oracle_performance,
        bt_scores,
        pearson_stat,
        comparison_counts=comparison_counts,
        save_dir=args.save_dir,
        metric=args.metric,
    )


if __name__ == "__main__":
    main()
