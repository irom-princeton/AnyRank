"""
Unified policy ranking via Bradley-Terry models.

Supports three fitting methods:
  classic   – iterative MM algorithm (Bradley-Terry v2)
  davidson  – BT with Davidson tie model (Newton's method)
  em        – EM hybrid (independent-solve model, RoboArena leaderboard algorithm)

Outcome encoding used throughout: 2 = i wins, 1 = tie, 0 = j wins.

Usage:
    python plot_policy_rankings.py \\
        --data_path /path/to/per_trial_progress_data.csv \\
        --pref_path /path/to/policy_preferences.csv \\
        --save_dir  /path/to/outputs \\
        --metric    preference \\
        --method    davidson
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import pearsonr


DEFAULT_POLICIES = [
    "paligemma_binning_droid",
    "pi0_droid",
    "paligemma_vq_droid",
    "paligemma_fast_specialist_droid",
    "paligemma_fast_droid",
    "paligemma_diffusion_droid",
    "pi0_fast_droid",
]

_EM_T_BUCKETS = 100
_EM_ITERS = 60


# ── Data loading ───────────────────────────────────────────────────────────────

def csv_to_pairwise_data(csv_path, columns=None, seed=None):
    """
    Convert a per-trial progress CSV to pairwise outcomes.
    Returns (data, policy_names, policy_progress, policy_counts, comparison_counts).
    """
    df = pd.read_csv(csv_path)
    df = df[df.notna().sum(axis=1) > 1]
    if columns is not None:
        df = df[columns]

    policy_progress = df.mean(axis=0, skipna=True)
    policy_counts = df.notna().sum(axis=0).to_numpy()
    comparison_counts = {p: 0 for p in df.columns}
    rows = []

    for _, row in df.iterrows():
        observed = np.where(row.notna().to_numpy())[0]
        for a in range(len(observed) - 1):
            for b in range(a + 1, len(observed)):
                i, j = observed[a], observed[b]
                xi, xj = row.iloc[i], row.iloc[j]
                if np.isclose(xi, xj):
                    outcome = 1
                elif xi > xj:
                    outcome = 2
                else:
                    outcome = 0
                comparison_counts[df.columns[i]] += 1
                comparison_counts[df.columns[j]] += 1
                rows.append([i, j, outcome])

    return np.asarray(rows, dtype=float), list(df.columns), policy_progress, policy_counts, comparison_counts


def csv_preferences_to_pairwise_data(csv_path, policy_names, ignore_ties=True, has_header=True):
    """
    Convert a preference CSV to pairwise outcomes.
    preference=0.0 → A preferred (outcome 2), 0.5 → tie (outcome 1), 1.0 → B preferred (outcome 0).
    Returns (data, preference_counts).
    """
    df = (
        pd.read_csv(csv_path)
        if has_header
        else pd.read_csv(csv_path, header=None, names=["A", "B", "preference"])
    )
    policy_to_idx = {p: i for i, p in enumerate(policy_names)}
    preference_counts = {p: 0 for p in policy_names}
    rows = []

    for _, row in df.iterrows():
        a, b, pref = row["A"], row["B"], float(row["preference"])
        if a not in policy_to_idx or b not in policy_to_idx:
            continue
        i, j = policy_to_idx[a], policy_to_idx[b]
        if np.isclose(pref, 0.5):
            if ignore_ties:
                continue
            outcome = 1
        elif np.isclose(pref, 0.0):
            outcome = 2
        elif np.isclose(pref, 1.0):
            outcome = 0
        else:
            raise ValueError(f"Invalid preference value: {pref}")
        preference_counts[policy_names[i]] += 1
        preference_counts[policy_names[j]] += 1
        rows.append([i, j, outcome])

    return np.array(rows, dtype=float), preference_counts


def subselect_and_remap(selected_policies, policy_names, data):
    """
    Filter data to selected_policies and remap indices to 0..N-1.
    Returns (filtered_data, old_to_new, new_to_old).
    """
    name_to_old = {name: i for i, name in enumerate(policy_names)}
    old_to_new = {name_to_old[name]: new for new, name in enumerate(selected_policies)}
    new_to_old = {v: k for k, v in old_to_new.items()}
    selected_old = set(old_to_new.keys())

    rows = [
        [old_to_new[int(r[0])], old_to_new[int(r[1])], r[2]]
        for r in data
        if int(r[0]) in selected_old and int(r[1]) in selected_old
    ]
    return np.asarray(rows, dtype=float), old_to_new, new_to_old


# ── BT fitting algorithms ──────────────────────────────────────────────────────

def fit_bt_classic(data, n_teams, abs_tol=1e-5):
    """
    Classic iterative MM (Bradley-Terry) algorithm.
    Returns score array indexed 0..n_teams-1.
    """
    wins = np.zeros((n_teams, n_teams))
    for row in data:
        i, j, outcome = int(row[0]), int(row[1]), int(row[2])
        if outcome == 2:
            wins[i, j] += 1
        elif outcome == 0:
            wins[j, i] += 1

    p = np.ones(n_teams)
    tol = 1.0
    while tol >= abs_tol:
        p_new = np.array([
            sum(wins[i, j] * p[j] / (p[i] + p[j]) for j in range(n_teams) if j != i)
            / max(sum(wins[j, i] / (p[i] + p[j]) for j in range(n_teams) if j != i), 1e-12)
            for i in range(n_teams)
        ])
        p_new /= np.prod(p_new) ** (1 / n_teams)
        tol = np.linalg.norm(p_new - p)
        p = p_new
    return p


def fit_bt_davidson(data, n_teams, max_iters=200, tol=1e-8, hess_ridge=1e-6):
    """
    BT with Davidson tie model fitted via Newton's method.
    Returns (scores, stds, tie_nu) indexed 0..n_teams-1.
    """
    i_idx = data[:, 0].astype(int)
    j_idx = data[:, 1].astype(int)
    y = data[:, 2]

    ref = n_teams - 1
    n_free = n_teams - 1
    phi = n_free
    n_params = n_free + 1

    params = np.zeros(n_params)
    params[phi] = np.log(0.5)

    def unpack(x):
        theta = np.zeros(n_teams)
        theta[:n_free] = x[:n_free]
        return theta

    def nll_grad_hess(x):
        theta = unpack(x)
        nu = np.exp(x[phi])
        nll, grad, hess = 0.0, np.zeros(n_params), np.zeros((n_params, n_params))

        for ii, jj, outcome in zip(i_idx, j_idx, y):
            a = np.exp(theta[ii])
            b = np.exp(theta[jj])
            c = 2.0 * nu * np.exp(0.5 * (theta[ii] + theta[jj]))
            z = a + b + c

            pw, pl, pt = a / z, b / z, c / z
            vw = np.zeros(n_params)
            vl = np.zeros(n_params)
            vt = np.zeros(n_params)
            if ii != ref:
                vw[ii] = 1.0
                vt[ii] += 0.5
            if jj != ref:
                vl[jj] = 1.0
                vt[jj] += 0.5
            vt[phi] = 1.0

            if outcome == 2:
                p_obs, v_obs = pw, vw
            elif outcome == 0:
                p_obs, v_obs = pl, vl
            else:
                p_obs, v_obs = pt, vt

            nll -= np.log(np.clip(p_obs, 1e-12, None))
            v_bar = pw * vw + pl * vl + pt * vt
            grad -= v_obs - v_bar
            hess += (
                pw * np.outer(vw, vw)
                + pl * np.outer(vl, vl)
                + pt * np.outer(vt, vt)
                - np.outer(v_bar, v_bar)
            )

        return nll, grad, hess

    for _ in range(max_iters):
        nll, grad, hess = nll_grad_hess(params)
        if np.linalg.norm(grad, ord=np.inf) < tol:
            break
        hess_reg = hess + hess_ridge * np.eye(n_params)
        try:
            step = np.linalg.solve(hess_reg, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess_reg, grad, rcond=None)[0]
        if not np.all(np.isfinite(step)) or np.linalg.norm(step, ord=np.inf) < tol:
            break
        directional = float(grad @ step)
        alpha, accepted = 1.0, False
        while alpha >= 1e-8:
            cand = params - alpha * step
            cand[phi] = float(np.clip(cand[phi], -10.0, 10.0))
            if nll_grad_hess(cand)[0] <= nll - 1e-4 * alpha * directional:
                params, accepted = cand, True
                break
            alpha *= 0.5
        if not accepted or alpha * np.linalg.norm(step, ord=np.inf) < tol:
            break

    _, _, final_hess = nll_grad_hess(params)
    hess_reg = final_hess + hess_ridge * np.eye(n_params)
    cov = np.linalg.pinv(hess_reg, rcond=1e-12)

    theta = unpack(params)
    theta -= theta.mean()

    cov_ref = np.zeros((n_teams, n_teams))
    cov_ref[:n_free, :n_free] = cov[:n_free, :n_free]
    center = np.eye(n_teams) - np.ones((n_teams, n_teams)) / n_teams
    stds = np.sqrt(np.clip(np.diag(center @ cov_ref @ center.T), 0.0, None))
    stds = np.nan_to_num(stds)

    return theta, stds, float(np.exp(params[phi]))


def fit_bt_em(data, n_teams, iters=_EM_ITERS, n_t=_EM_T_BUCKETS,
              step_clip=1.0, l2_psi=1e-2, l2_theta=1e-2,
              step_decay=0.99, tol=1e-4, n_restarts=1):
    """
    EM for independent-solve hybrid BT (RoboArena leaderboard algorithm).
    Returns score array indexed 0..n_teams-1.
    """
    df = pd.DataFrame(data, columns=["i", "j", "y"])
    pols = pd.unique(pd.concat([df.i, df.j]))
    idmap = {p: k for k, p in enumerate(pols)}
    P = len(pols)
    ii = df.i.map(idmap).to_numpy()
    jj = df.j.map(idmap).to_numpy()
    y = df.y.to_numpy()
    win = (y == 2)
    loss = (y == 0)
    tie = (y == 1)
    rng = np.random.default_rng(0)

    def clip_step(x, g, h, clip_val):
        return x if abs(h) < 1e-8 else x - np.clip(g / h, -clip_val, clip_val)

    best_ll, best_theta = -np.inf, np.zeros(P)

    for restart in range(n_restarts):
        rng.bit_generator.advance(restart * 1000)
        theta = rng.normal(0.0, 0.1, P)
        tau = rng.normal(0.0, 0.1, n_t)
        psi = np.zeros((P, n_t))
        pi = np.full(n_t, 1.0 / n_t)
        nu = 0.5

        for it in range(iters):
            clip = step_clip * (step_decay ** it)
            z_i = theta[ii, None] + psi[ii] - tau
            z_j = theta[jj, None] + psi[jj] - tau
            si, sj = expit(z_i), expit(z_j)

            p_win = si * (1 - sj)
            p_loss = (1 - si) * sj
            p_tie = 2 * nu * np.sqrt(p_win * p_loss)
            like = p_win * win[:, None] + p_loss * loss[:, None] + p_tie * tie[:, None]

            gamma = pi * np.clip(like, 1e-12, None)
            gamma /= gamma.sum(axis=1, keepdims=True)

            theta_prev = theta.copy()
            for p in range(P):
                mi, mj = (ii == p), (jj == p)
                g = h = 0.0
                for t in range(n_t):
                    s_i, s_ji = si[mi, t], sj[mi, t]
                    gm = gamma[mi, t]
                    w, l_, tt = win[mi], loss[mi], tie[mi]
                    g += ((w * (1 - s_ji) - l_ * s_ji + tt * (s_ji - s_i)) * gm).sum()
                    h -= ((s_i * (1 - s_i) + s_ji * (1 - s_ji)) * gm).sum()
                    s_ij, s_j = si[mj, t], sj[mj, t]
                    gmj = gamma[mj, t]
                    wj, lj, tj = win[mj], loss[mj], tie[mj]
                    g += ((lj * (1 - s_ij) - wj * s_ij + tj * (s_ij - s_j)) * gmj).sum()
                    h -= ((s_ij * (1 - s_ij) + s_j * (1 - s_j)) * gmj).sum()
                g -= l2_theta * theta[p]
                h -= l2_theta
                theta[p] = clip_step(theta[p], g, h, clip)
            theta -= theta.mean()

            for p in range(P):
                mi, mj = (ii == p), (jj == p)
                for t in range(n_t):
                    s_i, s_ji = si[mi, t], sj[mi, t]
                    s_ij, s_j = si[mj, t], sj[mj, t]
                    gm, gmj = gamma[mi, t], gamma[mj, t]
                    w, l_, tt = win[mi], loss[mi], tie[mi]
                    wj, lj, tj = win[mj], loss[mj], tie[mj]
                    g = (
                        ((w * (1 - s_ji) - l_ * s_ji + tt * (s_ji - s_i)) * gm).sum()
                        + ((lj * (1 - s_ij) - wj * s_ij + tj * (s_ij - s_j)) * gmj).sum()
                    )
                    h = -(
                        ((s_i * (1 - s_i) + s_ji * (1 - s_ji)) * gm).sum()
                        + ((s_ij * (1 - s_ij) + s_j * (1 - s_j)) * gmj).sum()
                    )
                    g += l2_psi * psi[p, t]
                    h -= l2_psi
                    psi[p, t] = clip_step(psi[p, t], g, h, clip)
            psi -= psi.mean(axis=1, keepdims=True)

            for t in range(n_t):
                si_t, sj_t = si[:, t], sj[:, t]
                g = (gamma[:, t] * (si_t + sj_t - 1.0)).sum()
                h = -(gamma[:, t] * (si_t * (1 - si_t) + sj_t * (1 - sj_t))).sum()
                tau[t] = clip_step(tau[t], g, h, clip)
            tau -= tau.mean()

            pi = gamma.mean(axis=0)
            pi /= pi.sum()
            nu = 0.5 * ((p_tie * gamma).sum() / max((p_win * gamma).sum(), 1e-9))

            if np.max(np.abs(theta - theta_prev)) < tol:
                break

        ll = np.sum(np.log((pi * like).sum(axis=1) + 1e-12))
        if ll > best_ll:
            best_ll, best_theta = ll, theta.copy()

    scores = np.zeros(n_teams)
    for orig_idx, local_idx in idmap.items():
        scores[int(orig_idx)] = best_theta[local_idx]
    return scores


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_rankings(
    policy_names,
    scores,
    oracle_progress,
    eval_counts,
    save_dir,
    metric="preference",
    method="davidson",
    stds=None,
):
    """
    Bar chart of oracle progress ranked by BT score, with BT score annotations.

    Args:
        policy_names:     list of policy names indexed 0..N-1
        scores:           BT scores, numpy array indexed 0..N-1
        oracle_progress:  oracle progress values, numpy array indexed 0..N-1
        eval_counts:      dict mapping policy index → eval count
        stds:             optional score standard deviations, numpy array indexed 0..N-1
    """
    idx = np.argsort(scores)[::-1]
    labels = [policy_names[i].replace("_droid", "").replace("paligemma", "pg") for i in idx]
    oracle_vals = oracle_progress[idx]
    score_vals = scores[idx]
    std_vals = stds[idx] if stds is not None else None
    count_vals = [eval_counts.get(i, 0) for i in idx]

    x = np.arange(len(policy_names))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, oracle_vals, width=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{name}\nevals={count_vals[i]}" for i, name in enumerate(labels)],
        fontsize=12,
    )

    for i, bar in enumerate(ax.patches):
        height = bar.get_height()
        annotation = f"BT={score_vals[i]:.3f}"
        if std_vals is not None:
            annotation += f"\n±{std_vals[i]:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            annotation,
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    pearson_r, _ = pearsonr(oracle_progress, scores)
    ax.text(
        0.98, 0.80,
        f"Pearson r = {pearson_r:.3f}",
        transform=ax.transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round", alpha=0.15),
        fontsize=14,
    )

    ax.set_ylabel("Oracle Progress", fontsize=14)
    ax.set_title(f"Bradley-Terry ({method}, {metric}) vs Oracle Progress", fontsize=16)
    ax.set_ylim(0, float(oracle_vals.max()) + 0.15)
    plt.tight_layout()

    fname = os.path.join(save_dir, f"bt_rankings_{method}_{metric}.png")
    plt.savefig(fname, dpi=200)
    print(f"Saved: {fname}")
    return fig


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rank policies with a Bradley-Terry model and plot results."
    )
    parser.add_argument("--data_path", required=True,
                        help="CSV of per-trial progress data (rows=trials, cols=policies)")
    parser.add_argument("--pref_path", default=None,
                        help="CSV of pairwise preferences (required for --metric=preference)")
    parser.add_argument("--save_dir", required=True,
                        help="Directory where output plots are written")
    parser.add_argument("--metric", choices=["progress", "preference"], default="preference",
                        help="Whether to fit BT on progress-derived pairs or human preferences")
    parser.add_argument("--method", choices=["classic", "davidson", "em"], default="davidson",
                        help="BT fitting algorithm to use")
    parser.add_argument("--policies", nargs="+", default=None,
                        help="Subset of policy names to rank (default: DEFAULT_POLICIES)")
    parser.add_argument("--abs_tol", type=float, default=1e-5,
                        help="Convergence tolerance for classic BT")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    all_policy_names = np.genfromtxt(
        args.data_path, delimiter=",", dtype=str, max_rows=1
    ).tolist()

    data, policy_names, policy_progress, policy_counts, comparison_counts = csv_to_pairwise_data(
        args.data_path, columns=all_policy_names, seed=0
    )

    if args.metric == "preference":
        if args.pref_path is None:
            raise ValueError("--pref_path is required when --metric=preference")
        data, pref_counts = csv_preferences_to_pairwise_data(
            args.pref_path, policy_names, ignore_ties=False
        )

    selected = args.policies if args.policies is not None else DEFAULT_POLICIES
    selected = [p for p in selected if p in policy_names]
    if not selected:
        raise ValueError("None of the specified policies were found in the data.")
    print(f"Ranking {len(selected)} policies with method={args.method}, metric={args.metric}")

    data, old_to_new, new_to_old = subselect_and_remap(selected, policy_names, data)
    n_teams = len(selected)

    oracle_progress = np.array([policy_progress[selected[i]] for i in range(n_teams)])

    df_tmp = pd.DataFrame(data, columns=["i", "j", "y"])
    counts_i = df_tmp["i"].value_counts()
    counts_j = df_tmp["j"].value_counts()
    eval_counts = {
        int(k): v
        for k, v in counts_i.add(counts_j, fill_value=0).astype(int).to_dict().items()
    }

    stds = None
    if args.method == "classic":
        scores = fit_bt_classic(data, n_teams, abs_tol=args.abs_tol)
    elif args.method == "davidson":
        scores, stds, tie_nu = fit_bt_davidson(data, n_teams)
        print(f"BT-Davidson converged: tie_nu={tie_nu:.4f}")
    elif args.method == "em":
        scores = fit_bt_em(data, n_teams)

    idx_ranked = np.argsort(scores)[::-1]
    print("\nPolicy ranking (strongest to weakest):")
    for rank, i in enumerate(idx_ranked):
        std_str = f"  ±{stds[i]:.4f}" if stds is not None else ""
        print(f"  #{rank + 1}: {selected[i]:<45s}  BT={scores[i]:.4f}{std_str}")

    pearson_r, _ = pearsonr(oracle_progress, scores)
    print(f"\nPearson r (BT score vs oracle progress): {pearson_r:.4f}")

    plot_rankings(
        selected, scores, oracle_progress, eval_counts,
        args.save_dir, metric=args.metric, method=args.method, stds=stds,
    )


if __name__ == "__main__":
    main()
