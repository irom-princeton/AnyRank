'''
RoboArena Bradley-Terry model fitting.
Source: https://github.com/pranavatreya/roboarena_backend/blob/main/central_server/central_server.py
'''

import numpy as np
import pandas as pd
from scipy.special import expit
import logging
import os
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
# Leaderboard algorithm hyper-params
EXCLUDE = {"PI0", "PI0_FAST"}
HYBRID_NUM_T_BUCKETS = 100
EM_ITERS = 60
NUM_RANDOM_SEEDS = 100

# Paths:
DATA_PATH = "/n/fs/irom-testing/multitest/data/roboarena/data2/per_trial_progress_data.csv"
PREF_PATH = "/n/fs/irom-testing/multitest/data/roboarena/data2/policy_preferences.csv"
SAVE_DIR = "/n/fs/irom-testing/multitest/outputs/roboarena/roboarena_bradley_terry"
os.makedirs(SAVE_DIR, exist_ok=True)
rng = np.random.default_rng(0)

# Legacy hybrid leaderboard model retained for reference only.
# It is intentionally disabled for leaderboard serving.
def em_hybrid(df,
              iters: int = EM_ITERS,
              step_clip: float = 1.0,
              l2_psi: float = 1e-2,
              l2_theta: float = 1e-2,
              step_decay: float = 0.99,
              tol: float = 1e-4,
              n_restarts: int = 1,
              use_partials: bool = False,
              sigma_partial: float = 0.3,
              partial_weight: float = 1.0): # 2.0 if you want to give partials more weight
    """
    EM for independent‐solve hybrid BT, with optional partial‐success signals.
    If use_partials=True, df must contain 'i_partial' and 'j_partial' in [0,1].
    """
    # ——— Precompute indices & masks ———
    pols   = pd.unique(pd.concat([df.i, df.j]))
    idmap  = {p: k for k, p in enumerate(pols)}
    P      = len(pols)
    i_idx  = df.i .map(idmap).to_numpy()
    j_idx  = df.j .map(idmap).to_numpy()
    y      = df.y .to_numpy()
    win    = (y == 2)
    loss   = (y == 0)
    tie    = (y == 1)

    if use_partials:
        s_i_par = df["i_partial"].to_numpy()
        s_j_par = df["j_partial"].to_numpy()

    best_ll, best_board = -np.inf, None

    for restart in range(n_restarts):
        rng.bit_generator.advance(restart * 1000)

        # ——— Initialize parameters ———
        theta = rng.normal(0., .1, P)
        tau = rng.normal(0., .1, HYBRID_NUM_T_BUCKETS)
        psi = np.zeros((P, HYBRID_NUM_T_BUCKETS))
        pi = np.full(HYBRID_NUM_T_BUCKETS, 1 / HYBRID_NUM_T_BUCKETS)
        nu = 0.5

        def clip_step(x, g, h, clip_val):
            if abs(h) < 1e-8:
                return x
            return x - np.clip(g/h, -clip_val, clip_val)

        # ——— EM loop ———
        for it in range(iters):
            curr_clip = step_clip * (step_decay ** it)

            # E-step: compute solve probabilities
            z_i     = theta[i_idx][:,None] + psi[i_idx] - tau
            z_j     = theta[j_idx][:,None] + psi[j_idx] - tau
            solve_i = expit(z_i)
            solve_j = expit(z_j)

            # A/B likelihoods
            p_win  = solve_i * (1 - solve_j)
            p_loss = (1 - solve_i) * solve_j
            p_tie  = 2 * nu * np.sqrt(p_win * p_loss)
            like_ab = (p_win*win[:,None]
                     + p_loss*loss[:,None]
                     + p_tie*tie[:,None])

            # optional partial‐success likelihood
            if use_partials:
                err_i  = (s_i_par[:,None] - solve_i)**2
                err_j  = (s_j_par[:,None] - solve_j)**2
                like_ps = np.exp(-(err_i + err_j)/(2*sigma_partial**2))**partial_weight
                like    = like_ab * like_ps
            else:
                like = like_ab

            # responsibilities gamma[n,t]
            gamma = pi * np.clip(like, 1e-12, None)
            gamma /= gamma.sum(axis=1, keepdims=True)

            # M-step: update theta
            theta_prev = theta.copy()
            for p in range(P):
                mi = (i_idx == p)
                mj = (j_idx == p)
                g = h = 0.0

                for t in range(HYBRID_NUM_T_BUCKETS):
                    # i-slot
                    si   = solve_i[mi, t]
                    sj_i = solve_j[mi, t]
                    gm   = gamma[mi, t]
                    w, l_, tt = win[mi], loss[mi], tie[mi]
                    g  += ((w*(1-sj_i) - l_*sj_i + tt*(sj_i-si)) * gm).sum()
                    h  -= ((si*(1-si) + sj_i*(1-sj_i)) * gm).sum()

                    if use_partials:
                        g  += partial_weight * (((s_i_par[mi]-si)*si*(1-si)) * gm).sum() / sigma_partial**2
                        h  -= partial_weight * (((si*(1-si))**2) * gm).sum() / sigma_partial**2

                    # j-slot
                    si_j = solve_i[mj, t]
                    sj_j = solve_j[mj, t]
                    gmj  = gamma[mj, t]
                    wj, lj, tj = win[mj], loss[mj], tie[mj]
                    g  += ((lj*(1-si_j) - wj*si_j + tj*(si_j-sj_j)) * gmj).sum()
                    h  -= ((si_j*(1-si_j) + sj_j*(1-sj_j)) * gmj).sum()

                    if use_partials:
                        g  += partial_weight * (((s_j_par[mj]-sj_j)*sj_j*(1-sj_j)) * gmj).sum() / sigma_partial**2
                        h  -= partial_weight * (((sj_j*(1-sj_j))**2) * gmj).sum() / sigma_partial**2

                # L2 on theta
                g -= l2_theta * theta[p]
                h -= l2_theta
                theta[p] = clip_step(theta[p], g, h, curr_clip)

            theta -= theta.mean()

            # M-step: update psi
            for p in range(P):
                mi = (i_idx == p)
                mj = (j_idx == p)
                for t in range(HYBRID_NUM_T_BUCKETS):
                    si   = solve_i[mi, t]
                    sj_i = solve_j[mi, t]
                    gm   = gamma[mi, t]
                    si_j = solve_i[mj, t]
                    sj_j = solve_j[mj, t]
                    gmj  = gamma[mj, t]

                    # A/B
                    w, l_, tt   = win[mi], loss[mi], tie[mi]
                    wj, lj, tj  = win[mj], loss[mj], tie[mj]
                    g = ((w*(1-sj_i) - l_*sj_i + tt*(sj_i-si)) * gm).sum() \
                      + ((lj*(1-si_j) - wj*si_j + tj*(si_j-sj_j)) * gmj).sum()
                    h = -(((si*(1-si) + sj_i*(1-sj_i)) * gm).sum()
                         + ((si_j*(1-si_j) + sj_j*(1-sj_j)) * gmj).sum())

                    # partials
                    if use_partials:
                        g  += partial_weight * (((s_i_par[mi]-si)*si*(1-si)) * gm).sum() / sigma_partial**2
                        h  -= partial_weight * (((si*(1-si))**2) * gm).sum() / sigma_partial**2
                        g  += partial_weight * (((s_j_par[mj]-sj_j)*sj_j*(1-sj_j)) * gmj).sum() / sigma_partial**2
                        h  -= partial_weight * (((sj_j*(1-sj_j))**2) * gmj).sum() / sigma_partial**2

                    # L2 on psi
                    g += l2_psi * psi[p, t]
                    h -= l2_psi
                    psi[p, t] = clip_step(psi[p, t], g, h, curr_clip)

            psi -= psi.mean(axis=1, keepdims=True)

            # M-step: update tau
            for t in range(HYBRID_NUM_T_BUCKETS):
                si_t = solve_i[:, t]
                sj_t = solve_j[:, t]
                g    = (gamma[:,t]*(si_t + sj_t - 1.0)).sum()
                h    = - (gamma[:,t]*(si_t*(1-si_t) + sj_t*(1-sj_t))).sum()
                tau[t] = clip_step(tau[t], g, h, curr_clip)
            tau -= tau.mean()

            # update pi, nu
            pi = gamma.mean(axis=0); pi /= pi.sum()
            nu = 0.5 * ((p_tie*gamma).sum() / max((p_win*gamma).sum(), 1e-9))

            if np.max(np.abs(theta - theta_prev)) < tol:
                break

        # finalize restart
        mixlik = (pi * like).sum(axis=1)
        ll_cur = np.sum(np.log(mixlik + 1e-12))
        board  = pd.DataFrame({"policy": pols, "score": theta})\
                     .sort_values("score", ascending=False)\
                     .reset_index(drop=True)
        if ll_cur > best_ll:
            best_ll, best_board = ll_cur, board
    return best_board


def fit_bt_davidson(
    df: pd.DataFrame,
    max_iters: int = 200,
    tol: float = 1e-8,
    hess_ridge: float = 1e-6,
) -> tuple[pd.DataFrame, float]:
    """
    Fit standard Bradley-Terry with Davidson ties.

    Outcome model for policy i vs j:
        p(i > j) = exp(theta_i) / Z
        p(j > i) = exp(theta_j) / Z
        p(tie)   = 2 * nu * exp((theta_i + theta_j)/2) / Z
    where Z is the sum of the three numerators and nu > 0.

    We optimize negative log-likelihood with a Newton method using
    analytic gradient/Hessian, and estimate per-policy standard
    deviations from the inverse observed Hessian.
    """
    pols = pd.unique(pd.concat([df.i, df.j]))
    num_policies = len(pols)
    if num_policies == 0:
        return pd.DataFrame(columns=["policy", "score", "std"]), 0.5
    if num_policies == 1:
        one = pd.DataFrame([{"policy": pols[0], "score": 0.0, "std": 0.0}])
        return one, 0.5

    idmap = {p: k for k, p in enumerate(pols)}
    i_idx = df.i.map(idmap).to_numpy()
    j_idx = df.j.map(idmap).to_numpy()
    y = df.y.to_numpy()

    # Fix one policy ability to 0 for identifiability.
    ref_idx = num_policies - 1
    num_theta_free = num_policies - 1
    phi_idx = num_theta_free  # phi = log(nu)
    num_params = num_theta_free + 1

    params = np.zeros(num_params, dtype=float)
    params[phi_idx] = np.log(0.5)

    def unpack_theta(x: np.ndarray) -> np.ndarray:
        theta = np.zeros(num_policies, dtype=float)
        theta[:num_theta_free] = x[:num_theta_free]
        return theta

    def nll_grad_hess(x: np.ndarray):
        theta = unpack_theta(x)
        nu = np.exp(x[phi_idx])

        nll = 0.0
        grad = np.zeros(num_params, dtype=float)
        hess = np.zeros((num_params, num_params), dtype=float)

        for i, j, outcome in zip(i_idx, j_idx, y):
            ti = theta[i]
            tj = theta[j]

            a = np.exp(ti)
            b = np.exp(tj)
            tie_num = 2.0 * nu * np.exp(0.5 * (ti + tj))
            z = a + b + tie_num

            p_i_win = a / z
            p_j_win = b / z
            p_tie = tie_num / z

            v_i_win = np.zeros(num_params, dtype=float)
            v_j_win = np.zeros(num_params, dtype=float)
            v_tie = np.zeros(num_params, dtype=float)

            if i != ref_idx:
                v_i_win[i] = 1.0
                v_tie[i] += 0.5
            if j != ref_idx:
                v_j_win[j] = 1.0
                v_tie[j] += 0.5
            v_tie[phi_idx] = 1.0

            if outcome == 2:
                p_obs = p_i_win
                v_obs = v_i_win
            elif outcome == 0:
                p_obs = p_j_win
                v_obs = v_j_win
            elif outcome == 1:
                p_obs = p_tie
                v_obs = v_tie
            else:
                raise ValueError(f"Unexpected preference value: {outcome}")

            nll -= np.log(np.clip(p_obs, 1e-12, None))

            v_bar = p_i_win * v_i_win + p_j_win * v_j_win + p_tie * v_tie
            grad -= (v_obs - v_bar)

            second_moment = (
                p_i_win * np.outer(v_i_win, v_i_win)
                + p_j_win * np.outer(v_j_win, v_j_win)
                + p_tie * np.outer(v_tie, v_tie)
            )
            hess += second_moment - np.outer(v_bar, v_bar)
        return nll, grad, hess

    for _ in range(max_iters):
        nll, grad, hess = nll_grad_hess(params)
        grad_inf = float(np.linalg.norm(grad, ord=np.inf))
        if grad_inf < tol:
            break

        hess_reg = hess + hess_ridge * np.eye(num_params)
        try:
            step = np.linalg.solve(hess_reg, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess_reg, grad, rcond=None)[0]

        if not np.all(np.isfinite(step)):
            logger.warning("BT-Davidson Newton step became non-finite; stopping early.")
            break

        step_inf = float(np.linalg.norm(step, ord=np.inf))
        if step_inf < tol:
            break

        directional = float(grad @ step)
        alpha = 1.0
        accepted = False
        while alpha >= 1e-8:
            cand = params - alpha * step
            cand[phi_idx] = float(np.clip(cand[phi_idx], -10.0, 10.0))
            cand_nll, _, _ = nll_grad_hess(cand)
            if cand_nll <= nll - 1e-4 * alpha * directional:
                params = cand
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            logger.warning("BT-Davidson line search failed; stopping early.")
            break

        if alpha * step_inf < tol:
            break

    _, _, final_hess = nll_grad_hess(params)
    hess_reg = final_hess + hess_ridge * np.eye(num_params)
    cov_params = np.linalg.pinv(hess_reg, rcond=1e-12)

    theta = unpack_theta(params)
    theta -= theta.mean()

    cov_theta_ref = np.zeros((num_policies, num_policies), dtype=float)
    cov_theta_ref[:num_theta_free, :num_theta_free] = cov_params[
        :num_theta_free, :num_theta_free
    ]

    center = np.eye(num_policies) - (1.0 / num_policies) * np.ones(
        (num_policies, num_policies)
    )
    cov_theta_centered = center @ cov_theta_ref @ center.T
    theta_std = np.sqrt(np.clip(np.diag(cov_theta_centered), 0.0, None))
    theta_std = np.nan_to_num(theta_std, nan=0.0, posinf=0.0, neginf=0.0)

    board = (
        pd.DataFrame({"policy": pols, "score": theta, "std": theta_std})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    tie_nu = float(np.exp(params[phi_idx]))
    return board, tie_nu


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