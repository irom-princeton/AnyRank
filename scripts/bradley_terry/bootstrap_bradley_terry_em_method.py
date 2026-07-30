import numpy as np
import pandas as pd
from scipy.special import expit

EM_ITERS = 60
HYBRID_NUM_T_BUCKETS = 100
rng = np.random.default_rng(0)


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