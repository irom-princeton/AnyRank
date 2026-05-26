import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)



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