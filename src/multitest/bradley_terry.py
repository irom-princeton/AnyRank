"""
Generic Bradley-Terry model fitting utilities, as described in
Equations (4) and (5) at https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model
"""

import numpy as np


def build_wins_matrix(data, n_teams):
    """
    Convert pairwise outcome data into a wins matrix.

    Parameters
    ----------
    data : np.ndarray, shape (N, 3)
        Each row is [i, j, outcome]:
          outcome == 2  -> i beat j
          outcome == 0  -> j beat i
          outcome == 1  -> tie, ignored
    n_teams : int

    Returns
    -------
    wins : np.ndarray, shape (n_teams, n_teams)
        wins[i, j] = number of times i beat j
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

    Parameters
    ----------
    wins_matrix : np.ndarray, shape (n, n)
        wins_matrix[i, j] = number of times i beat j
    abs_tol : float
        Convergence tolerance on the L2 norm of the score update.

    Returns
    -------
    p : np.ndarray, shape (n,)
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
    """
    Filter pairwise data to a subset of policies and remap indices to 0..k-1.

    Parameters
    ----------
    selected_policies : list[str]
        Ordered subset of policy names to keep.
    policy_names : list[str]
        Full ordered list of policy names corresponding to current data indices.
    data : np.ndarray, shape (N, 3)
        Pairwise data with columns [i, j, outcome] using old indices.

    Returns
    -------
    filtered_data : np.ndarray, shape (M, 3)
        Pairwise data with remapped indices.
    old_to_new : dict[int, int]
    new_to_old : dict[int, int]
    counts : dict[int, int]
        Number of comparisons each new-index policy appears in.
    """
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
