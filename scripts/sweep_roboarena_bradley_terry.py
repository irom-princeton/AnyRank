


import numpy as np
from multitest.bradley_terry_davidson import fit_bt_davidson
from multitest.bradley_terry_em import em_hybrid
from multitest.bradley_terry import build_wins_matrix, fit_bradley_terry
import pandas as pd
import os
from roboarena_bradley_terry import load_roboarena_data, save_bt_rankings, subselect_and_remap_prefs, csv_preferences_to_pairwise_data, csv_to_pairwise_data

# Paths:
DATA_PATH = "/n/fs/irom-testing/multitest/data/roboarena/data2/per_trial_progress_data.csv"
PREF_PATH = "/n/fs/irom-testing/multitest/data/roboarena/data2/policy_preferences.csv"
BASE_SAVE_DIR = "/n/fs/irom-testing/multitest/outputs/roboarena/roboarena_bradley_terry"

def roboarena_ranking_em(metric='preference', use_partials=False, subselect_and_remap=True, selected_policies=None, save_dir=BASE_SAVE_DIR, Nmax=None):
    if subselect_and_remap and selected_policies is None:
        selected_policies = ["paligemma_binning_droid", "pi0_droid",
                             "paligemma_diffusion_droid", "pi0_fast_droid"]
    pref_data, progress_data, pref_data_w_scores, policy_names, policy_progress, preference_counts, old_to_new, new_to_old, policy_indices = \
        load_roboarena_data(selected_policies if subselect_and_remap else None, Nmax=Nmax)

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
    return bt_board

if __name__ == "__main__":
    selected_policies = ["paligemma_binning_droid", "pi0_droid",
                        "paligemma_diffusion_droid", "pi0_fast_droid"]

    first_crossover = None
    for Nmax in range(20, 1450, 10):
        try:
            bt_board = roboarena_ranking_em(
                metric='preference',
                use_partials=False,
                subselect_and_remap=True,
                selected_policies=selected_policies,
                save_dir=f"/tmp/em_sweep/Nmax_{Nmax}",
                Nmax=Nmax,
            )
        except Exception as e:
            print(f"Nmax={Nmax}: failed ({e})")
            continue

        if bt_board is None or len(bt_board) == 0:
            continue

        scores = dict(zip(bt_board["policy_name"], bt_board["score"]))
        if "pi0_droid" not in scores or "paligemma_diffusion_droid" not in scores:
            continue

        pi0_score = scores["pi0_droid"]
        pg_diff_score = scores["paligemma_diffusion_droid"]

        crossed = pi0_score > pg_diff_score
        print(f"Nmax={Nmax}: pi0_droid={pi0_score:.4f}, paligemma_diffusion_droid={pg_diff_score:.4f}, pi0>pg_diff={crossed}")

        if crossed and first_crossover is None:
            first_crossover = Nmax
            print(f"--- First crossover at Nmax={Nmax} ---")

    print(f"\nFinal answer: first Nmax where pi0_droid ranks higher than paligemma_diffusion_droid = {first_crossover}")