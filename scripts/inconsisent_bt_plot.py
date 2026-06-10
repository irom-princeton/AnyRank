import numpy as np 
from matplotlib import pyplot as plt 

import json

if __name__ == "__main__":

    # holm_data = np.load("outputs/large_scale_graphical_test_results_v26_full_metrics/hyp_vs_time_N1020_n20_alpha0.05_beta0.0.npy")

    # holm_full_rankings = np.load("outputs/large_scale_graphical_test_results_v26_full_metrics/complete_ranking_vs_time_N1020_n20_alpha0.05_beta0.0.npy")
    
    bt_time = np.arange(1, 21) * 100 

    hyp_rejected_bt = np.zeros(20)
    full_rankings_bt = np.zeros(20)
    bt_fpr = np.zeros(20)

    correct_ordering_bt = np.zeros(20)
    correct_ordering_just_taking_means = np.zeros(20)

    key_idx = []

    for i in range(20):
        idx = int((i+1)*100)
        key_idx.append(idx)

        # Open the file and load its content
        with open(f"outputs/bootstrap_bt_full_metrics/Inconsistency/npolicy_3_npairs_{idx}_nredraws_100.json", 'r') as file:
            data = json.load(file)
        
        hyp_rejected_bt[i] = data["mean_fraction_correct_decisions"]
        full_rankings_bt[i] = data["fraction_fully_correct"]
        correct_ordering_bt[i] = data["fraction_consistent"]
        bt_fpr[i] = data["false_positive_rate"]
        correct_ordering_just_taking_means[i] = data["fraction_consistent (comp)"]
    

    # critical_holm_hypotheses_rejected = np.mean(holm_data, axis=0)[key_idx]
    # critical_holm_full_rankings = holm_full_rankings[key_idx]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(bt_time, correct_ordering_bt, 'r*', markersize=6)
    ax.plot(bt_time, correct_ordering_just_taking_means, 'b*', markersize=6)
    # ax.plot(bt_time, 0.5*np.ones(bt_time.shape[0]-1), 'k--', linewidth=3)
    ax.set_ylim([0., 1.])
    ax.legend(["Bradley-Terry", "Empirical Performance", "50% (Random Guessing)"], fontsize=16)
    ax.set_xlabel("Number of Robot Evaluations", fontsize=20)
    ax.set_ylabel("Probability of Correct Ranking Estimate", fontsize=20)
    ax.set_title("Fraction of Correct Ranking Estimates vs Number of Evals", fontsize=24)
    fig.savefig("outputs/bootstrap_bt_full_metrics/Inconsistency/tmp_correct_ranking_estimates.png", dpi=300)

    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.plot(bt_time, hyp_rejected_bt, 'r--', linewidth=5)
    # ax2.plot(bt_time, critical_holm_hypotheses_rejected, 'b--', linewidth=5)
    ax2.plot(bt_time, full_rankings_bt, 'r', linewidth=5)
    ax2.plot(bt_time, bt_fpr, 'k--', linewidth=5)
    # ax2.plot(bt_time, critical_holm_full_rankings, 'b', linewidth=5)
    # ax2.plot([0, bt_time[-2]], 0.9*np.ones(2), 'k--', linewidth=3)
    ax2.set_ylim([0., 1.])
    ax2.legend(["Hypotheses Rejected (BT)", "Significant Full Ranking (BT)", "FWER (BT)"], fontsize=16)
    ax2.set_xlabel("Number of Robot Evaluations", fontsize=20)
    ax2.set_ylabel("Fraction of Hypotheses Rejected / \n Sig. Full Rankings", fontsize=20)
    ax2.set_title("Fraction of Hypotheses Rejected / \n Sig. Full Rankings vs Number of Evals", fontsize=24)
    fig2.savefig("outputs/bootstrap_bt_full_metrics/Inconsistency/tmp_significant_decisions.png", dpi=300)
    
