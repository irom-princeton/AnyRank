"""
Implementation of basic Bradley-Terry Model fitting procedure, as described in 
Equations (4) and (5) at https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model. 

Designed to check self-consistency of procedure at varying numbers of games and varying policy means. 

"""

import numpy as np
import argparse 
import copy
from tqdm import tqdm
import json 
import pandas as pd

from bootstrap_bradley_terry_em_method import em_hybrid

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "This script fits a Bradley-Terry model to rank options."
        )
    )
    parser.add_argument(
        "-a",
        "--desired_alpha",
        type=float,
        default=0.05,
        help=("Desired Type-1 Error Control of bootstrap Bradley-Terry procedure." 
              "Defaults to 0.05."),
    )
    parser.add_argument(
        "-t",
        "--abs_tol",
        type=float,
        default=1e-5,
        help=("Absolute tolerance of Bradley Terry score ranks at termination. " 
              "Defaults to 1e-5."),
    )
    parser.add_argument(
        "-nt",
        "--n_teams",
        type=int,
        default=3,
        help=("Number of teams in the ranking / score computation. " 
              "Defaults to 3."),
    )
    parser.add_argument(
        "-ng",
        "--n_games",
        type=int,
        default=2000,
        help=("Number of total games played by each teams / policies. " 
              "Defaults to 1000."),
    )

    parser.add_argument(
        "-nr",
        "--n_reruns",
        type=int,
        default=50,
        help=("Number of reruns of the procedure. " 
              "Defaults to 100."),
    )
    parser.add_argument(
        "-nb",
        "--n_bootstrap",
        type=int,
        default=500,
        help=("Number of bootstrap redraws to get BT score intervals. " 
              "Defaults to 5000."),
    )

    args = parser.parse_args()

    results_dict_base = {}

    mu = np.array([0.3, 0.7, 0.85])
    std = np.array([(np.sqrt(0.3 * 0.7)), (np.sqrt(0.3 * 0.7)), 0.05])

    n_games_max = args.n_games
    n_games_array = np.arange(100, n_games_max + 5, 300)
    n_teams = args.n_teams
    abs_tol = args.abs_tol
    n_reruns = args.n_reruns
    n_bootstrap = args.n_bootstrap 
    desired_alpha = args.desired_alpha

    n_hypotheses = n_teams * (n_teams - 1) // 2

    # Assign parameters of run
    results_dict_base["n_policies"] = n_teams
    results_dict_base["n_redraws"] = n_reruns
    results_dict_base["n_bootstrap"] = n_bootstrap
    results_dict_base["n_hypotheses"] = n_hypotheses
    results_dict_base["alpha"] = desired_alpha

    policy_mean_args = []
    policy_std_args = []
    for i in range(n_teams):
        policy_mean_args.append(f"policy_{i}_mean")
        policy_std_args.append(f"policy_{i}std")
    
    for i in range(n_teams):
        results_dict_base[policy_mean_args[i]] = mu[i]
        results_dict_base[policy_std_args[i]] = std[i]

    results_dict_base["BT_tol"] = abs_tol

    for jj in tqdm(range(n_games_array.shape[0])):
        try:
            del results_dict, correct_decisions, incorrect_decisions, correct_point_rankings, simple_empirical_ranking_is_correct
            del HYPOTHESIS_INFO, HYPOTHESIS_TRACKER
        except:
            pass
        
        correct_point_rankings = np.zeros(n_reruns)
        correct_decisions = np.zeros(n_reruns)
        incorrect_decisions = np.zeros(n_reruns)

        simple_empirical_ranking_is_correct = np.zeros(n_reruns)

        HYPOTHESIS_INFO = np.zeros((2, n_hypotheses))
        HYPOTHESIS_TRACKER = np.zeros((n_reruns, n_hypotheses))
        populate_hypotheses = True

        results_dict = copy.deepcopy(results_dict_base)
        n_games = int(n_games_array[jj])
        results_dict["n_environments"] = n_games
        results_dict["n_evaluations"] = 2 * n_games

        for k in tqdm(range(n_reruns)):
            data = np.zeros((n_games, 3))
            for i in range(n_games):
                data_t = np.zeros(3)
                data_t[0] = np.random.binomial(1, 0.3, 1) + 0.005 - 0.005 * np.random.rand(1)
                data_t[1] = np.random.binomial(1, 0.7, 1) + 0.005 - 0.005 * np.random.rand(1)
                data_t[2] = 0.8 + 0.1*(np.random.binomial(1, 0.5, 1))
                
                idx0, idx1 = np.random.choice(n_teams, 2, replace=False)
                data[i, 0] = idx0
                data[i, 1] = idx1
                if data_t[idx0] > data_t[idx1]:
                    data[i, 2] = 2

            df_dict_base = {"i": data[:, 0], "j": data[:, 1], "y":data[:, 2]}
            df_base = pd.DataFrame(df_dict_base)

            nominal_data = np.zeros((n_games // n_teams, n_teams))
            for i in range(n_teams):
                nominal_data[:, i] = std[i] * np.random.standard_normal(size=n_games // n_teams) + mu[i]
            
            if np.min(np.diff(np.argsort(np.mean(nominal_data, axis=0)))) > 0.5:
                simple_empirical_ranking_is_correct[k] = 1.0 
            
            # Begin bootstrap procedure
            BT_TERMS = np.zeros((n_bootstrap, n_teams))

            for kk in tqdm(range(n_bootstrap)):
                # Sample indices with replacement
                bootstrap_idx = np.random.choice(n_games, size=n_games, replace=True)

                # Assign data accordingly
                bootstrap_data = copy.deepcopy(data[bootstrap_idx, :])

                df_dict_bootstrap = {"i": bootstrap_data[:, 0], "j": bootstrap_data[:, 1], "y":bootstrap_data[:, 2]}
                df_bootstrap = pd.DataFrame(df_dict_bootstrap)
                
                tmp_output = em_hybrid(df_bootstrap)
                BT_TERMS[kk, :] = copy.deepcopy(np.array(tmp_output['score']))

            # Once bootstrap is done, make intervals for each policy
            POLICY_BT_INTERVALS = np.zeros((2, n_teams))
            low_idx = int(np.floor((desired_alpha / (2 * n_teams)) * n_bootstrap))
            hi_idx = int(np.ceil((1. - (desired_alpha / (2 * n_teams))) * n_bootstrap))
            for kk in range(n_teams):
                tmp_data = np.sort(BT_TERMS[:, kk])
                POLICY_BT_INTERVALS[0, kk] = float(tmp_data[low_idx])
                POLICY_BT_INTERVALS[1, kk] = float(tmp_data[hi_idx])
            
            counter = 0
            for i in range(n_teams-1):
                for j in range(i+1, n_teams):
                    if populate_hypotheses:
                        HYPOTHESIS_INFO[0, counter] = mu[i]
                        HYPOTHESIS_INFO[1, counter] = mu[j]
                    
                    if POLICY_BT_INTERVALS[1, i] < POLICY_BT_INTERVALS[0, j]:
                        correct_decisions[k] += 1.0
                        HYPOTHESIS_TRACKER[k, counter] = 1.0 
                    
                    if POLICY_BT_INTERVALS[0, i] > POLICY_BT_INTERVALS[1, j]:
                        incorrect_decisions[k] = np.maximum(incorrect_decisions[k], 1.0)
                    
                    counter += 1

            if populate_hypotheses:
                populate_hypotheses = False
            
            if np.min(np.diff(np.argsort(np.mean(BT_TERMS, axis=0)))) > 0.5:
                correct_point_rankings[k] = 1.0

        print("Finished for loop!")
        print(f"Empirical fraction of consistent rankings: {np.mean(correct_point_rankings)}")
        print()

        # Print some summary results
        results_dict["fraction_consistent"] = np.mean(correct_point_rankings)
        results_dict["fraction_consistent (comp)"] = np.mean(simple_empirical_ranking_is_correct)
        
        true_values = np.ones(correct_decisions.shape)
        false_values = np.zeros(correct_decisions.shape)
        fully_correct = np.where(correct_decisions >= n_hypotheses, true_values, false_values)
        
        results_dict["fraction_fully_correct"] = np.mean(fully_correct)
        results_dict["mean_fraction_correct_decisions"] = np.mean(correct_decisions) / n_hypotheses
        results_dict["false_positive_rate"] = np.mean(incorrect_decisions)

        for ii in range(n_hypotheses):
            print(f"Hypothesis {ii}, ({HYPOTHESIS_INFO[0, ii]}, {HYPOTHESIS_INFO[1, ii]}): Power = {np.mean(HYPOTHESIS_TRACKER[:, ii])}")
            results_dict[f"hyp_{ii}"] = [HYPOTHESIS_INFO[0, ii], HYPOTHESIS_INFO[1, ii]]
        
        results_dict["power_by_hypothesis"] = np.mean(HYPOTHESIS_TRACKER, axis=0).tolist()

        with open(f"outputs/bootstrap_bt_full_metrics/Inconsistency/npolicy_{n_teams}_npairs_{n_games}_nredraws_{n_reruns}.json", "w") as file:
            json.dump(results_dict, file, indent=4)
    