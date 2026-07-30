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

from binomial_cis import binom_ci

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "This script fits a Bradley-Terry model to rank options."
        )
    )
    # parser.add_argument(
    #     "-p",
    #     "--is_pref",
    #     type=int,
    #     default=0,
    #     help=("Whether the data is already in preference form. If false, the data is in " 
    #           "progress form, and preferences will be assigned by pairwise higher progress. " 
    #           "Defaults to True."),
    # )
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
        "-v",
        "--variance_multiplier",
        type=float,
        default=1.0,
        help=("Variance multiplier of observations. " 
              "Defaults to 1.0."),
    )
    parser.add_argument(
        "-nt",
        "--n_teams",
        type=int,
        default=5,
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
        default=100,
        help=("Number of reruns of the procedure. " 
              "Defaults to 10."),
    )
    parser.add_argument(
        "-nb",
        "--n_bootstrap",
        type=int,
        default=2500,
        help=("Number of bootstrap redraws to get BT score intervals. " 
              "Defaults to 5000."),
    )
    # parser.add_argument(
    #     "-all",
    #     "--use_all_data",
    #     type=bool,
    #     default=False,
    #     help=("Whether to use all available data for Bradley-Terry score computation, or to use a subset of the data from n_games. "),
    # )

    args = parser.parse_args()

    results_dict_base = {}

    n_games_max = args.n_games
    n_games_array = np.arange(100, n_games_max + 5, 100)
    n_teams = args.n_teams
    abs_tol = args.abs_tol
    variance_multiplier = args.variance_multiplier
    n_reruns = args.n_reruns
    n_bootstrap = args.n_bootstrap 
    desired_alpha = args.desired_alpha 

    mu = np.array([0.2, 0.35, 0.45, 0.65, 0.9])

    std = (np.sqrt(mu * (1. - mu))) / np.sqrt(2)
    # print(std)
    # breakpoint()

    make_difficult = False 
    if make_difficult:
        special_idx = (n_teams // 2)
        mu[special_idx] = mu[special_idx-1] + 1e-6

    n_hypotheses = n_teams * (n_teams - 1) // 2

    # Assign parameters of run
    results_dict_base["n_policies"] = n_teams
    results_dict_base["n_redraws"] = n_reruns
    results_dict_base["n_bootstrap"] = n_bootstrap
    results_dict_base["outcome_variance"] = variance_multiplier
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
                idx0, idx1 = np.random.choice(n_teams, 2, replace=False)
                data[i, 0] = idx0
                data[i, 1] = idx1
                r0 = std[idx0] * np.random.standard_normal(1) + mu[idx0]
                r1 = std[idx1] * np.random.standard_normal(1) + mu[idx1]
                if r1 > r0:
                    data[i, 2] = 1.0

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

                ARRAY = np.ones((n_teams, n_teams)) * 1e-2
                for i in range(n_teams):
                    ARRAY[i, i] = 0.0 
                
                for i in range(n_games):
                    if bootstrap_data[i, 2] > 0.75: 
                        ARRAY[int(bootstrap_data[i, 1]), int(bootstrap_data[i, 0])] += 1
                    elif np.isclose(bootstrap_data[i, 2], 0.5):
                        pass
                        # ARRAY[int(bootstrap_data[i, 1]), int(bootstrap_data[i, 0])] += 0.5
                        # ARRAY[int(bootstrap_data[i, 0]), int(bootstrap_data[i, 1])] += 0.5
                    else:
                        ARRAY[int(bootstrap_data[i, 0]), int(bootstrap_data[i, 1])] += 1
        
                # print("Data array of 'wins' / 'losses': ")
                # print()
                # print(ARRAY)

                # Define some optimization initial conditions
                current_tol = 1. 
                p_old = np.ones(n_teams)
                iteration_counts = 0 

                #
                # Run Bradley-Terry updates to convergence 
                #
                while current_tol >= abs_tol and iteration_counts < 500: 
                    # Update to p_new
                    try: 
                        del p_new
                    except:
                        pass 
                
                    # Update p --> p' (Equation 5)
                    p_new = np.zeros(n_teams)
                    for i in range(n_teams):
                        num = 0. 
                        den = 0.
                        for j in range(n_teams):
                            if j != i:
                                num += (ARRAY[i, j]*p_old[j])/(p_old[i] + p_old[j])
                                den += (ARRAY[j, i])/(p_old[i] + p_old[j])
                        try:
                            assert not np.isclose(den, 0.0)
                        except:
                            print(ARRAY)
                            breakpoint()
                        p_new[i] = num / den
                
                    # Normalize p_new by geometric mean (Equation 4)
                    norm_const = (np.prod(p_new))**(1 / n_teams)
                    p_new /= norm_const

                    current_tol = np.linalg.norm(p_new - p_old)
                    p_old = copy.deepcopy(p_new)
                    iteration_counts += 1
                
                BT_TERMS[kk, :] = copy.deepcopy(p_old)

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

        with open(f"outputs/bootstrap_bt_full_metrics/FRF_Heterogeneous/npolicy_{n_teams}_npairs_{n_games}_nredraws_{n_reruns}.json", "w") as file:
            json.dump(results_dict, file, indent=4)
    