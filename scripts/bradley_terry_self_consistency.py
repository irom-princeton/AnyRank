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
        default=500,
        help=("Number of total games played by each teams / policies. " 
              "Defaults to 15."),
    )

    parser.add_argument(
        "-nr",
        "--n_reruns",
        type=int,
        default=10000,
        help=("Number of reruns of the procedure. " 
              "Defaults to 1000."),
    )

    # parser.add_argument(
    #     "-all",
    #     "--use_all_data",
    #     type=bool,
    #     default=False,
    #     help=("Whether to use all available data for Bradley-Terry score computation, or to use a subset of the data from n_games. "),
    # )

    args = parser.parse_args()

    results_dict = {}

    n_games = args.n_games
    n_teams = args.n_teams
    abs_tol = args.abs_tol
    variance_multiplier = args.variance_multiplier
    n_reruns = args.n_reruns

    mu = (np.arange(n_teams) + 0.5) / n_teams
    
    special_idx = (n_teams // 2)
    mu[special_idx] = mu[special_idx-1] + 1e-6


    # Assign parameters of run
    results_dict["n_environments"] = n_games
    results_dict["n_evaluations"] = 2 * n_games 
    results_dict["n_policies"] = n_teams
    results_dict["n_redraws"] = n_reruns
    results_dict["outcome_variance"] = variance_multiplier

    policy_mean_args = []
    for i in range(n_teams):
        policy_mean_args.append(f"policy_{i}_mean")
    
    for i in range(n_teams):
        results_dict[policy_mean_args[i]] = mu[i]

    results_dict["BT_tol"] = abs_tol

    correct_rankings = np.zeros(n_reruns)

    for k in tqdm(range(n_reruns)):
        data = np.zeros((n_games, 3))
        for i in range(n_games):
            idx0, idx1 = np.random.choice(n_teams, 2, replace=False)
            data[i, 0] = idx0
            data[i, 1] = idx1
            r0 = np.sqrt(variance_multiplier) * np.random.standard_normal(1) + mu[idx0]
            r1 = np.sqrt(variance_multiplier) * np.random.standard_normal(1) + mu[idx1]
            if r1 > r0:
                data[i, 2] = 1.0

        ARRAY = np.zeros((n_teams, n_teams))
        for i in range(n_games):
            if data[i, 2] > 0.75: 
                ARRAY[int(data[i, 1]), int(data[i, 0])] += 1
            elif np.isclose(data[i, 2], 0.5):
                pass
                # ARRAY[int(data[i, 1]), int(data[i, 0])] += 0.5
                # ARRAY[int(data[i, 0]), int(data[i, 1])] += 0.5
            else:
                ARRAY[int(data[i, 0]), int(data[i, 1])] += 1
    
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
        while current_tol >= abs_tol: 
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
                p_new[i] = num / den
            
            # Normalize p_new by geometric mean (Equation 4)
            norm_const = (np.prod(p_new))**(1 / n_teams)
            p_new /= norm_const

            current_tol = np.linalg.norm(p_new - p_old)
            p_old = copy.deepcopy(p_new)
            iteration_counts += 1

        if np.min(np.diff(np.argsort(p_old))) > 0.5:
            correct_rankings[k] = 1.0

    print("Finished for loop!")
    print(f"Empirical fraction of consistent rankings: {np.mean(correct_rankings)}")
    print()

    # Print some summary results
    k = int(np.sum(correct_rankings)) # number of successes
    delta = 1e-6 # miscoverage probability

    results_dict["delta"] = delta 
    results_dict["fraction_consistent"] = np.mean(correct_rankings)

    ub = binom_ci(k, n_reruns, delta, 'ub')

    results_dict["best_fraction_consistent"] = ub 
    results_dict["lb_alpha"] = 1.0 - ub - delta

    print(f"High-probability upper bound: {ub}")
    print(f"Lower bound on Type-1 Error control (alpha_min): {1.0 - ub}")

    with open(f"outputs/bt_consistency/npolicy_{n_teams}_npairs_{n_games}_nredraws_{n_reruns}.json", "w") as file:
        json.dump(results_dict, file, indent=4)
    