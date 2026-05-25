"""
Implementation of basic Bradley-Terry Model fitting procedure, as described in 
Equations (4) and (5) at https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model

"""

import numpy as np
import os 
import argparse 
import copy
from scipy.stats import pearsonr
from pathlib import Path

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "This script fits a Bradley-Terry model to rank options."
        )
    )
    parser.add_argument(
        "-p",
        "--is_pref",
        type=int,
        default=0,
        help=("Whether the data is already in preference form. If false, the data is in " 
              "progress form, and preferences will be assigned by pairwise higher progress. " 
              "Defaults to True."),
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
        default=4,
        help=("Number of teams in the ranking / score computation. " 
              "Defaults to 3."),
    )
    
    parser.add_argument(
        "-ng",
        "--n_games",
        type=int,
        default=15,
        help=("Number of total games played by each teams / policies. " 
              "Defaults to 15."),
    )

    parser.add_argument(
        "-all",
        "--use_all_data",
        type=bool,
        default=False,
        help=("Whether to use all available data for Bradley-Terry score computation, or to use a subset of the data from n_games. "),
    )

    current_file_path = Path(__file__).resolve()
    parent_dir = current_file_path.parent.parent
    data_dir = parent_dir / "data"
    data_file_path = "/n/fs/irom-testing/multitest/data/roboarena/data1/Roboarena_progress.npy"
    pref_file_path = "/n/fs/irom-testing/multitest/data/roboarena/data1/Roboarena_preference.npy"
    pref_data = np.load(pref_file_path)
    data_progress = np.load(data_file_path)
    
    parser.add_argument(
        "-dp",
        "--path",
        type=str,
        default=data_file_path,
        help=("Path to the data to load. " 
              "Defaults to None."),
    )
    parser.add_argument(
        "-it",
        "--is_test",
        type=int,
        default=0,
        help=("Specify if the run is meant for debugging / test purposes. " 
              "Defaults to False."),
    )
    args = parser.parse_args()

    n_games = args.n_games
    n_teams = args.n_teams
    is_pref = bool(args.is_pref)
    is_test = bool(args.is_test)
    abs_tol = args.abs_tol
    data_path = args.path

    policy_names = ["pi0_droid",
                    "pi0_fast_droid",
                    "paligemma_diffusion_droid",
                    "paligemma_binning_droid",
                    ]
    
    if is_test:
        n_games = args.n_games
        mu = (np.arange(n_teams) + 0.5) / n_teams 
        data = np.zeros((n_games, 3))
        for i in range(n_games):
            idx0, idx1 = np.random.choice(n_teams, 2, replace=False)
            data[i, 0] = idx0
            data[i, 1] = idx1
            r0 = np.random.standard_normal(1) + mu[idx0]
            r1 = np.random.standard_normal(1) + mu[idx1]
            if r1 > r0:
                data[i, 2] = 1.0 
    else:
        # Use load path
        if is_pref:
            try:
                data = np.load(data_path)
                if args.use_all_data:
                    n_games = data.shape[0]
                
                data = data[:n_games, :]
                assert data.shape[1] == 3
                
                current_file_path = Path(__file__).resolve()
                parent_dir = current_file_path.parent.parent
                data_dir = parent_dir / "data"
                data_file_path = data_dir / "roboarena/roboarena_policy_performance_oracle_progress.npy"
                policy_oracle_performance = np.load(data_file_path)
                n_teams = policy_oracle_performance.shape[0]
            except:
                raise ValueError("Cannot find data at specified path. Please verify the data location and try again")
        else:
            try:
                data_progress = np.load(data_file_path)
                print(data_progress.shape)
                # breakpoint()
            except:
                print(data_file_path)
                raise ValueError("Cannot find data at specified path. Please verify the data location and try again")
            breakpoint()
            # Assume data_progress is in form PROGRESS x [IDX0, IDX1, ..., IDX{N-1}]
            # the list of strings to compare; load those vectors and concatenate
            policies = [0,3,5,6]  
            data_progress = data_progress[:, policies]
            N0 = data_progress.shape[0]
            n_games = int(N0 * n_teams * (n_teams - 1) / 2)

            # Use every available pairwise preference for now
            data = np.zeros((n_games, 3))
            counter = 0
            for k in range(N0):
                for i in range(n_teams - 1):
                    for j in range(i+1, n_teams):
                        data[counter, 0] = i 
                        data[counter, 1] = j
                        if np.isclose(data_progress[k, j], data_progress[k, i]):
                            data[counter, 2] = np.random.binomial(1, 0.5, 1)
                        elif data_progress[k, j] > data_progress[k, i]:
                            data[counter, 2] = 1.0
                        else: # Remain 0
                            pass
                        
                        counter += 1

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
    
    print("Data array of 'wins' / 'losses': ")
    print()
    print(ARRAY)

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
    
    # Print some summary results
    print()
    print(f"Terminated in {iteration_counts} iterations!")
    print()
    print("Power scores: ")
    print(p_old)
    print()
    print("Ranking (ascending): ")
    print(np.argsort(p_old))
    print()
    print("Ranking (strongest to weakest): ")
    print(np.argsort(p_old)[::-1])

    idx_ranked = np.argsort(p_old)[::-1]
    print()
    print("Power scores (descending): ")
    print(p_old[idx_ranked])

    # policy_names = ["pi0_droid",
    #                 "paligemma_vq_droid",
    #                 "paligemma_fast_specialist_droid",
    #                 "pi0_fast_droid",
    #                 "paligemma_fast_droid",
    #                 "paligemma_diffusion_droid",
    #                 "paligemma_binning_droid",
    #                 ]
    
    print()
    print("Policy name ranking: ")
    for i in range(n_teams):
        print(f"Rank #{i+1}: ", policy_names[int(idx_ranked[i])])

    if is_pref:
        pass
    else:
        policy_oracle_performance = np.mean(data_progress, axis=0)
        np.save("roboarena_policy_performance_oracle_progress.npy", policy_oracle_performance)
    
    pearson_stat, pearson_pvalue = pearsonr(policy_oracle_performance, p_old)
    print("Pearson R^2: ", pearson_stat)

    ranking_vector = np.zeros(n_teams)
    for i in range(n_teams):
        ranking_vector[idx_ranked[n_teams-1-i]] = i 
    
    print(ranking_vector)
    silly_pearson_stat, silly_pearson_pvalue = pearsonr(policy_oracle_performance, ranking_vector)
    print("Silly Pearson R^2: ", silly_pearson_stat)