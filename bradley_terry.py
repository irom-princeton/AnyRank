"""
Implementation of basic Bradley-Terry Model fitting procedure, as described in 
Equations (4) and (5) at https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model

"""

import numpy as np
import os 
import argparse 
import copy

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "This script fits a Bradley-Terry model to rank options."
        )
    )
    parser.add_argument(
        "-p",
        "--is_pref",
        type=bool,
        default=True,
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
        default=3,
        help=("Number of teams in the ranking / score computation. " 
              "Defaults to 3."),
    )
    parser.add_argument(
        "-ng",
        "--n_games",
        type=int,
        default=15,
        help=("Number of total games played by all teams / policies. " 
              "Defaults to 15."),
    )
    parser.add_argument(
        "-dp",
        "--path",
        type=str,
        default=None,
        help=("Path to the data to load. " 
              "Defaults to None."),
    )
    parser.add_argument(
        "-it",
        "--is_test",
        type=bool,
        default=False,
        help=("Specify if the run is meant for debugging / test purposes. " 
              "Defaults to False."),
    )
    args = parser.parse_args()

    n_teams = args.n_teams
    is_pref = args.is_pref 
    is_test = args.is_test
    abs_tol = args.abs_tol
    data_path = args.path
    
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
                n_games = data.shape[0]
                assert data.shape[1] == 3
                # Assume data is in form n_games x [IDX0, IDX1, PREF]
            except:
                raise ValueError("Cannot find data at specified path. Please verify the data location and try again")
        else:
            try:
                data_progress = np.load(data_path)
            except:
                raise ValueError("Cannot find data at specified path. Please verify the data location and try again")
            
            # Assume data_progress is in form PROGRESS x [IDX0, IDX1, IDX{N-1}]
            assert data_progress.shape[1] == n_teams
            N0 = data_progress.shape[0]
            n_games = N0 * n_teams * (n_teams - 1) / 2
            # Use every available pairwise preference for now
            data = np.zeros((n_games, 3))
            counter = 0
            for k in range(N0):
                for i in range(n_teams - 1):
                    for j in range(i, n_teams):
                        data[counter, 0] = i 
                        data[counter, 1] = j 
                        if data_progress[counter, j] >= data_progress[counter, i]:
                            data[counter, 2] = 1.0 
                            # else, remains 0
                        
                        counter += 1
        
        # data = LOAD(...)
        # Assume that data is array of results
        # IDX0, IDX1, {0 or 1}
        # 0 if IDX0 wins

    ARRAY = np.zeros((n_teams, n_teams))
    for i in range(n_games):
        if data[i, 2] > 0.5: 
            ARRAY[int(data[i, 1]), int(data[i, 0])] += 1
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


