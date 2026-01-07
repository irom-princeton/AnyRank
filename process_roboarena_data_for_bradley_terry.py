"""
Implementation of basic Bradley-Terry Model fitting procedure, as described in 
Equations (4) and (5) at https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model

"""

import numpy as np
import os 
import argparse 
import copy

if __name__ == "__main__":

    data_progress = np.genfromtxt('data/per_trial_progress_data.csv', delimiter=',', dtype=float, skip_header=1, filling_values=-1)
    print()
    print("First 5 rows: ")
    print()
    print(data_progress[:5, :])
    print()

    N0 = data_progress.shape[0]
    n_policies = data_progress.shape[1]

    print("N0: ", N0)
    print("N policies: ", n_policies)

    counter = 0 
    data_progress_filtered = np.zeros((N0, n_policies))
    for i in range(N0):
        is_complete = True 
        if np.min(data_progress[i, :]) < 0:
            is_complete = False
        
        if is_complete:
            data_progress_filtered[counter, :] = copy.deepcopy(data_progress[i, :])
            counter += 1
        
    data_progress_filtered_truncated = copy.deepcopy(data_progress_filtered[:counter, :])
    print("Filtered and truncated data shape: ", data_progress_filtered_truncated.shape)
    np.save("Roboarena_BT.npy", data_progress_filtered_truncated)
    
