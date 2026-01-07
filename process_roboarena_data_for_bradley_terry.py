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
    data_preference = np.genfromtxt('data/policy_preferences.csv', delimiter=',', dtype=str, skip_header=1, filling_values='missing')
    
    policy_names = ["pi0_droid",
        "paligemma_vq_droid",
        "paligemma_fast_specialist_droid",
        "pi0_fast_droid",
        "paligemma_fast_droid",
        "paligemma_diffusion_droid",
        "paligemma_binning_droid",
    ]
    
    print()
    print("First 5 rows: ")
    print()
    print(data_progress[:5, :])
    print()

    N0 = data_progress.shape[0]
    n_policies = data_progress.shape[1]

    assert data_preference.shape[0] == N0


    print("N0: ", N0)
    print("N policies: ", n_policies)

    counter = 0 
    data_progress_filtered = np.zeros((N0, n_policies))
    data_preference_filtered = np.zeros((N0, 3))
    binning_preferred = 0
    binning_not_preferred = 0
    for i in range(N0):
        is_complete = True 
        if np.min(data_progress[i, :]) < 0:
            is_complete = False
        
        if is_complete:
            data_progress_filtered[counter, :] = copy.deepcopy(data_progress[i, :])
            bool0 = [policy == data_preference[i, 0] for policy in policy_names]
            idx0 = np.argwhere(bool0)
            bool1 = [policy == data_preference[i, 1] for policy in policy_names]
            idx1 = np.argwhere(bool1)
            data_preference_filtered[counter, 0] = idx0[0][0]
            data_preference_filtered[counter, 1] = idx1[0][0]

            if idx0 == 6:
                binning_preferred += 1
            elif idx1 == 6: 
                binning_not_preferred += 1
            # In every case, policy 0 is favored; therefore last column of data_preference_filtered is always 0

            # Update counter
            counter += 1
        
    data_progress_filtered_truncated = copy.deepcopy(data_progress_filtered[:counter, :])
    data_preference_filtered_truncated = copy.deepcopy(data_preference_filtered[:counter, :])
    print("Filtered and truncated data shape: ", data_progress_filtered_truncated.shape)
    np.save("Roboarena_progress.npy", data_progress_filtered_truncated)
    np.save("Roboarena_preference.npy", data_preference_filtered_truncated)

    print(f"Binning policy preferred {100.* binning_preferred / (binning_preferred + binning_not_preferred):.3f}% of the time it is queried.")

