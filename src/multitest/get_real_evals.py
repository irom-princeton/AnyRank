import numpy as np
import os
import pandas as pd

def get_real_evals(data_bins=["data1"], selected_policies=[]):
    filename = "per_trial_progress_data.csv"
    if selected_policies:
        keys = selected_policies
    else:
        keys = [
            "paligemma_binning_droid",
            "pi0_droid",
            "paligemma_vq_droid",
            "paligemma_fast_specialist_droid",
            "paligemma_fast_droid",
            "paligemma_diffusion_droid",
            "pi0_fast_droid",
        ]
    eval_results = {k: np.array([]) for k in keys}
    for data_bin in data_bins:
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "roboarena", data_bin, filename)
        df = pd.read_csv(data_path)
        for k in keys:
            eval_results[k] = np.concatenate([eval_results[k], df[k].dropna().to_numpy()])
    return eval_results