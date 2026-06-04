import json
import os
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from sequentialized_barnard_tests.tools.plotting import (
    compact_letter_display,
    plot_model_comparison
)

K_progress = 40
progress_bins = np.arange(K_progress + 1) / K_progress


def save_plot_bundle(
    path_prefix,
    model_name_list,
    labels,
    arrays,
    cld_letters,
    ttd,
    progress_bins,
    mode,
):
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)

    # ---- numeric data ----
    np.savez_compressed(
        path_prefix + ".npz",
        model_name_list=np.array(model_name_list, dtype=object),
        labels=np.array(labels, dtype=object),
        arrays=np.array(arrays, dtype=object),
        cld_letters=np.array(cld_letters, dtype=object),
        ttd=np.array([ttd[m] for m in model_name_list]),
        progress_bins=np.array(progress_bins),
    )

    # ---- lightweight metadata ----
    meta = {
        "mode": mode,
        "model_name_list": model_name_list,
        "ttd_dict": ttd,
    }
    with open(path_prefix + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def get_cld_dict_generic(method_empirical_means, method_input_lists):
    """
    Compute CLD letters for an arbitrary collection of testing methods.

    Parameters
    ----------
    method_empirical_means : dict[str, dict[str, float]]
        {method_name: {alg: empirical_mean at TTD cutoff}}
    method_input_lists : dict[str, list[tuple]]
        {method_name: [(alg1, alg2), ...] pairs that reached a decision}

    Returns
    -------
    dict[str, dict[str, str]]
        {method_name: {alg: CLD letters}}
    """
    result = {}
    for method, empirical_means in method_empirical_means.items():
        sorted_models = [
            m for m, _ in sorted(empirical_means.items(), key=lambda kv: kv[1], reverse=True)
        ]
        letters_list = compact_letter_display(method_input_lists.get(method, []), sorted_models)
        
        result[method] = dict(zip(sorted_models, letters_list))
    return result


def get_cld_letters_from_ttds(progress_array_dict, ttd_dicts, max_sample_size_per_model, policies):
    """
    Compute CLD letters and per-alg max-TTD from pre-computed decision times.

    Designed to accept the avg_ttd_* dicts produced by
    run_graphical_test.run_experiments (graphical, evalues, bonferroni, etc.).

    Parameters
    ----------
    progress_array_dict : dict[str, np.ndarray]
        {alg_name: array of per-trial evaluations}
    ttd_dicts : dict[str, dict[tuple, float | str]]
        {method_name: {(alg1, alg2): ttd_value}}
        alg1 and alg2 must be policy name strings matching keys in
        progress_array_dict.  When your source uses integer indices
        (as run_graphical_test does), convert with:
            {(policy_index[i], policy_index[j]): ttd for (i, j), ttd in avg_ttd.items()}
        before passing here.  "N/A" values are treated as max_sample_size_per_model.
    max_sample_size_per_model : int
        Pairs whose TTD >= this value are treated as undecided (excluded from CLD).
    policies : list[str]
        Ordered list of policy names; must be keys in progress_array_dict.

    Returns
    -------
    dict[str, tuple[dict[str, str], dict[str, int]]]
        {method_name: (cld_dict, max_ttd_dict)}
        cld_dict     : {alg: CLD letters string}
        max_ttd_dict : {alg: max TTD seen across all pairs involving that alg}
    """
    method_empirical_means = {}
    method_input_lists = {}
    method_max_ttd = {}

    for method, ttd_dict in ttd_dicts.items():
        max_ttd = {alg: 0 for alg in policies}
        input_list = []

        

        for (alg1, alg2), ttd in ttd_dict.items():
            max_sample_size_hyp = min(max_sample_size_per_model[alg1], max_sample_size_per_model[alg2])
            
            ## TODO: Correct logic for final plot
            ttd_val = max_sample_size_hyp if ttd == "N/A" else int(ttd)
            if ttd_val < max_sample_size_hyp:
                input_list.append((alg1, alg2))
            
            # if ttd != "N/A":
            #     ttd_val = int(ttd)
            #     input_list.append((alg1, alg2))
            # else:
            #     ttd_val = max_sample_size_hyp
            max_ttd[alg1] = max(ttd_val, max_ttd.get(alg1, 0))
            max_ttd[alg2] = max(ttd_val, max_ttd.get(alg2, 0))

        empirical_means = {}
        for alg in policies:
            data = progress_array_dict.get(alg, np.array([]))
            n = max_ttd.get(alg, 0)
            empirical_means[alg] = float(np.mean(data[:n])) if n > 0 and len(data) > 0 else 0.0

        method_empirical_means[method] = empirical_means
        method_input_lists[method] = input_list
        method_max_ttd[method] = max_ttd

    cld_by_method = get_cld_dict_generic(method_empirical_means, method_input_lists)
    return {method: (cld_by_method[method], method_max_ttd[method]) for method in ttd_dicts}


def make_violin_plots_from_ttds(
    progress_array_dict,
    ttd_dicts,
    policies,
    labels,
    max_sample_size_per_model,
    output_dir,
    mode="task_progress",
    K_progress=40,
):
    """
    Generate violin plots and save plot bundles for each testing method.

    Parameters
    ----------
    progress_array_dict : dict[str, np.ndarray]
        {alg_name: array of per-trial evaluations}
    ttd_dicts : dict[str, dict[tuple, float | str]]
        {method_name: {(alg1, alg2): ttd}} — pre-computed TTDs, e.g. from
        run_graphical_test.run_experiments.  Keys must be policy name strings.
    policies : list[str]
        Ordered list of policy names.
    labels : list[str]
        Display labels corresponding to policies (same length and order).
    max_sample_size_per_model : dict
        Threshold above which a pair is considered undecided.
    output_dir : str
        Directory where figures (.png) and bundles (.npz/.json) are saved.
    mode : str
        "task_progress" for continuous [0, 1] data (default) or
        "success_rate" for binary {0, 1} data. Controls rectification,
        progress_bins, and how plot_model_comparison renders the violin.
    K_progress : int
        Number of bins for violin rectification. Only used when
        mode="task_progress". Defaults to 40.
    """
    if mode not in ("task_progress", "success_rate"):
        raise ValueError(f"mode must be 'task_progress' or 'success_rate', got {mode!r}")

    os.makedirs(output_dir, exist_ok=True)
    if mode == "task_progress":
        local_progress_bins = np.arange(K_progress + 1) / K_progress
        arrays_dict = {
            alg: np.floor(K_progress * data) / K_progress
            for alg, data in progress_array_dict.items()
        }
    else:
        local_progress_bins = None
        arrays_dict = dict(progress_array_dict)

    print("Computing CLD letters from pre-computed TTDs...")
    cld_results = get_cld_letters_from_ttds(
        progress_array_dict, ttd_dicts, max_sample_size_per_model, policies
    )

    for method, (cld_dict, max_ttd) in cld_results.items():
        active = [alg for alg in policies if max_ttd[alg] > 0]
        active_labels = [labels[policies.index(alg)] for alg in active]
        if not active:
            print(f"  [{method}] skipping — no policies with evaluated data")
            continue
        cld_letters = [cld_dict[alg] for alg in active]
        arrays = [arrays_dict[alg][:int(max_ttd[alg])] for alg in active]
        ttd_for_bundle = {alg: int(max_ttd[alg]) for alg in active}

        plot_path = os.path.join(output_dir, f"violin_{method}.png")
        plot_model_comparison(
            model_name_list=active,
            labels=active_labels,
            result_arrays=arrays,
            cld_letters=cld_letters,
            rng=np.random.default_rng(123),
            mode=mode,
            progress_bins=local_progress_bins,
            output_path=plot_path,
            ttd=ttd_for_bundle,
        )

        save_plot_bundle(
            path_prefix=os.path.join(output_dir, method),
            model_name_list=active,
            labels=active_labels,
            arrays=arrays,
            cld_letters=cld_letters,
            ttd=ttd_for_bundle,
            progress_bins=local_progress_bins if local_progress_bins is not None else np.array([]),
            mode=mode,
        )
        print(f"  [{method}] violin saved to {plot_path}")
