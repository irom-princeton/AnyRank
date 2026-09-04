import json
import os
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from statistical_comparison_helpers import (
    compact_letter_display,
    draw_samples_from_beta_posterior,
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

def cld_rank(label):
    """Mean letter index: 'a'->0, 'ab'->0.5, 'b'->1, 'c'->2, 'd'->3."""
    idx = [ord(c) - ord("a") for c in label.lower()]
    return sum(idx) / len(idx)

def plot_model_comparison(
    model_name_list: List[str],
    labels: Optional[List[str]],
    result_arrays: List[np.ndarray],
    cld_letters: List[str],
    rng: np.random.Generator,
    mode: str = "success_rate",  # "success_rate" or "task_progress"
    progress_bins: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    add_legend: bool = False,
    unit_width: int = 5,
    height: int = 5,
    dpi: int = 360,
    ttd: Optional[Dict[str, int]] = None,
    CLD_COLOR: Optional[Dict[str, str]] = None,
) -> Union[None, plt.Figure]:
    """Makes a violin plot of success rate estimates with corresponding CLD letters
    for policy comparison.

    Args:
        model_name_list: A list of model names.
        result_arrays: A list of arrays indicating success/failure or progress for each model.
        cld_letters: A list of CLD letters corresponding to each model.
        rng: A numpy random Generator instance for posterior sampling.
        mode: Mode of the result arrays. "success_rate" for binary success/failure
            arrays, "task_progress" for progress arrays. Defaults to "success_rate".
        progress_bins: If mode is "task_progress", the unique progress bins. Defaults to None.
        output_path: Optional file path to save the plot. If None, the plot will not
            be saved but returned as a matplotlib Figure object. Defaults to None.
        title: Optional title for the plot. Defaults to None.
        add_legend: Whether to show legend on the plot. Defaults to False.
        unit_width: Figure width per model. Defaults to 6.
        height: Figure height. Defaults to 4.
        dpi: Resolution of the saved plot. Defaults to 100.
        ttd: Optional time-to-decision value to display on the plot. Defaults to None.
        imglib: Optional image library to use for saving the plot. Defaults to None.

    Returns:
        If output_path is None, returns a matplotlib Figure object containing
        the plot. Otherwise, saves the plot to the specified path and returns None.
    """
    if not mode in ["success_rate", "task_progress"]:
        raise ValueError(f"mode must be 'success_rate' or 'task_progress', got {mode}")
    if mode == "success_rate":
        if not progress_bins is None:
            raise ValueError(
                "progress_bins should be None when mode is 'success_rate'."
            )
        if any(
            not np.all(np.isin(result_array, [True, False]))
            for result_array in result_arrays
        ):
            raise ValueError(
                "All result arrays must be binary arrays of True/False "
                "when mode is 'success_rate'."
            )
    if mode == "task_progress":
        if progress_bins is None:
            raise ValueError(
                "progress_bins must be provided when mode is 'task_progress'."
            )
        if any(
            not np.all(np.isin(result_array, progress_bins))
            for result_array in result_arrays
        ):
            raise ValueError(
                "All result arrays must contain only values in progress_bins "
                "when mode is 'task_progress'."
            )
    num_models = len(model_name_list)

    posterior_samples = []
    means = []

    for result_array in result_arrays:
        if mode == "success_rate":
            samples = draw_samples_from_beta_posterior(result_array, rng)
        elif mode == "task_progress":
            # samples = draw_samples_from_dirichlet_posterior(
            #     result_array, progress_bins, rng
            # )
            samples = result_array  # Use the raw progress values for plotting instead of sampling from the posterior. Consistent with the LBM 1.0 paper.
        else:
            raise ValueError(f"Unknown mode: {mode}")
        posterior_samples.append(samples)
        means.append(np.mean(samples))

    fig, ax = plt.subplots(figsize=(max(unit_width, num_models), height), dpi=dpi)
    ax.tick_params(axis="y", labelsize=16)
    ax.tick_params(axis="x", labelsize=16)

    def cld_sort_key(s: str):
        # Standard CLD convention: 'a' is best.
        # Sort by first letter, then by length (so 'a' before 'ab' before 'b', etc.)
        return (s[0], len(s), s)

    groups = sorted(set(cld_letters), key=cld_sort_key)
    num_groups = len(groups)

    # Blues
    cmap = get_cmap("Blues")
    group_colors = [cmap(x) for x in np.linspace(0.9, 0.3, num_groups)]
    letter_to_color = {g: c for g, c in zip(groups, group_colors)}
    colors = [letter_to_color[cld] for cld in cld_letters]

    # Tab10:
    cmap = get_cmap("tab10")
    # Use first K tab10 colors (stable, categorical)
    group_colors = [cmap(i) for i in range(num_groups)]
    # Map CLD → color
    letter_to_color = dict(zip(groups, group_colors))
    # Final per-model colors (ties share color)
    colors = [letter_to_color[cld] for cld in cld_letters]

    parts = ax.violinplot(
        posterior_samples,
        positions=np.arange(num_models),
        showmeans=True,
        showmedians=False,
        showextrema=False,
        widths=0.8,
    )
    # for pc, color in zip(parts["bodies"], colors):
    #     pc.set_facecolor(color)
    #     pc.set_alpha(0.6)
    #     pc.set_edgecolor("black")
    #     pc.set_linewidth(1.2)
    # parts["cmeans"].set_color("black")
    # parts["cmeans"].set_linewidth(0.8)
    # if CLD_COLOR is None:
    #     CLD_COLOR = {
    #         "a": "#0076BA",  # dark blue (best)
    #         "b": "#7AACD2",  # medium blue
    #         "c": "#DFEAF6",  # light blue
    #         "d": "#D88281",  # red (worst)
    #         "ab": "#3D91C6",  # medium blue
    #         "bc": "#DFEAF6",
    #         "e": "#D88281",  # red (worst)
    #         "f": "#D88281",  # red (worst)
    #         "g": "#D88281",  # red (worst)
    #         "h": "#D88281",  # red (worst)
    #         "i": "#D88281",  # red (worst)
    #         "j": "#D88281",  # red (worst)
    #     }
    #     # get combination if cld letter not in script:
    #     # e.g. "ac" → mix of "a" and "c"
    #     letters = "abcdefghijklmnopqrstuvwxyz"
    #     for cld in set(cld_letters):
    #         if cld not in CLD_COLOR:
    #             base_colors = [CLD_COLOR[c] for c in cld if c in CLD_COLOR]
    #             if base_colors:
    #                 # Average the RGB values of the base colors
    #                 avg_color = np.mean(
    #                     [plt.colors.to_rgb(c) for c in base_colors], axis=0
    #                 )
    #                 CLD_COLOR[cld] = avg_color
    #             else:
    #                 # Default to gray if no known letters
    #                 CLD_COLOR[cld] = (0.5, 0.5, 0.5)

    letters = sorted({c for cld in cld_letters for c in cld})
    n = len(letters)
    cmap = plt.get_cmap("RdYlBu_r")
    LO, HI = 0.15, 0.9
    ranks = {L: cld_rank(L) for L in letters}
    vmax = max(ranks.values()) or 1
    # tab10
    # letter_colors = {letter: cmap(i % 10) for i, letter in enumerate(letters)}

    CLD_COLOR = {
        cld: cmap(LO + (HI - LO) * (cld_rank(cld) / vmax)) for cld in set(cld_letters)
    }

    # # letter_colors = {L: cmap(i / max(n - 1, 1)) for i, L in enumerate(sorted(letters))}
    # CLD_COLOR = {}

    # for cld in set(cld_letters):
    #     rgbs = [mcolors.to_rgb(letter_colors[c]) for c in cld]

    #     CLD_COLOR[cld] = np.mean(rgbs, axis=0)

    for pc, cld in zip(parts["bodies"], cld_letters):
        pc.set_facecolor(CLD_COLOR[cld])
        pc.set_alpha(1.0)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.2)
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(0.8)

    # Add CLD labels

    for i, (x, y, label) in enumerate(zip(np.arange(num_models), means, cld_letters)):
        if y < 0.03:
            ypos = 0.05  # Ensure label is above the x-axis
        elif y > 0.97:
            ypos = 0.95  # Ensure label is below the top of the plot
        else:
            ypos = y + 0.03  # Slightly above the mean
        ax.text(
            x - 0.025,
            ypos,
            label,
            fontsize=16,
            fontweight="bold",
            color="black",
            verticalalignment="center",
            zorder=4,
        )

    ax.set_xticks(np.arange(num_models))
    if labels:
        ax.set_xticklabels(labels, ha="center", rotation=30, fontsize=15)
    else:
        ax.set_xticklabels(
            [model.upper() for model in model_name_list], rotation=30, ha="center"
        )
    # --- top x-axis (TTD) ---
    if ttd is not None:
        ax_top = ax.secondary_xaxis("top")
        ax_top.set_xticks(np.arange(num_models))

        ttd_labels = []
        for model in model_name_list:
            v = ttd.get(model, None)
            ttd_labels.append(rf"${v:.0f}$" if v is not None else "")

        ax_top.set_xticklabels(ttd_labels, fontsize=15)
        ax_top.set_xlabel(
            "Time to decision", fontsize=16, labelpad=10, fontweight="bold"
        )

        # styling
        ax_top.tick_params(axis="x", length=0, pad=6)
        ax_top.spines["top"].set_visible(False)

    ax.set_ylim(0.0, 1.0)
    if mode == "success_rate":
        ax.set_ylabel("Success Rate", fontsize=18)
    elif mode == "task_progress":
        ax.set_ylabel("Task Progress", fontsize=18)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if title is not None:
        ax.set_title(title)
    if add_legend:
        ax.legend(
            parts["bodies"], [model.upper() for model in model_name_list], loc="best"
        )
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved a PNG plot to {output_path}")
    else:
        return fig