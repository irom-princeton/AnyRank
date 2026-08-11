####
# Plot of hypotheses rejected vs. number of samples
####

from multitest.plot_cld import make_violin_plots_from_ttds, get_cld_letters_from_ttds
import sys
sys.path.insert(0, "/n/fs/irom-testing/multitest")

import scripts.lbm_individual_test as lbm_mod
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json
import pandas as pd
from pdb import set_trace as st
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

from multitest.individual_test import (
    ExperimentConfig,
    split_eval,
    get_policy_summary,
)

# METHOD_COLORS = {
#     "AnyRank":                       "#00A6FF",
#     r"AnyRank$_0$":                  "#08306b",
#     r"AnyRank$_\infty$":             "#1f77b4",
#     "Bonferroni":                    "#ff7f0e",
#     "Weighted\nBonferroni":          "#d62728",
# }

METHOD_COLORS = {
    "AnyRank":                 "#00A6FF",
    r"AnyRank$_0$":            "#dc267f",
    r"AnyRank$_\infty$":       "#785ef0",
    "Bonferroni":              "#ff7f0e",
    "Weighted\nBonferroni":    "#d62728",
}

METHOD_ALPHA = {
    "AnyRank":                       1.0,
    r"AnyRank$_0$":                  1.0,
    r"AnyRank$_\infty$":             0.7,
    "Bonferroni":                    0.7,
    "Weighted\nBonferroni":          0.7,
}

METHOD_MARKERS = {
    "AnyRank":                       "o",
    r"AnyRank$_0$":                  "*",
    r"AnyRank$_\infty$":             "^",
    "Bonferroni":                    "s",
    "Weighted\nBonferroni":          "s",
}

METHOD_EDGE_COLORS = {
    "AnyRank":                       "#000000",
    r"AnyRank$_0$":                  "#000000",
    r"AnyRank$_\infty$":             "#000000",
    "Bonferroni":                    "#FFFFFF",
    "Weighted\nBonferroni":          "#FFFFFF",
}

EXP_NAMES = {
    "experiment_A":                                      "PushCoaster (Nominal)",
    "experiment_B_id_True_ood_False":                    "TurnMug (Nominal)",
    "experiment_B_id_False_ood_True":                    "LBM Exp B (Dist. Shift)",
    "experiment_B_id_True_ood_True":                     "TurnMug (Nominal + OOD)",
    "experiment_C_id_True_ood_False_distshift_None":     "Aggregate (Nominal)",
    "experiment_C_id_True_ood_True_distshift_distshift": "Aggregate (Nominal + OOD object)",
    "experiment_C_id_True_ood_True_distshift_novel":     "Nominal (Nominal + OOD Station)",
}

# Map each results-tree experiment folder to the lbm_individual_test loader
# that generated it (function name + keyword args). Re-running the same loader
# recovers the policy order and per-policy sample counts used to write results.
LBM_EXPERIMENTS = {
    "experiment_A":                                      ("experiment_A", {}),
    "experiment_B_id_True_ood_False":                    ("experiment_B", dict(id=True,  ood=False)),
    "experiment_B_id_False_ood_True":                    ("experiment_B", dict(id=False, ood=True)),
    "experiment_B_id_True_ood_True":                     ("experiment_B", dict(id=True,  ood=True)),
    "experiment_C_id_True_ood_False_distshift_None":     ("experiment_C", dict(id=True,  ood=False, distshift=None)),
    "experiment_C_id_True_ood_True_distshift_distshift": ("experiment_C", dict(id=True,  ood=True,  distshift="distshift")),
    "experiment_C_id_True_ood_True_distshift_novel":     ("experiment_C", dict(id=True,  ood=True,  distshift="novel")),
}


def _ensure_lbm_df():
    """Load the TRI-LBM Fig2 CSV into the lbm_individual_test module global `df`
    (the experiment_* loaders read it from module scope)."""
    if getattr(lbm_mod, "df", None) is None:
        lbm_mod.df = lbm_mod.load_full_data(lbm_mod.FULL_DATA_DIR / "Fig2.csv")


def lbm_experiment(exp_name):
    """Load TRI-LBM eval data for one experiment and return the generic inputs
    needed for plotting.

    Mirrors ``roboarena_wm_experiment`` but sources data from the TRI-LBM Fig2
    evals via the loaders in ``lbm_individual_test``. The policy order matches
    the "i,j" indices in the decision_times JSON because it re-runs the same
    loader that generated the results.

    Returns
    -------
    policy_data : dict[str, np.ndarray]
    policies : list[str]
        Ordered policy names (index i -> policies[i]).
    sim_means, real_means : dict[str, float]
    max_sample_size_per_model : dict[str, float]
        policy name -> number of real trials.
    """
    _ensure_lbm_df()
    fn_name, kwargs = LBM_EXPERIMENTS[exp_name]
    policy_data, real_means, sim_means = getattr(lbm_mod, fn_name)(**kwargs)
    policies = list(policy_data.keys())
    max_sample_size_per_model = {k: len(v) for k, v in policy_data.items()}
    return policy_data, policies, sim_means, real_means, max_sample_size_per_model


def parse_inputs_from_file(
    results_path: str,
    method_key: str,
    policy_names: list[str],
    max_sample_size_per_model: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Read decision_times and hypotheses_correct from a JSON results file,
    replacing "N/A" entries with the minimum of the two policies' max samples.
 
    Parameters
    ----------
    results_path : str
        Path to JSON file with "decision_times" and "hypotheses_correct" fields,
        each containing per-method lists of dicts.
    method_key : str
        Key into each top-level field, e.g. "graphical_active", "bonferroni".
    policy_names : list[str]
        policy_names[i] is the policy name for index i.
    max_sample_size_per_model : dict[str, float]
        Maps policy name -> maximum evaluations. "N/A" decision times are
        replaced with min(max[policy_a], max[policy_b]).
 
    Returns
    -------
    hypothesis_samples : dict[str, float]
        Keyed by "i,j"; decision times with N/A filled in.
    hypotheses_correct : dict[str, float]
        Keyed by "i,j"; values 0 (undecided) / 1 or -1 (rejected).
    """
    with open(results_path) as f:
        data = json.load(f)
    
    raw_times   = data["decision_times"][method_key][0]
    raw_correct = data["hypotheses_correct"][method_key][0]
    
    hypothesis_samples = {}
    for key, val in raw_times.items():
        if val == "N/A":
            i, j = map(int, key.split(","))
            policy_a, policy_b = policy_names[i], policy_names[j]
            val = min(
                max_sample_size_per_model[policy_a],
                max_sample_size_per_model[policy_b],
            )
        hypothesis_samples[key] = float(val)
 
    hypotheses_correct = {k: float(v) for k, v in raw_correct.items()}
    return hypothesis_samples, hypotheses_correct

def parse_inputs_cld(
    hypotheses_correct_active: list[dict[str, float]],
    ttd_graphical_active: dict[tuple[str, str], float],
    policy_names: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Parse raw experiment objects into aligned hypothesis_samples and
    hypotheses_correct dicts keyed by "i,j" index strings.
 
    Parameters
    ----------
    hypotheses_correct_active : list[dict]
        List with one dict; keys are "i,j" index strings, values are
        0.0 (undecided) or 1.0 / -1.0 (rejected).
    ttd_graphical_active : dict
        Keys are (policy_a, policy_b) name tuples; values are samples consumed.
    policy_names : list[str]
        policy_names[i] gives the policy name for index i.
 
    Returns
    -------
    hypothesis_samples : dict[str, float]
        Keyed by "i,j"; values are samples consumed.
    hypotheses_correct : dict[str, float]
        Keyed by "i,j"; values are 0 / 1 / -1.
    """
    # Build name -> index lookup
    name_to_idx = {name: i for i, name in enumerate(policy_names)}
 
    # Convert ttd (name-tuple keys) -> index-string keys
    hypothesis_samples = {}
    for (a, b), n in ttd_graphical_active.items():
        i, j = name_to_idx[a], name_to_idx[b]
        key = f"{i},{j}"
        hypothesis_samples[key] = n
 
    # hypotheses_correct_active is a list; take the first element
    hypotheses_correct = hypotheses_correct_active[0]
    return hypothesis_samples, hypotheses_correct

def parse_inputs(
    results_path: str,
    method_key: str,
    policy_names: list[str],
    max_sample_size_per_model: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Read decision_times and hypotheses_correct from a JSON results file,
    replacing "N/A" entries with the minimum of the two policies' max samples.
 
    Parameters
    ----------
    results_path : str
        Path to JSON file with "decision_times" and "hypotheses_correct" fields,
        each containing per-method lists of dicts.
    method_key : str
        Key into each top-level field, e.g. "graphical_active", "bonferroni".
    policy_names : list[str]
        policy_names[i] is the policy name for index i.
    max_sample_size_per_model : dict[str, float]
        Maps policy name -> maximum evaluations. "N/A" decision times are
        replaced with min(max[policy_a], max[policy_b]).
 
    Returns
    -------
    hypothesis_samples : dict[str, float]
        Keyed by "i,j"; decision times with N/A filled in.
    hypotheses_correct : dict[str, float]
        Keyed by "i,j"; values 0 (undecided) / 1 or -1 (rejected).
    """
    with open(results_path) as f:
        data = json.load(f)
    
    raw_times   = data["decision_times"][method_key][0]
    raw_correct = data["hypotheses_correct"][method_key][0]
    
    hypothesis_samples = {}

    for key, val in raw_times.items():
        if val == "N/A":
            i, j = map(int, key.split(","))
            policy_a, policy_b = policy_names[i], policy_names[j]
            val = min(
                max_sample_size_per_model[policy_a],
                max_sample_size_per_model[policy_b],
            )
        hypothesis_samples[key] = float(val)
         
    hypotheses_correct = {k: float(v) for k, v in raw_correct.items()}
    return hypothesis_samples, hypotheses_correct
 
 
def plot_efficiency(
    results_path: str,
    method_key: str,
    policy_names: list[str],
    max_sample_size_per_model: dict[str, float],
    color: str = "#185FA5",
    alpha: float = 1.0,
    marker: str | None = None,
    label: str | None = None,
    point_size: float = 120,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot one method's scatter of cumulative hypotheses rejected vs. sample efficiency.
    Intended to be called once per method when overlaying multiple methods.
 
        sample_efficiency = 1 - cumulative_samples_consumed / total_samples
    where total_samples = sum(max_sample_size_per_model.values()).
 
    Parameters
    ----------
    results_path : str
        Path to JSON results file.
    method_key : str
        Key into decision_times/hypotheses_correct, e.g. "graphical_active".
    policy_names : list[str]
        Index-to-name mapping.
    max_sample_size_per_model : dict[str, float]
        Maps policy name -> max evaluations. Used for N/A replacement and
        as total_samples denominator (summed across all policies).
    color : str
        Scatter point color for this method.
    label : str or None
        Legend label for this method.
    point_size : float
        Marker size.
    ax : plt.Axes or None
        Axes to draw on. If None, a new figure is created.
 
    Returns
    -------
    ax : plt.Axes
    """
    hypothesis_samples, hypotheses_correct = parse_inputs(
        results_path, method_key, policy_names, max_sample_size_per_model
    )
    total_samples = sum(max_sample_size_per_model.values())
 
    sorted_pairs = sorted(hypothesis_samples.items(), key=lambda kv: kv[1])
 
    xs, ys = [], []
    max_samples_seen = 0
    cum_rejected = 0
    for key, n_samples in sorted_pairs:
        # values are absolute sample counts, not deltas — track running max
        max_samples_seen = max(max_samples_seen, n_samples)
        if hypotheses_correct.get(key, 0) != 0:
            cum_rejected += 1
            xs.append(1.0 - max_samples_seen / total_samples)
            ys.append(cum_rejected)
 
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    
    max_rejected = int(np.max(ys)) if ys else 0
    label = f"{label:<24} {max_rejected:2d}/{len(hypotheses_correct):2d}" if label else None
    ax.scatter(xs, ys, s=point_size, color=color, alpha=alpha, marker=marker, label=label, zorder=3, edgecolors="black", clip_on=False)
    return ax


def plot_sample_complexity(
    results_path: str,
    method_key: str,
    policy_names: list[str],
    max_sample_size_per_model: dict[str, float],
    color: str = "#185FA5",
    alpha: float = 1.0,
    marker: str | None = None,
    label: str | None = None,
    point_size: float = 120,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot one method's scatter of cumulative hypotheses rejected vs. sample efficiency.
    Intended to be called once per method when overlaying multiple methods.
 
        sample_efficiency = 1 - cumulative_samples_consumed / total_samples
    where total_samples = sum(max_sample_size_per_model.values()).
 
    Parameters
    ----------
    results_path : str
        Path to JSON results file.
    method_key : str
        Key into decision_times/hypotheses_correct, e.g. "graphical_active".
    policy_names : list[str]
        Index-to-name mapping.
    max_sample_size_per_model : dict[str, float]
        Maps policy name -> max evaluations. Used for N/A replacement and
        as total_samples denominator (summed across all policies).
    color : str
        Scatter point color for this method.
    label : str or None
        Legend label for this method.
    point_size : float
        Marker size.
    ax : plt.Axes or None
        Axes to draw on. If None, a new figure is created.
 
    Returns
    -------
    ax : plt.Axes
    """
    hypothesis_samples, hypotheses_correct = parse_inputs(
        results_path, method_key, policy_names, max_sample_size_per_model
    )
    total_samples = sum(max_sample_size_per_model.values())
 
    sorted_pairs = sorted(hypothesis_samples.items(), key=lambda kv: kv[1])
 
    xs, ys = [], []
    max_samples_seen = 0
    cum_rejected = 0
    for key, n_samples in sorted_pairs:
        # values are absolute sample counts, not deltas — track running max
        max_samples_seen = max(max_samples_seen, n_samples)
        if hypotheses_correct.get(key, 0) != 0:
            cum_rejected += 1
            xs.append(-1*max_samples_seen)
            ys.append(cum_rejected)
 
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    
    
    max_rejected = int(np.max(ys)) if ys else 0 if ys else 0
    label = f"{label:<24} {max_rejected:2d}/{len(hypotheses_correct):2d}" if label else None
    ax.scatter(xs, ys, s=point_size, color=color, alpha=alpha, marker=marker, label=label, zorder=3, edgecolors="black", clip_on=False)
    # ax.set_xscale("log")
    return ax
 
 
def plot_sample_complexity_multi(
    methods: dict[str, tuple[str, str] | str],
    policy_names: list[str],
    max_sample_size_per_model: dict[str, float],
    results_path: str | None = None,
    colors: dict[str, str] | None = None,
    point_size: float =120,
    figsize: tuple[float, float] = (4, 4),
    title: str | None = None,
    plot_type="efficiency",
    save_dir: str = "imglib/comparisons",
) -> plt.Axes:
    """
    Overlay multiple methods on one scatter plot.
 
    Each method can read from its own JSON file or share a common one.
 
    Parameters
    ----------
    methods : dict[str, tuple[str, str] | str]
        Maps display name -> (results_path, method_key) or just method_key.
        If a bare method_key string is given, the top-level results_path is used.
        Examples:
            # all from one file
            {"AnyRank": "graphical_active", "Bonferroni": "bonferroni"}
            # each from its own file
            {"AnyRank": ("anyrank.json", "graphical_active"),
             "Bonferroni": ("bonferroni.json", "bonferroni")}
            # mixed
            {"AnyRank": ("anyrank.json", "graphical_active"),
             "Bonferroni": "bonferroni"}
    policy_names : list[str]
        Shared index-to-name mapping across all methods.
    max_sample_size_per_model : dict[str, float]
        Maps policy name -> max evaluations. Shared across all methods.
    results_path : str or None
        Fallback JSON file used when a method entry is a bare method_key string.
        Required if any method value is not a (path, key) tuple.
    colors : dict[str, str] or None
        Display name -> hex color. Falls back to METHOD_COLORS, then
        matplotlib default cycle for unrecognized names.
    point_size : float
        Marker size (shared across methods).
    figsize : tuple
        Figure size in inches.
    title : str or None
        Optional plot title.
 
    Returns
    -------
    ax : plt.Axes
 
    Example
    -------
    >>> ax = plot_sample_complexity_multi(
    ...     methods={
    ...         "AnyRank":    ("anyrank.json",    "graphical_active"),
    ...         "Bonferroni": ("bonferroni.json", "bonferroni"),
    ...     },
    ...     policy_names=policy_names,
    ...     max_sample_size_per_model=max_sample_size_per_model,
    ... )
    """
    color_map = {**METHOD_COLORS, **(colors or {})}
    color_alpha = {**METHOD_ALPHA}
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
 
    fig, ax = plt.subplots(figsize=figsize)

    for idx, (display_name, method_spec) in enumerate(methods.items()):
        if isinstance(method_spec, tuple):
            path, method_key = method_spec
        else:
            if results_path is None:
                raise ValueError(
                    f"Method '{display_name}' has no results_path and no "
                    "fallback results_path was provided."
                )
            path, method_key = results_path, method_spec
 
        color = color_map.get(display_name, default_cycle[idx % len(default_cycle)])
        alpha = color_alpha.get(display_name, 0.8)  # add alpha for transparency
        if plot_type == "efficiency":
            try:
                plot_efficiency(
                    results_path=path,
                    method_key=method_key,
                    policy_names=policy_names,
                    max_sample_size_per_model=max_sample_size_per_model,
                    color=color,
                    alpha=alpha,
                    marker=METHOD_MARKERS.get(display_name, "o"),
                    label=display_name,
                    point_size=point_size,
                    ax=ax,
                )
            except:
                st()
            ax.set_xlabel("Sample Efficiency", fontsize=12)
        else:
            plot_sample_complexity(
                results_path=path,
                method_key=method_key,
                policy_names=policy_names,
                max_sample_size_per_model=max_sample_size_per_model,
                color=color,
                alpha=alpha,
                marker=METHOD_MARKERS.get(display_name, "o"),
                label=display_name,
                point_size=point_size,
                ax=ax,
            )
            ax.set_xlabel("Sample Complexity", fontsize=12)
    ax.set_ylabel("Hypotheses Rejected", fontsize=12)
 
    if title:
        ax.set_title(title, fontsize=14)
    
    if plot_type=="complexity":
        ax.legend(frameon=True, fontsize=10, loc="best")
    else:
        ax.legend(frameon=True, fontsize=10, loc="best")
    # handles, labels = ax.get_legend_handles_labels()

    # fig_leg = plt.figure(figsize=(6, 0.5))
    # fig_leg.legend(handles, labels, frameon=True, fontsize=10,
    #             loc="center")
    # fig_leg.savefig(f"{save_dir}/legend.pdf", bbox_inches="tight", pad_inches=0.02)
    # plt.close(fig_leg)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    all_x = [pt.get_offsets()[:, 0].min() for pt in ax.collections if len(pt.get_offsets())]
    all_y = [pt.get_offsets()[:, 1].max() for pt in ax.collections if len(pt.get_offsets())]
    # x_min = min(all_x) - 0.01 if all_x else 0.0
    # x_min = int(x_min*10)/10
    # ax.set_xlim(x_min, 1)
    ax.set_ylim(0, max(all_y)+1)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    return ax
# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Never drop into pdb if an lbm loader hits its except-branch breakpoint().
    os.environ.setdefault("PYTHONBREAKPOINT", "0")

    # TRI-LBM sequential-test results live in a single tree per graph type,
    # with all methods stored in one decision_times JSON -- unlike the
    # per-method RoboArena folders.
    meta_results_dir = "/n/fs/irom-testing/multitest/corl_sub"
    meta_results_dir="/n/fs/irom-testing/multitest/lbm_outputs_new"
    lbm_results_subdir = "lbm_fully_connected_active_transitive_2026-07-30_alpha_0.1"

    method_names = [
        "AnyRank",
        r"AnyRank$_0$",
        r"AnyRank$_\infty$",
        "Bonferroni",
        "Weighted\nBonferroni",
    ]
    methods_dict = {
        "AnyRank":                "graphical_active",
        r"AnyRank$_0$":           "graphical_active",
        r"AnyRank$_\infty$":      "fixed",
        "Bonferroni":             "bonferroni",
        "Weighted\nBonferroni":   "weighted_bonferroni",
    }
    plot_type = "complexity"

    # Plot in reversed order so that AnyRank sits on top in the legend / z-order.
    METHOD_ORDER = [
        r"AnyRank$_0$",
        "AnyRank",
        r"AnyRank$_\infty$",
        "Bonferroni",
        "Weighted\nBonferroni",
    ][::-1]

    for exp_name in EXP_NAMES:
        results_dir = os.path.join(meta_results_dir, lbm_results_subdir, exp_name)
        rejection_folder = os.path.join(results_dir, "rejection_order")

        if not os.path.isdir(rejection_folder):
            print(f"[skip] {exp_name}: no rejection_order folder at {rejection_folder}")
            continue

        # All methods read from the same decision_times JSON; AnyRank_0 uses the
        # beta=0 file, every other method uses beta=1. Skip the whole experiment
        # (with a warning) if any required file is missing -- an incomplete run.
        methods = {}
        missing = False
        for m in method_names:
            beta = 0 if m == r"AnyRank$_0$" else 1
            method_ref = methods_dict[m]
            decision_times_file = None
            for file in os.listdir(rejection_folder):
                if "decision_times" in file and file.endswith(f"beta{beta}.json"):
                    decision_times_file = os.path.join(rejection_folder, file)
            if decision_times_file is None:
                print(f"[skip] {exp_name}: no decision_times beta{beta} JSON in {rejection_folder}")
                missing = True
                break
            methods[m] = (decision_times_file, method_ref)
        if missing:
            continue

        # Ordered policy names + per-policy max samples straight from the
        # TRI-LBM eval loaders (order matches the results' policy indices).
        policy_data, policies, sim_means, real_means, max_sample_size_per_model = lbm_experiment(exp_name)

        methods = {k: methods[k] for k in METHOD_ORDER}

        sub = "sample_efficiency" if plot_type == "efficiency" else "sample_complexity"
        save_dir = os.path.join("imglib", "lbm", exp_name, sub)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Plotting {exp_name} to {save_dir} ...")
        ax = plot_sample_complexity_multi(
            methods=methods,
            policy_names=policies,
            max_sample_size_per_model=max_sample_size_per_model,
            title=EXP_NAMES[exp_name],
            plot_type=plot_type,
            save_dir=save_dir
        )

        # sub = "sample_efficiency" if plot_type == "efficiency" else "sample_complexity"
        # save_dir = os.path.join("imglib", "lbm", exp_name, sub)
        # os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "sample_complexity.png")
        plt.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(ax.figure)
        print(f"Saved figure to: {out_path}")
