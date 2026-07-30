####
# Plot of hypotheses rejected vs. number of samples
####

from multitest.plot_cld import make_violin_plots_from_ttds, get_cld_letters_from_ttds
# from multitest.scripts.roboarena_individual_test import get_real_evals
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from brokenaxes import brokenaxes

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
    "AnyRank":                 "#f35b04",
    r"AnyRank$_0$":            "#f18701",
    r"AnyRank$_\infty$":       "#ffaa00",
    "Bonferroni":              "#023047",
    "Weighted\nBonferroni":    "#5fa8d3",
}

METHOD_ALPHA = {
    "AnyRank":                       1,
    r"AnyRank$_0$":                  0.8,
    r"AnyRank$_\infty$":             0.7,
    "Bonferroni":                    0.5,
    "Weighted\nBonferroni":          0.5,
}

METHOD_MARKERS = {
    "AnyRank":                       "o",
    r"AnyRank$_0$":                  "o",
    r"AnyRank$_\infty$":             "o",
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
    "roboarena7_wm_prior": "RoboArena7 (WM Prior)",
    "roboarena4_wm_prior": "RoboArena4 (WM Prior)",
    "roboarena7": "RoboArena7 (Heldout Evals Prior)",
    "roboarena4": "RoboArena4 (Heldout Evals Prior)",
}


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
        data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "roboarena", data_bin, filename)
        df = pd.read_csv(data_path)
        for k in keys:
            eval_results[k] = np.concatenate([eval_results[k], df[k].dropna().to_numpy()])
    return eval_results

def roboarena_wm_experiment(selected_policies=None, wm_prior_path=None, cfg=None):
    """Load roboarena data and return the generic inputs needed by main()."""
    if cfg is None:
        cfg = ExperimentConfig()
    bernoulli = False
    data_bins = ["data2"]
    if not selected_policies:
        selected_policies = ["paligemma_binning_droid", "pi0_droid", "paligemma_diffusion_droid", "pi0_fast_droid"]
    policy_data = get_real_evals(data_bins=data_bins, selected_policies=selected_policies)

    get_policy_summary(policy_data)
    Nmax_csv = min(len(v) for v in policy_data.values())
    _, real_evals = split_eval(policy_data, Nmax_csv, nruns=cfg.n_prior)
    policies = list(real_evals.keys())

    df_wm = pd.read_csv(wm_prior_path)
    wm_evals = {k: df_wm[df_wm['wm_id'] == k]['score'].to_numpy() for k in policies}
    sim_means = {k: np.mean(wm_evals[k]) for k in policies}
    real_means = {k: np.mean(real_evals[k]) for k in policies}
    return policy_data, policies, sim_means, real_means, bernoulli


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
    
    max_rejected = np.max(ys)
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
    
    max_rejected = np.max(ys)
    label = f"{label:<24} {max_rejected:2d}/{len(hypotheses_correct):2d}" if label else None
    ax.scatter(xs, ys, s=point_size, color=color, alpha=alpha, marker=marker, label=label, zorder=3, edgecolors="black", clip_on=False)
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
    plot_type="efficiency"
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
    
    ax.set_xlabel("Sample Efficiency", fontsize=12)
    ax.set_ylabel("Hypotheses Rejected", fontsize=12)
 
    if title:
        ax.set_title(title, fontsize=14)
    
    if plot_type=="complexity":
        ax.legend(frameon=True, fontsize=10, loc="lower left")
    else:
        ax.legend(frameon=True, fontsize=10, loc="lower left")
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
    meta_results_dir = "/n/fs/irom-testing/multitest/roboarena_outputs_graphical_active"
    postfix = "alpha_0.05_2026-07-29"
    method_names= [
        "AnyRank",
        r"AnyRank$_0$",
        r"AnyRank$_\infty$",
        "Bonferroni",
        "Weighted\nBonferroni",
    ]
    methods_dict = {"AnyRank": "graphical_active",  r"AnyRank$_0$": "graphical_active", r"AnyRank$_\infty$": "fixed", "Bonferroni": "bonferroni", "Weighted\nBonferroni": "weighted_bonferroni"}
    methods = {m: None for m in method_names}
    beta = 1
    plot_type="complexity"

    for exp_name in ["roboarena4_wm_prior", "roboarena7_wm_prior", "roboarena4", "roboarena7"]:
        for m in method_names:
            if m == r"AnyRank$_0$":
                beta=0
            else:
                beta=1
            method_ref = methods_dict[m]
            if postfix != "":
                results_dir = os.path.join(meta_results_dir, f"roboarena_{method_ref}_fully_connected_transitive_{postfix}", exp_name)
            else:
                results_dir = os.path.join(meta_results_dir, f"roboarena_{method_ref}_fully_connected_transitive", exp_name)
            hypotheses_reject_folder = os.path.join(results_dir, "rejection_order")
            for file in os.listdir(hypotheses_reject_folder):
                if "decision_times" in file and f"beta{beta}.json" in file:
                    hypotheses_reject_file = os.path.join(hypotheses_reject_folder, file)
            methods[m] = (hypotheses_reject_file, method_ref)

        four_policies = ["paligemma_binning_droid", "pi0_droid", "paligemma_diffusion_droid", "pi0_fast_droid"]
        seven_policies = ["paligemma_binning_droid",
                "pi0_droid",
                "paligemma_diffusion_droid",
                "paligemma_vq_droid",
                "paligemma_fast_specialist_droid",
                "paligemma_fast_droid",
                "pi0_fast_droid"]

        plot_policy_names = {
            "paligemma_binning_droid":        "PG-Bin",
            "paligemma_diffusion_droid":      "PG-Diff",
            "paligemma_vq_droid":             "PG-FSQ",
            "paligemma_fast_droid":           "PG-Fast",
            "paligemma_fast_specialist_droid": "PG-Fast-Spec",
            "pi0_droid":                      r"$\pi_0$",
            "pi0_fast_droid":                 r"$\pi_0$-Fast",
        }

        if "roboarena4" in exp_name:
            WM_cfg = ExperimentConfig(n_prior=0, beta=beta, results_dir=results_dir, graph_type="fully_connected")
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=four_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=WM_cfg)
            max_sample_size_per_model = {k:len(v) for k,v in policy_data.items()}
        else:
            WM_cfg = ExperimentConfig(n_prior=0, beta=beta, results_dir=results_dir, graph_type="fully_connected")
            policy_data, policies, sim_means, real_means, bernoulli = roboarena_wm_experiment(selected_policies=seven_policies, wm_prior_path='data/roboarena/wm_evals/evaluations.csv', cfg=WM_cfg)
            max_sample_size_per_model = {k:len(v) for k,v in policy_data.items()}
        
        # Plot in reversed order so that AnyRank is on top of the other methods in the legend
        METHOD_ORDER = {
            "roboarena4_wm_prior": [
                r"AnyRank$_\infty$",
                r"AnyRank$_0$",
                "AnyRank",
                "Bonferroni",
                "Weighted\nBonferroni",
            ][::-1],
            "roboarena4": [
                r"AnyRank$_\infty$",
                r"AnyRank$_0$",
                "AnyRank",
                "Bonferroni",
                "Weighted\nBonferroni",
            ][::-1],
            "roboarena7_wm_prior": [
                r"AnyRank$_0$",
                "AnyRank",
                r"AnyRank$_\infty$",
                "Bonferroni",
                "Weighted\nBonferroni",
            ][::-1],
            "roboarena7": [
                r"AnyRank$_0$",
                "AnyRank",
                r"AnyRank$_\infty$",
                "Bonferroni",
                "Weighted\nBonferroni",
            ][::-1]  # reverse order for plotting,
        }
        
        methods = {k: methods[k] for k in METHOD_ORDER[exp_name]}

        ax = plot_sample_complexity_multi(
            methods=methods,
            policy_names=policies,
            max_sample_size_per_model=max_sample_size_per_model,
            title=EXP_NAMES[exp_name],
            plot_type=plot_type
        )

        if plot_type == "efficiency":
            save_dir = os.path.join("imglib", exp_name, "sample_efficiency")
        else:
            save_dir = os.path.join("imglib", exp_name, "sample_complexity")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "sample_complexity.png"), dpi=600, bbox_inches="tight")
        print(f"Saved figure to: {os.path.join(save_dir, 'sample_complexity.png')}")
