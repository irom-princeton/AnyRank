### Barplot comparing hypotheses rejected against sample complexity
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D


# (rejections, sample complexity)
lbm_data = {
    "PushCoaster": {
        "AnyRank": ("2/3", 125),
        r"AnyRank$_0$": ("2/3", 117),
        r"AnyRank$_\infty$": ("2/3", 110),
        "Bonferroni": ("2/3", 125),
        "Weighted\nBonferroni": ("2/3", 146),
    },
    "Aggregate": {
        "AnyRank": ("3/3", 166),
        r"AnyRank$_0$": ("3/3", 156),
        r"AnyRank$_\infty$": ("3/3", 154),
        "Bonferroni": ("3/3", 166),
        "Weighted\nBonferroni": ("3/3", 254),
    },
}

lbm_ood_data = {
    "PushCoaster": {
        "AnyRank": ("2/3", 125),
        r"AnyRank$_0$": ("2/3", 117),
        r"AnyRank$_\infty$": ("2/3", 110),
        "Bonferroni": ("2/3", 125),
        "Weighted\nBonferroni": ("2/3", 146),
    },
    "TurnMug": {
        "AnyRank": ("3/3", 128),
        r"AnyRank$_0$": ("3/3", 66),
        r"AnyRank$_\infty$": ("1/3", 150),
        "Bonferroni": ("2/3", 130),
        "Weighted\nBonferroni": ("2/3", 138),
    },
    "Aggregate": {
        "AnyRank": ("3/3", 166),
        r"AnyRank$_0$": ("3/3", 156),
        r"AnyRank$_\infty$": ("3/3", 154),
        "Bonferroni": ("3/3", 166),
        "Weighted\nBonferroni": ("3/3", 254),
    },
    "TurnMug (Nom.+OOD)": {
        "AnyRank": ("8/15", 300),
        r"AnyRank$_0$": ("14/15", 168),
        r"AnyRank$_\infty$": ("0/15", 100),
        "Bonferroni": ("11/15", 299),
        "Weighted\nBonferroni": ("10/15", 300),
    },
    "Aggregate (Nom.+OOD obj)": {
        "AnyRank": ("15/15", 622),
        r"AnyRank$_0$": ("15/15", 286),
        r"AnyRank$_\infty$": ("9/15", 551),
        "Bonferroni": ("11/15", 818),
        "Weighted\nBonferroni": ("6/15", 900),
    },
    "Aggregate (Nom.+OOD station)": {
        "AnyRank": ("15/15", 614),
        r"AnyRank$_0$": ("15/15", 224),
        r"AnyRank$_\infty$": ("2/15", 416),
        "Bonferroni": ("10/15", 618),
        "Weighted\nBonferroni": ("8/15", 750),
    },
}

roboarena_data = {
    "RoboArena-4 (held-out)": {
        "AnyRank": ("6/6", 1366),
        r"AnyRank$_0$": ("6/6", 1380),
        r"AnyRank$_\infty$": ("6/6", 1317),
        "Bonferroni": ("6/6", 2599),
        "Weighted\nBonferroni": ("6/6", 2912),
    },
    "RoboArena-7 (held-out)": {
        "AnyRank": ("10/21", 7729),
        r"AnyRank$_0$": ("12/21", 6929),
        r"AnyRank$_\infty$": ("9/21", 3738),
        "Bonferroni": ("12/21", 6950),
        "Weighted\nBonferroni": ("11/21", 7138),
    },
    "RoboArena-4 (WM)": {
        "AnyRank": ("6/6", 2445),
        r"AnyRank$_0$": ("6/6", 1380),
        r"AnyRank$_\infty$": ("6/6", 1317),
        "Bonferroni": ("6/6", 2599),
        "Weighted\nBonferroni": ("5/6", 3020),
    },
    "RoboArena-7 (WM)": {
        "AnyRank": ("10/21", 7529),
        r"AnyRank$_0$": ("12/21", 6929),
        r"AnyRank$_\infty$": ("9/21", 3738),
        "Bonferroni": ("12/21", 6950),
        "Weighted\nBonferroni": ("11/21", 7151),
    },
}

colors = {
    "AnyRank": "#00A6FF",            # blue
    r"AnyRank$_0$": "#08306b",       # light blue
    r"AnyRank$_\infty$": "#1f77b4",  # light blue

    "Bonferroni": "#ff7f0e",         # orange
    "Weighted\nBonferroni": "#d62728",  # red
}

color_alpha = {
    "AnyRank": 1,
    r"AnyRank$_0$": 1,
    r"AnyRank$_\infty$": 1,
    "Bonferroni": 1,
    "Weighted\nBonferroni": 1,
}

# Canonical ordering for methods in the legend/bars
METHOD_ORDER = [
    "AnyRank",
    r"AnyRank$_0$",
    r"AnyRank$_\infty$",
    "Bonferroni",
    "Weighted\nBonferroni",
]


def frac_to_float(frac):
    a, b = frac.split("/")
    return int(a) / int(b)

def plot_tradeoff(data, name, save_dir="imglib/comparisons"):
    os.makedirs(save_dir, exist_ok=True)
    markers = {
        "PushCoaster": "o",
        "TurnMug": "X",
        "Aggregate": "s",
        "TurnMug (Nom.+OOD)": "D",
        "Aggregate (Nom.+OOD obj)": "P",
        "Aggregate (Nom.+OOD station)": "*",
        "RoboArena-4 (held-out)": "^",
        "RoboArena-7 (held-out)": "d",
        "RoboArena-4 (WM)": "v",
        "RoboArena-7 (WM)": "p",
    }

    method_offsets = {
        "AnyRank": -0.012,
        r"AnyRank$_0$": -0.006,
        r"AnyRank$_\infty$": 0.0,
        "Bonferroni": 0.006,
        "Weighted\nBonferroni": 0.012,
    }


    present_methods = [
        m for m in METHOD_ORDER
        if any(m in vals for vals in data.values())
    ]

    present_experiments = [
        e for e in data.keys()
        if any(m in data[e] for m in present_methods)
    ]

    n_exp = len(present_experiments)

    fig, axes = plt.subplots(
        1,
        n_exp,
        figsize=(3.0 * n_exp, 3.0),
        sharey=True,
    )

    if n_exp == 1:
        axes = [axes]

    for ax, setting in zip(axes, present_experiments):
        vals = data[setting]

        for method in present_methods:
            if method not in vals:
                continue

            rej, complexity = vals[method]

            x = complexity
            y = frac_to_float(rej)

            ax.scatter(
                x,
                y,
                s=120,
                marker="o",
                color=colors[method],
                alpha=color_alpha[method],
                edgecolor="black",
                linewidth=0.7,
                zorder=2,
                label=method,
            )

        ax.set_title(setting, fontsize=16)
        ax.set_xlabel("Sample Complexity")
        ax.set_ylim(0.0, 1.05)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Fraction of Hypotheses Rejected")
    method_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=9,
            markerfacecolor=colors[m],
            markeredgecolor="black",
            label=m,
            alpha=color_alpha[m],
        )
        for m in present_methods
    ]

    fig.legend(
        handles=method_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=len(present_methods),
    )

    plt.tight_layout()

    plt.savefig(
        f"{save_dir}/tradeoff_fraction_vs_sample_complexity_{name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved plot to {save_dir}/tradeoff_fraction_vs_sample_complexity_{name}.png"
    )

    plt.show()


def plot(data, name, save_dir="imglib/comparisons"):
    os.makedirs(save_dir, exist_ok=True)
    # For each setting, keep only methods achieving the maximum rejection rate
    filtered_data = {}
    for setting, vals in data.items():
        max_rej = max(frac_to_float(v[0]) for v in vals.values())
        filtered_data[setting] = {
            method: complexity
            for method, (rej, complexity) in vals.items()
            if frac_to_float(rej) == max_rej
        }

    # Methods present in at least one setting, in canonical order
    present_methods = [
        m for m in METHOD_ORDER
        if any(m in fd for fd in filtered_data.values())
    ]

    experiments = list(filtered_data.keys())
    x = np.arange(len(experiments))
    n_methods = len(present_methods)
    width = 0.16

    fig, ax = plt.subplots(figsize=(12, 4))

    labeled = set()
    for j, setting in enumerate(experiments):
        local_methods = [m for m in present_methods if filtered_data[setting].get(m) is not None]
        n_local = len(local_methods)
        for k, method in enumerate(local_methods):
            offset = (k - (n_local - 1) / 2) * width
            val = filtered_data[setting][method]

            label = method if method not in labeled else "_nolegend_"
            labeled.add(method)

            ax.bar(
                x[j] + offset,
                val,
                width,
                label=label,
                color=colors[method],
                alpha=color_alpha[method],
                edgecolor="black",
                linewidth=0.5,
            )

            ax.annotate(
                f"{val}",
                xy=(x[j] + offset, val),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Evaluations")
    ax.set_xlabel("Experiment")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        ncol=n_methods,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
    )

    plt.tight_layout()
    plt.savefig(
        f"{save_dir}/sample_complexity_{name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved plot to {save_dir}/sample_complexity_{name}.png")
    plt.show()

if __name__ == "__main__":
    # plot(lbm_data, "lbm")
    # plot(roboarena_data, "roboarena")

    # plot_tradeoff(lbm_data, "lbm")
    # plot_tradeoff(roboarena_data, "roboarena")
    plot_tradeoff(roboarena_data, "roboarena")