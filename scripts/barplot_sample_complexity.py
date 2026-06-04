# Compare sample complexity of methods with the same rejection rate, for each setting. This is a "fair" comparison since we only compare methods that achieve the same performance in terms of rejections.
import matplotlib.pyplot as plt
import numpy as np
import os

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

roboarena_data = {
    "RoboArena-4 (held-out)": {
        "AnyRank": ("6/6", 1366),
        r"AnyRank$_0$": ("6/6", 1380),
        r"AnyRank$_\infty$": ("6/6", 1317),
        "Bonferroni": ("6/6", 2599),
        "Weighted\nBonferroni": ("6/6", 2912),
    },
    "RoboArena-7 (held-out)": {
        r"AnyRank$_0$": ("12/21", 6929),
        "Bonferroni": ("12/21", 6950),
    },
    "RoboArena-4 (WM)": {
        "AnyRank": ("6/6", 2445),
        r"AnyRank$_0$": ("6/6", 1380),
        r"AnyRank$_\infty$": ("6/6", 1317),
        "Bonferroni": ("6/6", 2599),
    },
    "RoboArena-7 (WM)": {
        r"AnyRank$_0$": ("12/21", 6929),
        "Bonferroni": ("12/21", 6950),
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
    "Bonferroni": 0.7,
    "Weighted\nBonferroni": 0.7,
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
    plot(lbm_data, "lbm")
    plot(roboarena_data, "roboarena")
