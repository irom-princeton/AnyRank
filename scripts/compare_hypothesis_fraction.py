import matplotlib.pyplot as plt
import numpy as np

lbm_experiments = [
    "PushCoaster",
    "TurnMug",
    "Aggregate",
    "TurnMug\n+OOD",
    "Aggregate\n+OOD Obj",
    "Aggregate\n+OOD Station",
]

lbm_data = {
    "AnyRank": [
        "2/3",
        "3/3",
        "3/3",
        "8/15",
        "15/15",
        "15/15",
    ],
    r"AnyRank$_0$": [
        "2/3",
        "3/3",
        "3/3",
        "14/15",
        "15/15",
        "15/15",
    ],
    r"AnyRank$_\infty$": [
        "2/3",
        "1/3",
        "3/3",
        "0/15",
        "9/15",
        "2/15",
    ],
    "Bonferroni": [
        "2/3",
        "2/3",
        "3/3",
        "11/15",
        "11/15",
        "10/15",
    ],
    "Weighted\nBonferroni": [
        "2/3",
        "2/3",
        "3/3",
        "10/15",
        "6/15",
        "8/15",
    ],
}

roboarena_experiments = [
    "RoboArena-4\n(Held-out)",
    "RoboArena-7\n(Held-out)",
    "RoboArena-4\n(WM)",
    "RoboArena-7\n(WM)",
]

roboarena_data = {
    "AnyRank": [
        "6/6",
        "10/21",
        "6/6",
        "10/21",
    ],
    r"AnyRank$_0$": [
        "6/6",
        "12/21",
        "6/6",
        "12/21",
    ],
    r"AnyRank$_\infty$": [
        "6/6",
        "9/21",
        "6/6",
        "9/21",
    ],
    "Bonferroni": [
        "6/6",
        "12/21",
        "6/6",
        "12/21",
    ],
    "Weighted\nBonferroni": [
        "6/6",
        "11/21",
        "5/6",
        "11/21",
    ],
}

colors = {
    "AnyRank": "#00A6FF",            # blue
    r"AnyRank$_0$": "#08306b",       # light blue
    r"AnyRank$_\infty$": "#1f77b4",  # light blue

    "Bonferroni": "#ff7f0e",         # orange
    "Weighted\nBonferroni": "#d62728",  # red
}

color_alpha = {
    "AnyRank": 1,            # blue
    r"AnyRank$_0$": 1,       # light blue
    r"AnyRank$_\infty$": 1,  # light blue

    "Bonferroni": 0.7,         # orange
    "Weighted\nBonferroni": 0.7,  # red
}

def latex_frac(frac):
    num, den = frac.split("/")
    return rf"$\frac{{{num}}}{{{den}}}$"

def frac_to_float(s):
    num, den = s.split("/")
    return int(num) / int(den)


def plot(fractions, experiments, exp_name, save_dir="imglib/comparisons"):
    data = {
        method: [frac_to_float(x) for x in vals]
        for method, vals in fractions.items()
    }

    fig, ax = plt.subplots(figsize=(12, 4))

    x = np.arange(len(experiments))

    n_methods = len(data)
    width = 0.16

    for i, (method, vals) in enumerate(data.items()):

        offset = (i - (n_methods - 1) / 2) * width

        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=method,
            color=colors[method],
            alpha=color_alpha[method],
            edgecolor="black",
            linewidth=0.5,
        )

        # Add exact fractions above bars
        for bar, frac_label in zip(bars, fractions[method]):
            ax.annotate(
                latex_frac(frac_label),
                xy=(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                ),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------

    ax.set_ylabel("Fraction of Hypotheses Decided")
    ax.set_xlabel("Experiment")

    ax.set_xticks(x)
    ax.set_xticklabels(experiments)

    ax.set_ylim(0, 1.12)

    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        ncol=len(fractions),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
    )

    plt.tight_layout()

    plt.savefig(
        f"{save_dir}/fraction_hypotheses_rejected_{exp_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved plot to {save_dir}/fraction_hypotheses_rejected_{exp_name}.png")
    plt.show()

if __name__ == "__main__":
    plot(lbm_data, lbm_experiments, "lbm")
    plot(roboarena_data, roboarena_experiments, "roboarena")
    # plot(irom_experiments_data, irom_experiments, "irom")