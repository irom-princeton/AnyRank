import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans Mono'

output_folder = "/n/fs/irom-testing/multitest/imglib/ranking"  # Update this to your desired output path
srs = {
        "pi05": {"real": 1, "base": 0.970, "specialist": 0.909},
        "v0_tip": {"real": 0.560, "base": 0.621, "specialist": 0.258},
        "v0_miss_grasp": {"real": 0.413, "base": 0.700, "specialist": 0.455},
        "v0_30demo": {"real": 0.493, "base": 0.667, "specialist": 0.727},
        "v0_10demo": {"real": 0.060, "base": 0.303, "specialist": 0.789},
}

# subset of policies for plotting carrot
carrot_srs = {
        "pi05": {"real": 1, "specialist": 0.909},
        "v0_tip": {"real": 0.560, "specialist": 0.258},
        "v0_miss_grasp": {"real": 0.413,  "specialist": 0.455},
        "v0_10demo": {"real": 0.060, "specialist": 0.789},
}

carrot_policy_colors = {
    "pi05": "#FD9800",  # orange
    "v0_tip": "#79B353",  # green
    "v0_miss_grasp": "#E48888",  # red
    "v0_30demo": "#B3CD72",  # blue
    "v0_10demo": "#68C3A9",  # teal
}

# subset of policies for plotting fold towel
towel_srs = {"pi05": {"real": 0.975, "base": 0.772, "specialist": 0.919},
        "pi0": {"real": 0.633, "base": 0.728, "specialist": 0.75},
        "v2_30demo": {"real": 0.457, "base": 0.669, "specialist": 0.882}, 
        "v2_half": {"real": 0.267, "base": 0.471, "specialist": 0.588},
        "v2_miss2": {"real": 0.040, "base": 0.081, "specialist": 0.338},
}

towel_policy_colors = {
    "pi05": "#FD9800",  # orange
    "pi0": "#79B353",  # green
    "v2_30demo": "#B3CD72",  # blue
    "v2_half": "#68C3A9",  # teal
    "v2_miss2": "#E48888",  # red
}
# Colors for each policy type


wm_markers = {
    "base": "o",
    "specialist": "s"
}

# Marker settings
marker_size = 400
marker_edgewidth = 2
marker_edgecolor = 'black'
marker_alpha = 0.8

# Best fit line settings
line_alpha = 0.8
line_width = 4
line_colors = {
    "base": "#D0A011",   # orange
    "specialist": "#796ECB",  # blue
}

# Grid settings
grid_color = '#CCCCCC'  # light grey (adjustable)
grid_alpha = 0.7

# Font sizes
label_fontsize = 15
legend_fontsize = 15
tick_fontsize = 15

def plot(srs, eval_types, policy_colors, exp_name, show_legend=True):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')

    corr_vals = {}

    for eval_type in eval_types:
        marker = wm_markers[eval_type]
        r_list, e_list = [], []

        for policy, outcomes in srs.items():
            color = policy_colors[policy]
            r = outcomes.get("real")
            e = outcomes.get(eval_type)
            if r is not None and e is not None:
                ax.scatter(r, e, c=color, marker=marker, s=marker_size,
                           edgecolors=marker_edgecolor, linewidths=marker_edgewidth,
                           alpha=marker_alpha)
                r_list.append(r)
                e_list.append(e)

        if len(r_list) >= 2:
            real_arr = np.array(r_list)
            sim_arr = np.array(e_list)
            corr_vals[eval_type] = np.corrcoef(real_arr, sim_arr)[0, 1]
            m, b = np.polyfit(real_arr, sim_arr, 1)
            x_fit = np.linspace(real_arr.min(), real_arr.max(), 100)
            ax.plot(x_fit, m * x_fit + b, linestyle="--", linewidth=4,
                    color=line_colors[eval_type], alpha=1)

    ax.set_xlabel("Real SR", fontsize=label_fontsize)
    ax.set_ylabel("WM SR", fontsize=label_fontsize)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(True, color=grid_color, alpha=grid_alpha, linestyle='-', linewidth=0.5)
    if show_legend:
        from matplotlib.patches import Patch
        color_handles = [
            mpatches.Patch(facecolor=policy_colors[p], edgecolor='black', label=p)
            for p in srs
        ]

        if len(eval_types) > 1:
            shape_handles = [
                mlines.Line2D(
                    [], [],
                    color=line_colors[et],
                    marker=wm_markers[et],
                    markersize=10,
                    markeredgecolor='black',
                    linestyle='--',
                    linewidth=3,
                    label=f"{et}: r = {corr_vals[et]:.2f}"
                )
                for et in eval_types
            ]
            all_handles = (
                [Patch(visible=False, label='— policy —')] + color_handles +
                [Patch(visible=False, label='— eval type —')] + shape_handles
            )
        else:
            et = eval_types[0]
            shape_handles = [
                mlines.Line2D(
                    [], [],
                    color=line_colors[et],
                    marker=wm_markers[et],
                    markersize=10,
                    markeredgecolor='black',
                    linestyle='--',
                    linewidth=3,
                    label=f"r = {corr_vals[et]:.2f}"
                )
            ] if et in corr_vals else []
            all_handles = (
                [Patch(visible=False, label='— policy —')] + color_handles +
                ([Patch(visible=False, label='— eval type —')] + shape_handles if shape_handles else [])
            )

        # --- separate legend figure ---
        fig_legend, ax_legend = plt.subplots()
        ax_legend.axis('off')
        legend = ax_legend.legend(
            handles=all_handles,
            fontsize=legend_fontsize,
            loc='center',
            framealpha=0.9,
        )

        # Crop the figure tightly to just the legend box
        fig_legend.canvas.draw()
        bbox = legend.get_window_extent().transformed(
            fig_legend.dpi_scale_trans.inverted()
        )
        fig_legend.savefig(f"{output_folder}/{exp_name}_legend.png", dpi=150, bbox_inches=bbox)
        plt.close(fig_legend)

    plt.savefig(f"{output_folder}/corr_{exp_name}.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    plot(carrot_srs, ["specialist"], carrot_policy_colors, "carrot", show_legend=True)
    plot(towel_srs, ["base"], towel_policy_colors, "towel", show_legend=True)