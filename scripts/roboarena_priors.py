import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

plt.rcParams['font.family'] = 'DejaVu Sans Mono'

output_folder = "/n/fs/irom-testing/multitest/imglib/roboarena"  # Update this to your desired output path
# srs = {
#         "pi05": {"real": 1, "base": 0.970, "specialist": 0.909},
#         "v0_tip": {"real": 0.560, "base": 0.621, "specialist": 0.258},
#         "v0_miss_grasp": {"real": 0.413, "base": 0.700, "specialist": 0.455},
#         "v0_30demo": {"real": 0.493, "base": 0.667, "specialist": 0.727},
#         "v0_10demo": {"real": 0.060, "base": 0.303, "specialist": 0.789},
# }

# # subset of policies for plotting
# srs = {
#         "pi05": {"real": 1, "specialist": 0.909},
#         "v0_tip": {"real": 0.560, "specialist": 0.258},
#         "v0_miss_grasp": {"real": 0.413,  "specialist": 0.455},
#         "v0_10demo": {"real": 0.060, "specialist": 0.789},
# }

plot_policy_names = {
        "paligemma_binning_droid":        "PG-Bin",
        "paligemma_diffusion_droid":      "PG-Diff",
        "paligemma_vq_droid":             "PG-VQ",
        "paligemma_fast_droid":           "PG-Fast",
        "paligemma_fast_specialist_droid": "PG-Fast-Spec",
        "pi0_droid":                      r"$\pi_0$",
        "pi0_fast_droid":                 r"$\pi_0$-Fast",
    }

srs = {
    "paligemma_binning_droid":         {"real": 0.0479, "heldout": 0.0200, "wm": 0.1500},
    "paligemma_diffusion_droid":       {"real": 0.3500, "heldout": 0.4575, "wm": 0.4562},
    # "paligemma_vq_droid":              {"real": 0.4091, "heldout": 0.4575, "wm": 0.7250},
    # "paligemma_fast_droid":            {"real": 0.4158, "heldout": 0.4050, "wm": 0.5062},
    # "paligemma_fast_specialist_droid": {"real": 0.4266, "heldout": 0.4525, "wm": 0.5875},
    "pi0_droid":                       {"real": 0.4355, "heldout": 0.4760, "wm": 0.6000},
    "pi0_fast_droid":                  {"real": 0.4587, "heldout": 0.5075, "wm": 0.4250},
}
# Colors for each policy type
policy_colors = {
    "paligemma_binning_droid":         "#4878CF",  # blue
    "paligemma_diffusion_droid":       "#FD9800",  # orange
    "paligemma_vq_droid":              "#79B353",  # green
    "paligemma_fast_droid":            "#E48888",  # red
    "paligemma_fast_specialist_droid": "#68C3A9",  # teal
    "pi0_droid":                       "#796ECB",  # purple
    "pi0_fast_droid":                  "#B3CD72",  # yellow-green
}

sims = ["heldout", "wm"]
wm_markers = {
    "heldout": "o",
    "wm": "s"
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
    "heldout": "grey",   # orange
    "wm": "black",  # blue
}

# Grid settings
grid_color = '#CCCCCC'  # light grey (adjustable)
grid_alpha = 0.7

# Font sizes
label_fontsize = 15
legend_fontsize = 15
tick_fontsize = 15

def plot(savefile, show_legend=True):
    fig, ax = plt.subplots(figsize=(6, 6))

    real_vals = [v["real"] for v in srs.values()]

    corr_vals = {}
    for eval_type in sims:
        marker = wm_markers[eval_type]
        sim_vals = [v[eval_type] for v in srs.values()]
        real_arr = np.array(real_vals)
        corr_vals[eval_type] = np.corrcoef(real_arr, sim_vals)[0, 1]

        # line of best fit: sim = m * real + b
        m, b = np.polyfit(real_arr, sim_vals, 1)

        x_fit = np.linspace(real_arr.min(), real_arr.max(), 100)
        y_fit = m * x_fit + b
        ax.plot(
            x_fit,
            y_fit,
            linestyle="--",
            linewidth=4,
            color=line_colors[eval_type],
            alpha=1,
        )

        for policy, outcomes in srs.items():
            color = policy_colors[policy]
            r = outcomes.get("real")
            e = outcomes.get(eval_type)
            if r is not None and e is not None:
                ax.scatter(r, e, c=color, marker=marker, s=marker_size,
                           edgecolors=marker_edgecolor, linewidths=marker_edgewidth,
                           alpha=marker_alpha)
    
    if "paligemma_binning_droid" in srs:
        min_real = min(real_vals)
        max_real = max(real_vals)
        xmin = np.floor((min_real - 0.05) / 0.05) * 0.05
        xmax = np.ceil((max_real + 0.05) / 0.05) * 0.05 
    else:
        xmin = 0.0
        xmax = 0.5
    ax.set_xlabel("Real SR", fontsize=label_fontsize)
    ax.set_ylabel("Simulation Prior", fontsize=label_fontsize)
    ax.set_xlim(xmin, xmax)
    # ax.xaxis.set_major_locator(MultipleLocator(0.05))
    # ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_ylim(-0.05, 1.05)
    # #ax.set_xticks(np.arange(xmin, xmax, 0.05))
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    #ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(True, color=grid_color, alpha=grid_alpha, linestyle='-', linewidth=0.5)

    if show_legend:
        from matplotlib.patches import Patch
        color_handles = [
            mpatches.Patch(facecolor=policy_colors[p], edgecolor='black', label=plot_policy_names[p])
            for p in srs
        ]
        # shape_handles = [
        #     mlines.Line2D([], [], color='grey',
        #                   marker=wm_markers[et], markersize=10,
        #                   markeredgecolor='black', linestyle='None',
        #                   label=f"{et}: r = {corr_vals[et]:.2f}")
        #     for et in wm_markers
        # ]
        shape_handles = [
            mlines.Line2D(
                [], [],
                color=line_colors[et],      # line color
                marker=wm_markers[et],      # point shape
                markersize=10,
                markeredgecolor='black',
                linestyle='--',
                linewidth=3,
                label=f"{et}: r = {corr_vals[et]:.2f}"
            )
            for et in sims
        ]
        all_handles = (
            [Patch(visible=False, label='— policy —')] + color_handles +
            [Patch(visible=False, label='— eval type —')] + shape_handles
        )
        fig_legend, ax_legend = plt.subplots()
        ax_legend.axis('off')
        legend = ax_legend.legend(
            handles=all_handles,
            fontsize=legend_fontsize,
            loc='center',
            framealpha=0.9,
        )
        fig_legend.canvas.draw()
        bbox = legend.get_window_extent().transformed(
            fig_legend.dpi_scale_trans.inverted()
        )
        fig_legend.savefig(f"{output_folder}/{savefile}_legend.png", dpi=150, bbox_inches=bbox)
        plt.close(fig_legend)

    plt.savefig(f"{output_folder}/{savefile}.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    plot("roboarena_4policies")