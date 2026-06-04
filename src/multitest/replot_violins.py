"""
Replot violin plots from saved violin data.

Usage examples:
    # Replot a specific run by pointing at its meta JSON:
    python -m multitest.replot_violins --meta-json outputs/roboarena/violin_data/violin_N5000_n20_alpha0.1_beta1.0_meta.json

    # Replot all runs found in a results directory (optionally filter by parameters):
    python -m multitest.replot_violins --results-dir outputs/roboarena
    python -m multitest.replot_violins --results-dir outputs/roboarena --beta 1.0 --mode success_rate
"""

import argparse
import json
import os

import numpy as np

from multitest.plot_cld import make_violin_plots_from_ttds


def load_and_replot(meta_path, output_dir=None, mode=None, labels=None):
    progress_path = meta_path.replace("_meta.json", "_progress.npz")
    if not os.path.isfile(progress_path):
        raise FileNotFoundError(f"Expected progress file not found: {progress_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    progress_npz = np.load(progress_path, allow_pickle=True)
    policies = meta["policies"]

    progress_array_dict = {name: progress_npz[name] for name in policies}
    ttd_dicts = {
        method: {(pi, pj): ttd for pi, pj, ttd in entries}
        for method, entries in meta["ttd_dicts"].items()
    }
    max_sample_size_per_model = {name: len(progress_array_dict[name]) for name in policies}

    if output_dir is None:
        violin_dir = os.path.dirname(meta_path)
        results_dir = os.path.dirname(violin_dir)
        base = os.path.basename(meta_path)
        beta_part = base.split("_beta", 1)[1].replace("_meta.json", "")
        output_dir = os.path.join(results_dir, "cld_violins", f"beta{beta_part}")

    if mode is None:
        mode = "task_progress"

    plot_labels = labels if labels is not None else policies

    make_violin_plots_from_ttds(
        progress_array_dict=progress_array_dict,
        ttd_dicts=ttd_dicts,
        policies=policies,
        labels=plot_labels,
        max_sample_size_per_model=max_sample_size_per_model,
        mode=mode,
        output_dir=output_dir,
    )
    print(f"Violin plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Replot violin plots from saved violin data.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--meta-json", type=str,
                       help="Path to a specific *_meta.json file produced by run_graphical_test")
    group.add_argument("--results-dir", type=str,
                       help="Results directory containing a violin_data/ subdirectory")

    parser.add_argument("--Nmax", type=int, default=None,
                        help="Filter by Nmax (only used with --results-dir)")
    parser.add_argument("--n-prior", type=int, default=None,
                        help="Filter by n_prior (only used with --results-dir)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Filter by alpha (only used with --results-dir)")
    parser.add_argument("--beta", type=float, default=None,
                        help="Filter by beta (only used with --results-dir)")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["task_progress", "success_rate"],
                        help="Violin mode: 'task_progress' (default) or 'success_rate'")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory for violin plots")
    parser.add_argument("--labels", type=str, default=None,
                        help="JSON array of display labels matching policy order, e.g. '[\"A\",\"B\"]'")

    args = parser.parse_args()
    labels = json.loads(args.labels) if args.labels else None

    if args.meta_json:
        load_and_replot(args.meta_json, output_dir=args.output_dir, mode=args.mode, labels=labels)
        return

    violin_dir = os.path.join(args.results_dir, "violin_data")
    if not os.path.isdir(violin_dir):
        raise FileNotFoundError(f"No violin_data directory found at {violin_dir}")

    meta_files = sorted(f for f in os.listdir(violin_dir) if f.endswith("_meta.json"))

    if args.Nmax is not None:
        meta_files = [f for f in meta_files if f"_N{args.Nmax}_" in f]
    if args.n_prior is not None:
        meta_files = [f for f in meta_files if f"_n{args.n_prior}_" in f]
    if args.alpha is not None:
        meta_files = [f for f in meta_files if f"_alpha{args.alpha}_" in f]
    if args.beta is not None:
        meta_files = [f for f in meta_files if f"_beta{args.beta}" in f]

    if not meta_files:
        raise FileNotFoundError(f"No matching *_meta.json files found in {violin_dir}")

    for meta_file in meta_files:
        meta_path = os.path.join(violin_dir, meta_file)
        print(f"\nReplotting from {meta_path}")
        load_and_replot(meta_path, output_dir=args.output_dir, mode=args.mode, labels=labels)


if __name__ == "__main__":
    main()
