"""
Plot total sample complexity vs beta for each testing method.

Reads sample_complexity txt files from:
  <results_dir>/sample_complexity/sample_complexity_N<Nmax>_n<n>_alpha<alpha>_beta<beta>.txt

Each file has lines:
  Policy <i>: Graphical=<v>, EGraphical=<v>, EGraphicalActive=<v>,
              GraphicalActive=<v>, Bonferroni=<v>, Fixed=<v>, Weighted_Bonferroni=<v>

Generates:
  - One plot for p-value methods (total sample complexity vs beta)
  - One plot for e-value methods (total sample complexity vs beta)
  - One plot per policy (all methods, per-policy sample complexity vs beta)

Usage:
  python plot_sample_complexity_vs_beta.py <results_dir> [--Nmax 500] [--n 20] [--alpha 0.1]
"""

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt


PVALUE_METHODS = ["Graphical", "GraphicalActive", "Bonferroni", "Fixed", "Weighted_Bonferroni"]
EVALUE_METHODS = ["EGraphical", "EGraphicalActive"]
ALL_METHODS    = PVALUE_METHODS + EVALUE_METHODS

METHOD_LABELS = {
    "Graphical":           "Graphical (p-value)",
    "EGraphical":          "E-value graphical",
    "EGraphicalActive":    "E-value graphical (active)",
    "GraphicalActive":     "Graphical active (p-value)",
    "Bonferroni":          "Bonferroni",
    "Fixed":               "Fixed sequence",
    "Weighted_Bonferroni": "Weighted Bonferroni",
}

METHOD_COLORS = {
    "Graphical":           "tab:blue",
    "EGraphical":          "tab:orange",
    "EGraphicalActive":    "tab:green",
    "GraphicalActive":     "tab:red",
    "Bonferroni":          "tab:purple",
    "Fixed":               "tab:brown",
    "Weighted_Bonferroni": "tab:pink",
}


def parse_sample_complexity_file(path):
    """Return dict {method: [value_per_policy, ...]} from one txt file."""
    data = {m: [] for m in ALL_METHODS}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("Policy"):
                continue
            for method in ALL_METHODS:
                m = re.search(rf"{method}=([\d.]+)", line)
                if m:
                    data[method].append(float(m.group(1)))
    return data


def extract_beta(filename):
    m = re.search(r"_beta([\d.]+)\.txt$", filename)
    if m is None:
        raise ValueError(f"Cannot extract beta from filename: {filename}")
    return float(m.group(1))


def find_files(results_dir, Nmax, n_prior, alpha):
    sc_dir = os.path.join(results_dir, "sample_complexity")
    if not os.path.isdir(sc_dir):
        raise FileNotFoundError(f"No sample_complexity directory found in {results_dir}")
    pattern = re.compile(
        rf"^sample_complexity_N{Nmax}_n{n_prior}_alpha{alpha}_beta[\d.]+\.txt$"
    )
    matches = []
    for fname in os.listdir(sc_dir):
        if pattern.match(fname):
            matches.append((extract_beta(fname), os.path.join(sc_dir, fname)))
    if not matches:
        raise FileNotFoundError(
            f"No files matching N={Nmax}, n={n_prior}, alpha={alpha} found in {sc_dir}"
        )
    return sorted(matches, key=lambda x: x[0])


def load_data(results_dir, Nmax, n_prior, alpha):
    """
    Returns:
      betas      : sorted list of beta values
      per_policy : {method: np.array shape (n_betas, n_policies)}
    """
    files = find_files(results_dir, Nmax, n_prior, alpha)
    betas = [b for b, _ in files]
    per_policy = {m: [] for m in ALL_METHODS}

    for _, fpath in files:
        file_data = parse_sample_complexity_file(fpath)
        for method in ALL_METHODS:
            per_policy[method].append(file_data[method])

    for method in ALL_METHODS:
        per_policy[method] = np.array(per_policy[method])  # (n_betas, n_policies)

    return betas, per_policy


def _make_plot(betas, totals, methods, title, save_path):
    """Plot total sample complexity vs beta for the given set of methods."""
    fig, ax = plt.subplots(figsize=(8, 5))
    betas_arr = np.array(betas)
    for method in methods:
        ax.plot(betas_arr, totals[method], marker="o",
                label=METHOD_LABELS[method], color=METHOD_COLORS[method])
    ax.set_xlabel(r"$\beta$", fontsize=13)
    ax.set_ylabel("Total sample complexity (sum over policies)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_all(betas, per_policy, results_dir, Nmax, n_prior, alpha):
    n_policies = per_policy[ALL_METHODS[0]].shape[1]
    totals = {m: per_policy[m].sum(axis=1) for m in ALL_METHODS}
    tag = f"N{Nmax}_n{n_prior}_alpha{alpha}"

    # P-value methods plot
    _make_plot(
        betas, totals, PVALUE_METHODS,
        title=f"Sample complexity vs $\\beta$ — p-value methods\n(N={Nmax}, n={n_prior}, $\\alpha$={alpha})",
        save_path=os.path.join(results_dir, f"sample_complexity_vs_beta_pvalue_{tag}.png"),
    )

    # E-value methods plot
    _make_plot(
        betas, totals, EVALUE_METHODS,
        title=f"Sample complexity vs $\\beta$ — e-value methods\n(N={Nmax}, n={n_prior}, $\\alpha$={alpha})",
        save_path=os.path.join(results_dir, f"sample_complexity_vs_beta_evalue_{tag}.png"),
    )

    # Per-policy plots (all methods)
    betas_arr = np.array(betas)
    for i in range(n_policies):
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in ALL_METHODS:
            ax.plot(betas_arr, per_policy[method][:, i], marker="o",
                    label=METHOD_LABELS[method], color=METHOD_COLORS[method])
        ax.set_xlabel(r"$\beta$", fontsize=13)
        ax.set_ylabel("Sample complexity", fontsize=12)
        ax.set_title(
            f"Sample complexity vs $\\beta$ — Policy {i}\n(N={Nmax}, n={n_prior}, $\\alpha$={alpha})",
            fontsize=13,
        )
        ax.legend(fontsize=10, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        save_path = os.path.join(results_dir, f"sample_complexity_vs_beta_policy{i}_{tag}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", help="Directory containing the sample_complexity subdirectory")
    parser.add_argument("--Nmax",  type=int,   default=500,  help="Nmax used in filenames (default: 500)")
    parser.add_argument("--n",     type=int,   default=20,   help="n_prior used in filenames (default: 20)")
    parser.add_argument("--alpha", type=float, default=0.1,  help="alpha used in filenames (default: 0.1)")
    args = parser.parse_args()

    alpha_str = str(args.alpha).rstrip("0").rstrip(".")
    betas, per_policy = load_data(args.results_dir, args.Nmax, args.n, alpha_str)

    print(f"Found {len(betas)} beta values: {betas}")
    n_policies = per_policy[ALL_METHODS[0]].shape[1]
    print(f"Found {n_policies} policies per file")

    plot_all(betas, per_policy, args.results_dir, args.Nmax, args.n, alpha_str)


if __name__ == "__main__":
    main()
