import argparse
import numpy as np
import os
import copy
import tqdm
import tempfile
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import pandas as pd
import ast
from multitest.individual_test import ExperimentConfig, main
from pathlib import Path

########################################
#### Data Loading and Preprocessing ####
########################################
np.random.seed(42)

# SMALL_DATA_DIR = "/n/fs/irom-testing/multitest/data/lbm_data_small/LBM_DATA"
# FULL_DATA_DIR = "/n/fs/irom-testing/multitest/data/lbm_large/full_data"

# Directory containing the current Python file
ROOT = Path(__file__).resolve().parent.parent
ALL_METHODS = ['fixed', 'bonferroni', 'weighted_bonferroni', 'graphical_active']
FULL_DATA_DIR = ROOT / "data" / "lbm_large" / "full_data"

def towel(wm, init_cond):
    tasks = ["FoldTowel"]
    policies = ["pi05", "pi0", "v2_30demo", "v2_half", "v2_miss2"][::-1]
    # dirname = os.path.dirname(__file__)
    dirname = "/n/fs/irom-testing/multitest/data/irom_data"

    policy_data = {}
    sim_means = {}
    real_means = {}
    
    for policy in policies:
        # load rw data
        real_path = os.path.join(dirname, f"scores_for_ranking/towel/real/{policy}.csv")
        if os.path.exists(real_path):
            real_data = pd.read_csv(real_path)
            policy_data[policy] = real_data["success"].astype(float)
            real_means[policy] = float(real_data["avg"][0])
        else:
            print(f"missing real data for {policy}")

        # load sim data
        sim_path = os.path.join(dirname, f"scores_for_ranking/towel/wm/{policy}.csv")
        if os.path.exists(sim_path):
            sim_data = pd.read_csv(sim_path)
            # select avg for specified sim condition
            sim_avgs = sim_data.iloc[:, 7]
            if init_cond.lower() == "old":
                sim_avgs = sim_avgs.iloc[0:2]
            elif init_cond.lower() == "new":
                sim_avgs = sim_avgs.iloc[2:4]
            elif init_cond.lower() == "all":
                sim_avgs = sim_avgs.iloc[4:6]
            else:
                print("invalid init cond type (choose from 'old', 'new', or 'all')")
            
            if wm.lower() == "base":
                sim_avgs = sim_avgs.iloc[0]
            elif wm.lower() == "specialist":
                sim_avgs = sim_avgs.iloc[1]
            else:
                print("invalid wm type (choose from 'base' or 'specialist')")
            
            sim_means[policy] = float(sim_avgs)
        else:
            print(f"missing sim data for {policy}")

    return policy_data, real_means, sim_means

def carrot(wm, init_cond):

    tasks = ["PickCarrot"]
    # policies = ["pi05", "v0_tip", "v0_miss_grasp", "v0_30demo", "v0_10demo"]
    policies = ["pi05", "v0_tip", "v0_miss_grasp", "v0_10demo"][::-1]
    # policies = ["pi05", "v0_tip", "v0_10demo"]
    dirname = "/n/fs/irom-testing/multitest/data/irom_data"

    policy_data = {}
    sim_means = {}
    real_means = {}
    
    for policy in policies:
        # load rw data
        real_path = os.path.join(dirname, f"scores_for_ranking/carrot/real/{policy}.csv")
        if os.path.exists(real_path):
            real_data = pd.read_csv(real_path)
            policy_data[policy] = real_data["success"].astype(float)
            real_means[policy] = float(real_data["avg"][0])
        else:
            print(f"missing real data for {policy}")

        # load sim data
        sim_path = os.path.join(dirname, f"scores_for_ranking/carrot/wm/{policy}.csv")
        if os.path.exists(sim_path):
            sim_data = pd.read_csv(sim_path)
            # select avg for specified sim condition
            sim_avgs = sim_data.iloc[:, 7]
            if init_cond.lower() == "old":
                sim_avgs = sim_avgs.iloc[0:2]
            elif init_cond.lower() == "new":
                sim_avgs = sim_avgs.iloc[2:4]
            elif init_cond.lower() == "all":
                sim_avgs = sim_avgs.iloc[4:6]
            else:
                print("invalid init cond type (choose from 'old', 'new', or 'all')")
            
            if wm.lower() == "base":
                sim_avgs = sim_avgs.iloc[0]
            elif wm.lower() == "specialist":
                sim_avgs = sim_avgs.iloc[1]
            else:
                print("invalid wm type (choose from 'base' or 'specialist')")
            
            sim_means[policy] = float(sim_avgs)
        else:
            print(f"missing sim data for {policy}")

    return policy_data, real_means, sim_means

def _results_dir(subfolder, name):
    base = 'irom_data_outputs'
    return os.path.join(base, subfolder, name) if subfolder else os.path.join(base, name)

def run_graphical_carrot(wm, init_cond, beta=1, labels=None, subfolder=None, graph_type="soft_masked", methods=None, allow_transitive=True):
    policy_data, real_means, sim_means = carrot(wm, init_cond)
    cfg = ExperimentConfig(alpha=0.1, beta=beta, results_dir=_results_dir(subfolder, f'carrot_wm_{wm}_init_{init_cond}'), graph_type=graph_type)
    main(policy_data, sim_means=sim_means, real_means=real_means, bernoulli=False, cfg=cfg, labels=labels, methods=methods, allow_transitive=allow_transitive)

def run_graphical_towel(wm, init_cond, beta=1, labels=None, subfolder=None, graph_type="soft_masked", methods=None, allow_transitive=True):
    policy_data, real_means, sim_means = towel(wm, init_cond)
    for policy, trials in policy_data.items():
        print(f"Policy: {policy}, Number of evals: {len(trials)}, Success rate: {real_means[policy]:.2f}, Sim success rate: {sim_means[policy]:.2f}")
    cfg = ExperimentConfig(alpha=0.1, beta=beta, results_dir=_results_dir(subfolder, f'towel_wm_{wm}_init_{init_cond}'), graph_type=graph_type)
    main(policy_data, sim_means=sim_means, real_means=real_means, bernoulli=False, cfg=cfg, labels=labels, methods=methods, allow_transitive=allow_transitive)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subfolder", type=str, default=None, help="Optional subfolder under outputs/lbm_graphical_test/")
    parser.add_argument("--graph_type", type=str, default="fully_connected",
                        choices=["soft_masked", "fully_connected"],
                        help="Graph type for alpha transfer weights")
    parser.add_argument("--transitive", type=bool, default=True,
                        choices=[True, False],
                        help="Whether to use transitive policy evaluation")
    parser.add_argument("--methods", type=str, nargs='+', default=ALL_METHODS,
                        choices=ALL_METHODS,
                        metavar="METHOD",
                        help=f"Methods to run (default: all). Choices: {', '.join(ALL_METHODS)}")

    args = parser.parse_args()

    methods = args.methods  # None means all

    beta_range = [0, 10]
    wm_types_towel = ["base"]
    init_conds_towel = ["new"]

    wm_types_carrot = ["specialist"]
    init_conds_carrot = ["new"]

    plot_policy_names = {
        "pi05":          "pi0.5",
        "pi0":           "pi0",
        "v2_30demo":     "DP (30)",
        "v2_half":       "DP (half)",
        "v2_miss2":      "DP (miss2)",
        "v0_tip":        "DP (tip)",
        "v0_miss_grasp": "DP (miss)",
        "v0_30demo":     "DP (30)",
        "v0_10demo":     "DP (10)",
    }

    for beta in beta_range:
        for wm in wm_types_towel:
            for init_cond in init_conds_towel:
                # run_graphical_carrot(wm=wm, init_cond=init_cond, beta=beta, labels=plot_policy_names, subfolder=args.subfolder, graph_type=args.graph_type, methods=methods, allow_transitive=args.transitive)
                run_graphical_towel(wm=wm, init_cond=init_cond, beta=beta, labels=plot_policy_names, subfolder=args.subfolder, graph_type=args.graph_type, methods=methods, allow_transitive=args.transitive)

    for beta in beta_range:
        for wm in wm_types_carrot:
            for init_cond in init_conds_carrot:
                run_graphical_carrot(wm=wm, init_cond=init_cond, beta=beta, labels=plot_policy_names, subfolder=args.subfolder, graph_type=args.graph_type, methods=methods, allow_transitive=args.transitive)
    
    # for beta in beta_range:
    #     run_graphical_carrot(wm="specialist", init_cond="all", beta=beta, labels=plot_policy_names, subfolder=args.subfolder, graph_type=args.graph_type, methods=methods, allow_transitive=args.transitive)

    

