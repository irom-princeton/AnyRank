import numpy as np
import os
import copy
import tqdm
import tempfile
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from multitest.sequential_graphical import SequentialGraphicalTest
from multitest.sequential_graphical_evalue import SequentialGraphicalTest as ESequentialGraphicalTest
import pandas as pd
import ast
from multitest.run_graphical_test import ExperimentConfig, main
from pathlib import Path

########################################
#### Data Loading and Preprocessing ####
########################################
np.random.seed(42)

# SMALL_DATA_DIR = "/n/fs/irom-testing/multitest/data/lbm_data_small/LBM_DATA"
# FULL_DATA_DIR = "/n/fs/irom-testing/multitest/data/lbm_large/full_data"

# Directory containing the current Python file
ROOT = Path(__file__).resolve().parent.parent

SMALL_DATA_DIR = ROOT / "data" / "lbm_data_small" / "LBM_DATA"
FULL_DATA_DIR = ROOT / "data" / "lbm_large" / "full_data"

def load_small_data():
    subfolders = [f for f in os.listdir(SMALL_DATA_DIR) if os.path.isdir(os.path.join(SMALL_DATA_DIR, f))]
    data = {}
    means = {}
    for subfolder in subfolders:
        data[subfolder] = {}
        subfolder_path = os.path.join(SMALL_DATA_DIR, subfolder)
        for file in os.listdir(subfolder_path):
            exp_name = file.split(".")[0]
            file_path = os.path.join(subfolder_path, file)
            exp_data = np.load(file_path, allow_pickle=True)
            data[exp_name] = exp_data
            means[exp_name] = np.mean(exp_data)
    return data, means


def load_full_data(data_file):
    df = pd.read_csv(data_file)
    return df

def load_evals(df, panel, task, method, filename=None):
    evals = df[
        (df["Panel"] == panel)
        & (df["Task"] == task)
        & (df["Method"] == method)
    ]
    trials = ast.literal_eval(evals["Success/Failure"].iloc[0])
    # convert bool to int
    trials = np.array(trials).astype(int) # shuffle the trials to avoid any ordering effects
    # trials = np.random.permutation(trials)
    success_rate = np.mean(trials)
    tri_rank = evals["CLD_Letter"].iloc[0]
    return trials, success_rate, tri_rank

def experiment_A():
    # In distribution data 
    # Policies: Single task, LBM finetuned, LBM zeroshot
    # Task: PutKiwiInCenterOfTable
    # Panel: Fig2A_HW_Seen_Nominal
    hw_panel = "Fig2A_HW_Seen_Nominal"
    sim_panel = "Fig2A_Sim_Seen_Nominal"   
    
    tasks = ["PushCoasterToMug"]
    policies = ["Single Task", "LBM finetuned", "LBM zeroshot"]
    sim_means = {}
    real_means = {}

    policy_data = {}
    for policy in policies:
        trials, success_rate, tri_rank = load_evals(df, hw_panel, tasks[0], policy)
        policy_data[policy] = trials
        real_means[policy] = success_rate
        sim_trials, sim_success_rate, sim_tri_rank = load_evals(df, sim_panel, tasks[0], policy)
        sim_means[policy] = sim_success_rate
    return policy_data, real_means, sim_means

def experiment_C(id=True, ood=False, distshift=None):
    # In distribution data 
    # Policies: Single task, LBM finetuned, LBM zeroshot
    # Task: PutKiwiInCenterOfTable
    # Panel: Fig2A_HW_Seen_Nominal
    if id and not ood:
        hw_panels = ["Fig2A_HW_Seen_Nominal"]
        sim_panels = ["Fig2A_Sim_Seen_Nominal"] 
    elif not id and ood:
        hw_panels = ["Fig2B_HW_Seen_DistShift"]
        sim_panels = ["Fig2B_Sim_Seen_DistShift"]
    elif id and ood:
        hw_panels = ["Fig2A_HW_Seen_Nominal", "Fig2B_HW_Seen_DistShift"]
        sim_panels = ["Fig2A_Sim_Seen_Nominal", "Fig2B_Sim_Seen_DistShift"]  
    
    hw_tasks = ["Aggregate - Hardware"]
    if distshift == "distshift":
        dist_shift_hw_tasks = ["Aggregate - Distribution shift"]
    elif distshift == "novel":
        hw_panels = ["Fig2A_HW_Seen_Nominal", "Fig2B_HW_Seen_NovelStation"]
        dist_shift_hw_tasks = ["Aggregate - Novel station"]
    sim_tasks = ["Aggregate - Simulation"]
    base_policies = ["Single Task", "LBM finetuned", "LBM zeroshot"]
    sim_means = {}
    real_means = {}

    policy_data = {}
    for base_policy in base_policies:
        for hw_panel, sim_panel in zip(hw_panels, sim_panels):
            policy = f"{base_policy} ({hw_panel.split('_')[3]})"
            if hw_panel.split('_')[3] == "DistShift" or hw_panel.split('_')[3] == "NovelStation":
                try:
                    trials, success_rate, tri_rank = load_evals(df, hw_panel, dist_shift_hw_tasks[0], base_policy)
                except:
                    breakpoint()
            else:
                trials, success_rate, tri_rank = load_evals(df, hw_panel, hw_tasks[0], base_policy)
            
            policy_data[policy] = trials
            real_means[policy] = success_rate
            sim_trials, sim_success_rate, sim_tri_rank = load_evals(df, sim_panel, sim_tasks[0], base_policy)
            sim_means[policy] = sim_success_rate
            
    return policy_data, real_means, sim_means

def experiment_B(id = True, ood=False):
    # In distribution data 
    # Policies: Single task, LBM finetuned, LBM zeroshot
    # Task: TurnMugRightSideUp
    # Panel: Fig2A_HW_Seen_Nominal
    if id and not ood:
        hw_panels = ["Fig2A_HW_Seen_Nominal"]
        sim_panels = ["Fig2A_Sim_Seen_Nominal"] 
    elif not id and ood:
        hw_panels = ["Fig2B_HW_Seen_DistShift"]
        sim_panels = ["Fig2B_Sim_Seen_DistShift"]
    elif id and ood:
        hw_panels = ["Fig2A_HW_Seen_Nominal", "Fig2B_HW_Seen_DistShift"]
        sim_panels = ["Fig2A_Sim_Seen_Nominal", "Fig2B_Sim_Seen_DistShift"]

    tasks = ["TurnMugRightsideUp"]
    base_policies = ["Single Task", "LBM finetuned", "LBM zeroshot"]
    
    sim_means = {}
    real_means = {}
    policy_data = {}

    for base_policy in base_policies:
        for hw_panel, sim_panel in zip(hw_panels, sim_panels):
            policy = f"{base_policy} ({hw_panel.split('_')[3]})"
            trials, success_rate, tri_rank = load_evals(df, hw_panel, tasks[0], base_policy)
            policy_data[policy] = trials
            real_means[policy] = success_rate
            sim_trials, sim_success_rate, sim_tri_rank = load_evals(df, sim_panel, tasks[0], base_policy)
            sim_means[policy] = sim_success_rate
    return policy_data, real_means, sim_means

def run_graphical_experiment_A(beta=1):
    policy_data, real_means, sim_means = experiment_A()
    cfg = ExperimentConfig(alpha=0.1, beta=beta, results_dir='outputs/lbm_graphical_test/experiment_A')
    main(policy_data, sim_means=sim_means, real_means=real_means, bernoulli=True, cfg=cfg)

def run_graphical_experiment_B(id=True, ood=False, beta=1):
    policy_data, real_means, sim_means = experiment_B(id=id, ood=ood)
    cfg = ExperimentConfig(alpha=0.1, beta=beta, results_dir='outputs/lbm_graphical_test/experiment_B_id_{}_ood_{}'.format(id, ood))
    main(policy_data, sim_means=sim_means, real_means=real_means, bernoulli=True, cfg=cfg)

def run_graphical_experiment_C(id=True, ood=False, distshift=None, beta=1):
    policy_data, real_means, sim_means = experiment_C(id=id, ood=ood, distshift=distshift)
    # Print the number of evals per policy:
    for policy, trials in policy_data.items():
        print(f"Policy: {policy}, Number of evals: {len(trials)}, Success rate: {real_means[policy]:.2f}, Sim success rate: {sim_means[policy]:.2f}")
    cfg = ExperimentConfig(alpha=0.1, beta=beta, results_dir='outputs/lbm_graphical_test/experiment_C_id_{}_ood_{}_distshift_{}'.format(id, ood, distshift))
    main(policy_data, sim_means=sim_means, real_means=real_means, bernoulli=True, cfg=cfg)

    
if __name__ == "__main__":
    df = load_full_data(FULL_DATA_DIR / "Fig2.csv")    
    beta_range = [0, 1, 5, 10, 25, 50, 100]

    for beta in beta_range:
    ## Experiment A: In-distribution performance comparison
        run_graphical_experiment_A(beta=beta)

        ## Experiment B: In-distribution performance comparison
        run_graphical_experiment_B(id=True, ood=False, beta=beta) # ID only

        ## Experiment C: In-distribution performance comparison
        run_graphical_experiment_C(id=True, ood=False, distshift=None, beta=beta) # ID only

        ## Experiment B: Out-of-distribution performance comparison
        run_graphical_experiment_B(id=False, ood=True, beta=beta) # OOD only

        ## Experiment B: Out-of-distribution and nominal performance comparison
        run_graphical_experiment_B(id=True, ood=True, beta=beta) # ID and O

        ## Experiment C: Out-of-distribution objects and nominal performance comparison
        run_graphical_experiment_C(id=True, ood=True, distshift="distshift", beta=beta) # ID and OOD with distribution shift

        ## Experiment C: Out-of-distribution stations and nominal performance comparison 
        run_graphical_experiment_C(id=True, ood=True, distshift="novel", beta=beta) # ID and OOD with novel station
        