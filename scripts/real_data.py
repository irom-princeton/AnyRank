import pandas as pd
import numpy as np
import os

# IROM Carrot data:
def save_carrot_data(data_dir="carrot_policy_data"):
    os.makedirs(data_dir, exist_ok=True)
    tasks = ["PickCarrot"]
    policies = ["pi05", "v0_tip", "v0_miss_grasp", "v0_10demo"][::-1]
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

    # Save policy data:
    for policy, data in policy_data.items():
        print(f"Policy: {policy}, Data: {data}")
        data.to_csv(f"{data_dir}/policy_{policy}_data.csv", index=False)

def save_towel_data(data_dir="towel_policy_data"):
    os.makedirs(data_dir, exist_ok=True)
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
        
    # Save policy data:
    for policy, data in policy_data.items():
        print(f"Policy: {policy}, Data: {data}")
        data.to_csv(f"{data_dir}/policy_{policy}_data.csv", index=False)

if __name__ == "__main__":
    
    # IROM
    save_towel_data()
    save_carrot_data()