from datasets import load_dataset, Dataset
from huggingface_hub import list_repo_files, hf_hub_download
import yaml
from collections import defaultdict
import os 
import pandas as pd
import numpy as np

# ===================================================== #
## DIRECTORIES AND PATHS
script_dir = os.path.dirname(os.path.abspath(__file__))
agg_out_path = os.path.join(script_dir, "policy_stats.csv")
roboarena_data_dump = "RoboArena/DataDump_08-05-2025"

# ===================================================== #
## Load and parse all evaluation session metadata
def load_data():
    # 1) List all files in the dataset repo
    files = list_repo_files(roboarena_data_dump, repo_type="dataset")

    # 2) Keep only per-session metadata.yaml files
    meta_files = [
        f for f in files
        if f.startswith("evaluation_sessions/") and f.endswith("/metadata.yaml")
    ]

    # 3) Download + parse all metadata.yaml files into rows
    rows = []
    for f in meta_files:
        session_id = f.split("/")[1]  # evaluation_sessions/<session_id>/metadata.yaml
        local_path = hf_hub_download(roboarena_data_dump, filename=f, repo_type="dataset")
        with open(local_path, "r") as fh:
            meta = yaml.safe_load(fh) or {}

        # Store session_id plus all metadata fields (nested YAML stays nested)
        rows.append({"session_id": session_id, **meta})
    
    sessions = Dataset.from_list(rows)
    return sessions

def parse(sessions):
    # Extract relevant fields from each session and its policies
    rows = []
    for example in sessions:
        for policy_letter, policy_data in example["policies"].items():
            if policy_data is not None:
                rows.append({
                    "session_id": example["session_id"],
                    "language_instruction": example["language_instruction"],
                    "policy_letter": policy_letter,
                    "policy_name": policy_data["policy_name"],
                    "binary_success": policy_data["binary_success"],
                    "partial_success": policy_data["partial_success"],
                    "duration": policy_data["duration"],
                    "preference": example["preference"],
                })
            else:
                # Handle missing policy data
                rows.append({
                    "session_id": example["session_id"],
                    "language_instruction": example["language_instruction"],
                    "policy_letter": policy_letter,
                    "policy_name": None,
                    "binary_success": None,
                    "partial_success": None,
                    "duration": None,
                    "preference": example["preference"],
                })
    return Dataset.from_list(rows)

def aggregate(parsed_sessions, verbose=False):
    # Aggregate statistics per policy_name
    stats = defaultdict(lambda: {
        "binary_success": 0,
        "partial_success": 0,
        "count": 0,
    })

    for row in parsed_sessions:
        name = row["policy_name"]
        if name is not None:
            stats[name]["binary_success"] += row["binary_success"]
            stats[name]["partial_success"] += row["partial_success"]
            stats[name]["count"] += 1

    aggregated_rows = []
    for name, stat in stats.items():
        aggregated_rows.append({
            "policy_name": name,
            "binary_success": stat["binary_success"] / stat["count"],
            "partial_success": stat["partial_success"] / stat["count"],
            "count": stat["count"],
        })

    if verbose:
        for row in aggregated_rows:
            print(f"Policy: {row['policy_name']}, Binary Success: {row['binary_success']:.3f}, Partial Success: {row['partial_success']:.3f}, Count: {row['count']}")

    df_stats = pd.DataFrame(rows)
    df_stats.to_csv(agg_out_path, index=False)
    return df_stats

def save_per_trial_data(policies, parsed_sessions):
    per_trial_binary_data = {policy: [] for policy in policies}
    per_trial_progress_data = {policy: [] for policy in policies}
    sorted_ds = parsed_sessions.sort("session_id")
    current_session = None

    for row in sorted_ds:
        if row["session_id"] != current_session:
            if current_session is not None:
                # Fill in None for untested policies at the end of the previous session
                untested_policies = set(policies) - tested_policies
                for policy in untested_policies:
                    per_trial_binary_data[policy].append(None)
                    per_trial_progress_data[policy].append(None)

            current_session = row["session_id"]
            tested_policies = set()

        policy_name = row["policy_name"]
        if policy_name in policies:
            tested_policies.add(policy_name)
            per_trial_binary_data[policy_name].append(row["binary_success"])
            per_trial_progress_data[policy_name].append(row["partial_success"])

    for policy in policies:
        print(f"Policy: {policy}, Binary Data Count: {len(per_trial_binary_data[policy])}, Progress Data Count: {len(per_trial_progress_data[policy])}")

    per_trial_binary_out_path = os.path.join(script_dir, "per_trial_binary_data.csv")
    df_binary = pd.DataFrame(per_trial_binary_data)
    df_binary.to_csv(per_trial_binary_out_path, index=False)

    per_trial_progress_out_path = os.path.join(script_dir, "per_trial_progress_data.csv")
    df_progress = pd.DataFrame(per_trial_progress_data)
    df_progress.to_csv(per_trial_progress_out_path, index=False)

def get_policy_names(parsed_sessions):
    df = parsed_sessions.to_pandas()
    policies = df["policy_name"].dropna().unique()
    policies = [p for p in policies if p != "pi05_droid"]
    return policies

def save_policy_preferences(parsed_sessions):
    df = parsed_sessions.to_pandas()
    # Preference values are updated as: 0: A is preferred, 1: B is preferred, 0.5: TIE
    preferences = {"A": [], "B": [], "preference": []}
    sorted_ds = parsed_sessions.sort("session_id")
    current_session = None
    
    for row in sorted_ds:
        session_id = row["session_id"]
        preference = row["preference"]
        policy_letter = row["policy_letter"]
        policy_name = row["policy_name"]

        if preference == "A":
            preference_value = 0
        elif preference == "B":
            preference_value = 1
        elif preference == "TIE":
            preference_value = 0.5
        else:
            raise ValueError(f"Unknown preference value: {preference}")

        if session_id != current_session:
            current_session = session_id
            preference_value_recorded = False
            
        if policy_name is not None:
            if policy_letter == "A" or policy_letter == "B":
                preferences[policy_letter].append(policy_name)
                if not preference_value_recorded:
                    preferences["preference"].append(preference_value)
                    preference_value_recorded = True
                
    assert len(preferences["A"]) == len(preferences["B"]), "Mismatched preference counts"
    preference_stats = pd.DataFrame(preferences)
    pref_out_path = os.path.join(script_dir, "policy_preferences.csv")
    preference_stats.to_csv(pref_out_path, index=False)
    return preference_stats

def main():
    print("Loading data...")
    sessions = load_data()
    print(f"Loaded {len(sessions)} sessions.")
    parsed_sessions = parse(sessions)
    policies = get_policy_names(parsed_sessions)
    save_per_trial_data(policies, parsed_sessions)
    save_policy_preferences(parsed_sessions)

if __name__ == "__main__":
    main()