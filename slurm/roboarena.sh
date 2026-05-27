#!/bin/bash

#SBATCH --job-name=multitest
#SBATCH --output=logs/%A_evalpi0.out
#SBATCH --error=logs/%A_evalpi0.err
#SBATCH --time=3:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=25G
#SBATCH --partition=all
#SBATCH --signal=USR1@60
#SBATCH --array=0-3

source ~/.bashrc

conda activate policy_comp

cd /n/fs/irom-testing/multitest

EXPERIMENTS=(
    "roboarena4"
    "roboarena7"
    "roboarena4_wm_prior"
    "roboarena7_wm_prior"
)

EXP_NAME=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}
SUBFOLDER="roboarena_forward_graph"
GRAPH_TYPE="soft_masked" # soft_masked, fully_connected

echo "Running experiment: ${EXP_NAME}"

python scripts/roboarena_graphical_test.py \
    --subfolder "${SUBFOLDER}" \
    --exp_name "${EXP_NAME}" \
    --graph_type "${GRAPH_TYPE}"