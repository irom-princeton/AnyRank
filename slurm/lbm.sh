#!/bin/bash

#SBATCH --job-name=multitest
#SBATCH --output=logs/%A_evalpi0.out
#SBATCH --error=logs/%A_evalpi0.err
#SBATCH --time=1:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=25G
#SBATCH --partition=all
#SBATCH --signal=USR1@60

source ~/.bashrc

conda activate policy_comp

cd /n/fs/irom-testing/multitest

SUBFOLDER="lbm_forward_pref_graph"
GRAPH_TYPE="soft_masked" # soft_masked, fully_connected

python scripts/lbm_graphical_test.py --subfolder "${SUBFOLDER}" --graph_type "${GRAPH_TYPE}"