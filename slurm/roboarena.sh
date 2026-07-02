#!/bin/bash

#SBATCH --job-name=multitest
#SBATCH --output=logs/%A_evalpi0.out
#SBATCH --error=logs/%A_evalpi0.err
#SBATCH --time=6:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=25G
#SBATCH --partition=all
#SBATCH --signal=USR1@60
#SBATCH --array=0

source ~/.bashrc

conda activate policy_comp

cd /n/fs/irom-testing/multitest

EXPERIMENTS=(
    "roboarena7_wm_prior_0.1_half_new_run"
)

METHODS=(
    "weighted_bonferroni"
)
GRAPH_TYPE="fully_connected"


NUM_METHODS=${#METHODS[@]}
EXP_IDX=$((SLURM_ARRAY_TASK_ID / NUM_METHODS))
METHOD_IDX=$((SLURM_ARRAY_TASK_ID % NUM_METHODS))

EXP_NAME=${EXPERIMENTS[$EXP_IDX]}
METHOD=${METHODS[$METHOD_IDX]}

SUBFOLDER="roboarena_${METHOD}_${GRAPH_TYPE}_transitive"
echo "Running experiment: ${EXP_NAME}"

python scripts/roboarena_individual_test.py \
    --subfolder "${SUBFOLDER}" \
    --exp_name "${EXP_NAME}" \
    --graph_type "${GRAPH_TYPE}" \
    --methods "${METHOD}"