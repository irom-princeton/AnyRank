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
#SBATCH --array=0-3

source ~/.bashrc

conda activate policy_comp

cd /n/fs/irom-testing/multitest

EXPERIMENTS=(
    "roboarena7_bradley_terry_half"
)

METHODS=(
    "graphical_active"
    "bonferroni"
    "weighted_bonferroni"
    "fixed"
)
GRAPH_TYPE="fully_connected"

ALPHA=${1:-0.1}
NUM_METHODS=${#METHODS[@]}
EXP_IDX=$((SLURM_ARRAY_TASK_ID / NUM_METHODS))
METHOD_IDX=$((SLURM_ARRAY_TASK_ID % NUM_METHODS))

EXP_NAME=${EXPERIMENTS[$EXP_IDX]}
METHOD=${METHODS[$METHOD_IDX]}

SUBFOLDER="roboarena_${METHOD}_${GRAPH_TYPE}_transitive_alpha_${ALPHA}"
echo "Running experiment: ${EXP_NAME}"

python scripts/roboarena_individual_test.py \
    --subfolder "${SUBFOLDER}" \
    --exp_name "${EXP_NAME}" \
    --alpha ${ALPHA} \
    --graph_type "${GRAPH_TYPE}" \
    --methods "${METHOD}"