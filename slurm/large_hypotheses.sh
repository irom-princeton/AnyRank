#!/bin/bash

#SBATCH --job-name=multitest
#SBATCH --output=logs/%A_evalpi0.out
#SBATCH --error=logs/%A_evalpi0.err
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=all
#SBATCH --signal=USR1@60
#SBATCH --array=0-1

source ~/.bashrc

conda activate policy_comp

cd /n/fs/irom-testing/multitest

EXPERIMENTS=(
    "synthetic_reshuffle"
)

METHODS=(
    "graphical_active"
    "bonferroni"
)
GRAPH_TYPE="fully_connected"


NUM_METHODS=${#METHODS[@]}
EXP_IDX=$((SLURM_ARRAY_TASK_ID / NUM_METHODS))
METHOD_IDX=$((SLURM_ARRAY_TASK_ID % NUM_METHODS))

ALPHA=${1:-0.1}
EXP_NAME=${EXPERIMENTS[$EXP_IDX]}
METHOD=${METHODS[$METHOD_IDX]}

SUBFOLDER="roboarena_${METHOD}_${GRAPH_TYPE}_transitive"
echo "Running experiment: ${EXP_NAME}"

python scripts/tests/large_hypotheses.py \
    --subfolder "${SUBFOLDER}" \
    --alpha ${ALPHA} \
    --n_policies 20 \
    --exp_name "${EXP_NAME}" \
    --graph_type "${GRAPH_TYPE}" \
    --methods "${METHOD}"