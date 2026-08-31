#!/bin/bash
#SBATCH --job-name=gnn-sweep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-35%16
#SBATCH --output=logs/gnn-sweep-%A_%a.out
#SBATCH --error=logs/gnn-sweep-%A_%a.err

set -euo pipefail

PROJECT_ROOT=/homes/"$USER"/ml-for-wdn-project-4

cd "$PROJECT_ROOT"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate wdn-gnn-kalman

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

BUFFER_SIZES=(1 5 10 15 20 25)
ROLLOUTS=(1 5 10 15 20 25)
NUM_ROLLOUTS=${#ROLLOUTS[@]}
BUFFER_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_ROLLOUTS))
ROLLOUT_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_ROLLOUTS))
BUFFER_SIZE="${BUFFER_SIZES[$BUFFER_INDEX]}"
ROLLOUT="${ROLLOUTS[$ROLLOUT_INDEX]}"

echo "================================"
echo "GNN training sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Buffer size: $BUFFER_SIZE"
echo "Rollout length: $ROLLOUT"
echo "================================"

python -u scripts/run_gnn_training_sweep.py \
    --dataset net1 \
    --buffer-size "$BUFFER_SIZE" \
    --unroll-steps "$ROLLOUT" \
    --epochs 600 \
    --lr 0.001 \
    --seed 42 \
    --link-loss-weight 1.0 \
    --buffer-spread-weight 0.01