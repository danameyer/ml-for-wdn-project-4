#!/bin/bash

#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-29%8
#SBATCH --output=logs/gnn-ekf-%A_%a.out
#SBATCH --error=logs/gnn-ekf-%A_%a.err

set -euo pipefail


# Set arguments

FILTER_TYPE="${1:?Expected 'ekf' or 'enkf'}"
DATASET="${2:?Expected 'net1' or 'hanoi'}"

if [[ "$FILTER_TYPE" != "ekf" && "$FILTER_TYPE" != "enkf" ]]; then
    echo "FILTER_TYPE must be 'ekf' or 'enkf'."
    exit 1
fi

if [[ "$DATASET" != "net1" && "$DATASET" != "hanoi" ]]; then
    echo "DATASET must be 'net1' or 'hanoi'."
    exit 1
fi

# Submit data to \logs folder

PROJECT_ROOT=/homes/"$USER"/ml-for-wdn-project-4

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    mkdir -p "$PROJECT_ROOT/logs"

    echo "Submitting GNN $FILTER_TYPE array..."

    sbatch \
        --job-name="gnn-$FILTER_TYPE" \
        --output="$PROJECT_ROOT/logs/gnn-${FILTER_TYPE}-%A_%a.out" \
        --error="$PROJECT_ROOT/logs/gnn-${FILTER_TYPE}-%A_%a.err" \
        "$0" \
        "$FILTER_TYPE" \
        "$DATASET"

    exit 0
fi


# Project setup

cd /homes/"$USER"/ml-for-wdn-project-4


# define Python environment

source ~/miniconda3/etc/profile.d/conda.sh
conda activate wdn-gnn-kalman


# CPU threading

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"


# set one placement per array task

SEED=$((42 + SLURM_ARRAY_TASK_ID))


echo "Filter:    $FILTER_TYPE"
echo "Job:       $SLURM_JOB_ID"
echo "Array ID:  $SLURM_ARRAY_TASK_ID"
echo "Seed:      $SEED"
echo "Host:      $(hostname)"
echo "Dataset:   $DATASET"
echo

# Run filter experiment

if [[ "$FILTER_TYPE" == "ekf" ]]; then

    python scripts/run_ekf.py \
        --dataset "$DATASET" \
        --model gnn \
        --buffer-size 20 \
        --unroll-steps 20 \
        --n-iters 1 \
        --seed "$SEED"

else

    python scripts/run_enkf.py \
        --dataset "$DATASET" \
        --model gnn \
        --buffer-size 20 \
        --unroll-steps 20 \
        --n-iters 1 \
        --ensemble-size 50 \
        --seed "$SEED"

fi