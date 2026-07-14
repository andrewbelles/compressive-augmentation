#!/bin/bash -l
#SBATCH --job-name=src-analyze
#SBATCH --account=free
#SBATCH --partition=standard
#SBATCH --time=0-01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --output=logs/src-analyze_%j.out
#SBATCH --error=logs/src-analyze_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SCRATCH=/dartfs-hpc/scratch/$USER
REPO=$SCRATCH/compressive-augmentation
CONDA_ENV=compressive-augmentation

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$REPO"
export PYTHONPATH=$REPO/src

mkdir -p "$REPO/logs" "$REPO/analysis/src_ladder"

echo "START src-analyze node=$(hostname) $(date -Iseconds)"

python analyze_src.py \
    --results-dir "$REPO/results/src_ladder" \
    --output-dir  "$REPO/analysis/src_ladder" \
    2>&1 | tee "$REPO/logs/src_analyze_${SLURM_JOB_ID}.log"

echo "DONE src-analyze $(date -Iseconds)"
