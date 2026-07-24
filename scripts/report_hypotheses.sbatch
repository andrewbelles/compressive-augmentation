#!/bin/bash -l
#SBATCH --job-name=coaug-report
#SBATCH --account=free
#SBATCH --partition=standard
#SBATCH --time=0:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/coaug-report_%j.out
#SBATCH --error=logs/coaug-report_%j.err
#SBATCH --mail-type=END,FAIL

# report_hypotheses.sbatch
#
# Aggregate seed artifacts into the verdict table and figures. Submit with a dependency
# on the array job:
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> scripts/report_hypotheses.sbatch

set -euo pipefail
export PYTHONUNBUFFERED=1

SCRATCH=/dartfs-hpc/scratch/$USER
REPO=$SCRATCH/compressive-augmentation
ARTIFACTS=$REPO/analysis/rf_hypotheses
CONDA_ENV=compressive-augmentation

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$REPO"

echo "START report node=$(hostname) $(date -Iseconds)"
PYTHONPATH=src python report_hypotheses.py --out "$ARTIFACTS"
echo "DONE report $(date -Iseconds)"
echo "Verdicts: $ARTIFACTS/verdicts.csv"
