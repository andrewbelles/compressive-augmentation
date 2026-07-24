#!/bin/bash -l
#SBATCH --job-name=coaug-hypotheses
#SBATCH --account=free
#SBATCH --partition=h200_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-15
#SBATCH --output=logs/coaug-hypotheses_%A_%a.out
#SBATCH --error=logs/coaug-hypotheses_%A_%a.err
#SBATCH --mail-type=END,FAIL

# run_hypotheses.sbatch
#
# Stage-1 CoAug hypothesis battery on Discovery. The SLURM array index is the seed:
# each task runs every mechanism for its seed and writes seed-tagged parquet artifacts.
#
#   sbatch scripts/run_hypotheses.sbatch
#
# Aggregate into verdicts once the array finishes (single dependent task):
#
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> scripts/report_hypotheses.sbatch
#
# or, interactively after completion:
#
#   PYTHONPATH=src python report_hypotheses.py --out "$ARTIFACTS"

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

SCRATCH=/dartfs-hpc/scratch/$USER
REPO=$SCRATCH/compressive-augmentation
RML_DIR=$REPO/data/rml2018
HDF5=$RML_DIR/GOLD_XYZ_OSC.0001_1024.hdf5
MANIFEST=$RML_DIR/manifest_all.csv
ARTIFACTS=$REPO/analysis/rf_hypotheses
CONDA_ENV=compressive-augmentation
SEED=${SLURM_ARRAY_TASK_ID:-0}

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$REPO"
mkdir -p "$ARTIFACTS" "$REPO/logs"

test -f "$HDF5"     || { echo "FATAL: missing $HDF5 (run acquire_rml2018.sbatch first)" >&2; exit 1; }
test -f "$MANIFEST" || { echo "FATAL: missing $MANIFEST (run acquire_rml2018.sbatch first)" >&2; exit 1; }

echo "START hypotheses seed=$SEED node=$(hostname) $(date -Iseconds)"

for MECH in operator_isometry nuisance_equivariance gabor_compressibility \
            se_calibration structure_vs_unstructured cumulant_margin admissible_band; do
    echo "--- $MECH seed=$SEED ---"
    PYTHONPATH=src python run_hypotheses.py \
        --mechanism "$MECH" \
        --seed "$SEED" \
        --hdf5 "$HDF5" \
        --manifest "$MANIFEST" \
        --out "$ARTIFACTS" \
        --per-group 64
done

echo "DONE hypotheses seed=$SEED $(date -Iseconds)"
