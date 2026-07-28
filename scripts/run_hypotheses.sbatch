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

SNRS=-10,0,6,10,14,18,22,26,30
COARSE=0.3,0.4,0.5,0.6,0.7,0.8,0.9
FINE=0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95

run () {  # mechanism, ratios, extra flags
    echo "--- $1 seed=$SEED ${3:-} ---"
    PYTHONPATH=src python run_hypotheses.py \
        --mechanism "$1" --seed "$SEED" \
        --hdf5 "$HDF5" --manifest "$MANIFEST" --out "$ARTIFACTS" \
        --snrs "$SNRS" --ratios "$2" --per-group 64 ${3:-}
}

# operator algebra and controls: no measurement-noise axis
run operator_isometry "$COARSE"
run operator_equivariance "$COARSE"
run backprojection_law "$COARSE"
run dictionary_compressibility "$COARSE"
run operator_draw_variance "$COARSE" "--draws 64"

# measurement SNR is the second knob: sigma sets the ceiling, so the knee needs the sweep
for MSNR in 30 20 10; do
    run se_calibration "$COARSE" "--measurement-snr $MSNR"
    run se_kappa_prediction "$COARSE" "--measurement-snr $MSNR"
    run cumulant_margin "$COARSE" "--measurement-snr $MSNR"
    run admissible_band "$COARSE" "--measurement-snr $MSNR"
    run se_knee "$FINE" "--measurement-snr $MSNR"
    run kernel_geometry "$FINE" "--measurement-snr $MSNR"
    run label_nuisance_tradeoff "$FINE" "--measurement-snr $MSNR"
done
run se_knee "$FINE"          # noiseless reference: the argmax should sit at the boundary
run kernel_geometry "$FINE"
run label_nuisance_tradeoff "$FINE"

echo "DONE hypotheses seed=$SEED $(date -Iseconds)"
