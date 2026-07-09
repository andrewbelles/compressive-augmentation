#!/bin/bash -l
#SBATCH --job-name=cs-ssl-analyze
#SBATCH --account=free
#SBATCH --partition=h200_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH --time=0-04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --output=logs/cs-ssl-analyze_%j.out
#SBATCH --error=logs/cs-ssl-analyze_%j.err
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

mkdir -p "$REPO/logs" "$REPO/analysis"

echo "START analyze node=$(hostname) $(date -Iseconds)"
nvidia-smi -L

python analyze.py \
    --parquet        "$REPO/data/wave_barlow_fma_small.parquet" \
    --output-dir     "$REPO/analysis" \
    --checkpoint-dir "$REPO/checkpoints" \
    --audio-root     "$REPO/preprocess/data/fma_small_mel" \
    --dataset-name   fma_small \
    --ref-method     supcon_w3_d256_nopop \
    --exclude-genres Pop \
    --n-aug-epochs   4 \
    --device         cuda:0 \
    2>&1 | tee "$REPO/logs/analyze_${SLURM_JOB_ID}.log"

echo "DONE analyze $(date -Iseconds)"
