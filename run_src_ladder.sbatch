#!/bin/bash -l
#SBATCH --job-name=src-ladder
#SBATCH --account=free
#SBATCH --partition=h200_preemptable
#SBATCH --gres=gpu:h200:2
#SBATCH --time=0-08:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --output=logs/src-ladder_%j.out
#SBATCH --error=logs/src-ladder_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRATCH=/dartfs-hpc/scratch/$USER
REPO=$SCRATCH/compressive-augmentation
CONDA_ENV=compressive-augmentation

HDF5=$REPO/data/rml2018/GOLD_XYZ_OSC.0001_1024.hdf5
MANIFESTS=$REPO/data/rml2018

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$REPO"
export PYTHONPATH=$REPO/src

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" \
    || { echo "FATAL: CUDA not available on $(hostname)" >&2; exit 1; }

python -c "import torch; n=torch.cuda.device_count(); assert n>=2, f'need 2 GPUs, got {n}'" \
    || { echo "FATAL: need 2 H200s, got fewer" >&2; exit 1; }

mkdir -p "$REPO/logs" "$REPO/results/src_ladder" "$REPO/dicts"

echo "START src-ladder node=$(hostname) $(date -Iseconds)"
nvidia-smi -L

CUDA_VISIBLE_DEVICES=0 python run_src_ladder.py \
    --half 0 \
    --scratch-dir "$REPO" \
    --hdf5 "$HDF5" \
    --manifest-dir "$MANIFESTS" \
    > "$REPO/logs/src_half0_${SLURM_JOB_ID}.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 python run_src_ladder.py \
    --half 1 \
    --scratch-dir "$REPO" \
    --hdf5 "$HDF5" \
    --manifest-dir "$MANIFESTS" \
    > "$REPO/logs/src_half1_${SLURM_JOB_ID}.log" 2>&1 &

wait

echo "DONE src-ladder $(date -Iseconds)"
