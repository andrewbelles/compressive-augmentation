#!/bin/bash -l
#SBATCH --job-name=cs-ssl-sweep
#SBATCH --account=free
#SBATCH --partition=h200_preemptable
#SBATCH --gres=gpu:h200:2
#SBATCH --time=1-12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --output=logs/cs-ssl-sweep_%j.out
#SBATCH --error=logs/cs-ssl-sweep_%j.err
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

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$REPO"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" \
    || { echo "FATAL: CUDA not available on $(hostname)" >&2; exit 1; }

python -c "import torch; n=torch.cuda.device_count(); assert n>=2, f'need 2 GPUs, got {n}'" \
    || { echo "FATAL: need 2 H200s, got fewer" >&2; exit 1; }

python -c "from representation import run_wave_barlow; from representation.supcon import train_supcon" \
    || { echo "FATAL: spiky package not importable — run pip install -e . first" >&2; exit 1; }

mkdir -p "$REPO/checkpoints" "$REPO/data" "$REPO/logs"

export TRITON_CACHE_DIR=$REPO/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=$REPO/.inductor_cache
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "START sweep node=$(hostname) gpus=$(python -c 'import torch; print(torch.cuda.device_count())') $(date -Iseconds)"
nvidia-smi -L

CUDA_VISIBLE_DEVICES=0 python train_sweep.py \
    --half 0 \
    --scratch-dir "$REPO" \
    --data-dir "$REPO/preprocess/data/fma_small_mel" \
    --audio-root "$REPO/preprocess/data" \
    --num-workers 2 \
    --kinds cs_biased cs_uniform cs_srht \
    > "$REPO/logs/half0_${SLURM_JOB_ID}.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 python train_sweep.py \
    --half 1 \
    --scratch-dir "$REPO" \
    --data-dir "$REPO/preprocess/data/fma_small_mel" \
    --audio-root "$REPO/preprocess/data" \
    --num-workers 2 \
    --kinds cs_biased cs_uniform cs_srht \
    > "$REPO/logs/half1_${SLURM_JOB_ID}.log" 2>&1 &

wait

echo "DONE sweep $(date -Iseconds)"
