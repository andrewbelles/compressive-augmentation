#!/bin/bash -l
#SBATCH --job-name=rml2018-acquire
#SBATCH --account=free
#SBATCH --partition=standard
#SBATCH --time=4:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/rml2018-acquire_%j.out
#SBATCH --error=logs/rml2018-acquire_%j.err
#SBATCH --mail-type=END,FAIL

# acquire_rml2018.sbatch
#
# SLURM job for acquiring and indexing RadioML 2018.01A on Dartmouth Discovery.
#
# Preferred workflow (direct download on-cluster, requires compute-node
# internet egress -- confirmed available via `curl -sI https://www.kaggle.com`
# from an srun session):
#
#   1. Put Kaggle credentials on Discovery (not in the repo):
#        scp kaggle.json $USER@discovery.dartmouth.edu:~/.kaggle/kaggle.json
#        ssh $USER@discovery.dartmouth.edu chmod 600 ~/.kaggle/kaggle.json
#
#   2. Make sure `kaggle` is installed in the CONDA_ENV below
#        (conda activate compressive-augmentation && pip install kaggle)
#
#   3. Submit this job -- it downloads, decompresses (if needed), and writes
#      manifests in one step:
#        sbatch scripts/acquire_rml2018.sbatch
#
# Fallback workflow (no compute-node egress): download+compress locally,
# rsync the archive to scratch, then submit this job to decompress instead:
#
#   1. bash scripts/ingest_rml2018.sh data/ --compress
#   2. rsync -avz --progress \
#          data/rml2018/rml2018.hdf5.zst \
#          $USER@discovery.dartmouth.edu:/dartfs-hpc/scratch/$USER/compressive-augmentation/data/rml2018/
#   3. sbatch scripts/acquire_rml2018.sbatch

set -euo pipefail
export PYTHONUNBUFFERED=1

SCRATCH=/dartfs-hpc/scratch/$USER
REPO=$SCRATCH/compressive-augmentation
DATA_DIR=$REPO/data
RML_DIR=$DATA_DIR/rml2018
HDF5=$RML_DIR/GOLD_XYZ_OSC.0001_1024.hdf5
ARCHIVE=$RML_DIR/rml2018.hdf5.zst
CONDA_ENV=compressive-augmentation

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

mkdir -p "$RML_DIR" "$REPO/logs"

echo "START rml2018-acquire node=$(hostname) $(date -Iseconds)"

# ---------------------------------------------------------------------------
# Acquire HDF5: decompress a pre-staged archive, or download directly
# ---------------------------------------------------------------------------
if [ -f "$HDF5" ]; then
    echo "HDF5 already present, skipping acquisition"
elif [ -f "$ARCHIVE" ]; then
    echo "Decompressing $ARCHIVE (threads=8) ..."
    module load zstd 2>/dev/null || true
    zstd -d -T8 "$ARCHIVE" -o "$HDF5"
    echo "Decompression complete: $(du -sh "$HDF5" | cut -f1)"
else
    echo "No local HDF5 or archive found -- downloading directly from Kaggle ..."
    bash "$REPO/scripts/ingest_rml2018.sh" "$DATA_DIR" -j 8
fi

# ---------------------------------------------------------------------------
# Write split manifests
# ---------------------------------------------------------------------------
MANIFEST="$RML_DIR/manifest_all.csv"
if [ -f "$MANIFEST" ]; then
    echo "Manifests already present, skipping"
else
    echo "Writing split manifests ..."
    PYTHONPATH="$REPO/src" python -m rf.preprocess.manifests \
        --hdf5 "$HDF5" \
        --output-dir "$RML_DIR"
    echo "Manifests written"
fi

# ---------------------------------------------------------------------------
# Integrity check
# ---------------------------------------------------------------------------
echo "Running integrity check ..."
INGEST_HDF5="$HDF5" PYTHONPATH="$REPO/src" python - <<'PYEOF'
import os, h5py, numpy as np
hdf5_path = os.environ["INGEST_HDF5"]
with h5py.File(hdf5_path, "r") as f:
    x_shape    = f["X"].shape
    y_shape    = f["Y"].shape
    z_shape    = f["Z"].shape
    frame_0    = f["X"][0]
    frame_last = f["X"][-1]
assert x_shape == (2555904, 1024, 2), f"unexpected X shape: {x_shape}"
assert y_shape == (2555904, 24),      f"unexpected Y shape: {y_shape}"
assert z_shape == (2555904,),         f"unexpected Z shape: {z_shape}"
assert np.isfinite(frame_0).all(),    "NaN/Inf in frame 0"
assert np.isfinite(frame_last).all(), "NaN/Inf in final frame"
print(f"[integrity] OK  X={x_shape} Y={y_shape} Z={z_shape}")
PYEOF

echo "DONE rml2018-acquire $(date -Iseconds)"
echo "Data ready at: $RML_DIR"
