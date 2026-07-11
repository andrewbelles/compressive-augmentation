#!/bin/bash -l
#SBATCH --job-name=rml2018-acquire
#SBATCH --account=free
#SBATCH --partition=standard
#SBATCH --time=2:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/rml2018-acquire_%j.out
#SBATCH --error=logs/rml2018-acquire_%j.err
#SBATCH --mail-type=END,FAIL

# acquire_rml2018.sbatch
#
# SLURM job for acquiring and indexing RadioML 2018.01A on Dartmouth Discovery.
#
# This job assumes the HDF5 file has already been transferred to scratch.
# Typical workflow:
#
#   1. Run locally to download and compress:
#        bash scripts/ingest_rml2018.sh data/ --compress
#
#   2. Transfer the archive to Discovery scratch:
#        rsync -avz --progress \
#            data/rml2018/rml2018.hdf5.zst \
#            $USER@discovery.dartmouth.edu:/dartfs-hpc/scratch/$USER/compressive-augmentation/data/rml2018/
#
#   3. Submit this job to decompress and write manifests:
#        sbatch scripts/acquire_rml2018.sbatch
#
# Alternatively, if Kaggle credentials are configured on Discovery, this job
# can run the full ingest_rml2018.sh directly (step 1 above on the cluster).

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
# Decompress zstd archive if HDF5 not yet present
# ---------------------------------------------------------------------------
if [ -f "$HDF5" ]; then
    echo "HDF5 already present, skipping decompression"
elif [ -f "$ARCHIVE" ]; then
    echo "Decompressing $ARCHIVE (threads=8) ..."
    module load zstd 2>/dev/null || true
    zstd -d -T8 "$ARCHIVE" -o "$HDF5"
    echo "Decompression complete: $(du -sh "$HDF5" | cut -f1)"
else
    echo "Neither HDF5 nor zstd archive found at $RML_DIR" >&2
    echo "Options:" >&2
    echo "  A) rsync rml2018.hdf5.zst from local machine (see script header)" >&2
    echo "  B) Run ingest_rml2018.sh directly if Kaggle credentials are available on Discovery" >&2
    exit 1
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
