#!/usr/bin/env bash
set -euo pipefail

# ingest_rml2018.sh  --  acquire and index RadioML 2018.01A
#
# Works identically on a local workstation and on Dartmouth Discovery HPC.
#
# Usage:
#   bash scripts/ingest_rml2018.sh [DATA_DIR] [-j THREADS] [--compress] [--prefetch-check]
#
# Arguments:
#   DATA_DIR          directory for all data (default: data/ relative to repo root)
#   -j N              threads for decompression / zstd (default: nproc)
#   --compress        after download, create a portable zstd archive for HPC transfer
#   --prefetch-check  after manifests are written, open HDF5 and verify integrity
#
# Prerequisites:
#   - Kaggle CLI:  pip install kaggle
#   - Credentials: ~/.kaggle/kaggle.json  (from https://www.kaggle.com/settings → API)
#   - zstd (optional, needed only with --compress):
#       NixOS / nix-shell: nix-shell -p zstd
#       HPC module:        module load zstd
#
# HPC transport workflow (after running this locally with --compress):
#   rsync -avz --progress \
#       data/rml2018/rml2018.hdf5.zst \
#       $USER@discovery.dartmouth.edu:/dartfs-hpc/scratch/$USER/compressive-augmentation/data/rml2018/
#
#   Then on Discovery, decompress:
#       zstd -d -T8 /dartfs-hpc/scratch/$USER/.../rml2018.hdf5.zst
#
#   Then run manifests:
#       sbatch scripts/acquire_rml2018.sbatch
#
# Expected directory layout after completion:
#   data/
#     rml2018/
#       GOLD_XYZ_OSC.0001_1024.hdf5   (~21 GB, canonical source, never modified)
#       rml2018.hdf5.zst               (~13 GB, created only with --compress)
#       manifest_all.csv
#       manifest_training.csv
#       manifest_validation.csv
#       manifest_test.csv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="$REPO_ROOT/data"
THREADS="$(nproc 2>/dev/null || echo 4)"
DO_COMPRESS=0
DO_PREFETCH_CHECK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -j)           THREADS="$2"; shift 2 ;;
        -j*)          THREADS="${1#-j}"; shift ;;
        --compress)   DO_COMPRESS=1; shift ;;
        --prefetch-check) DO_PREFETCH_CHECK=1; shift ;;
        --*)          echo "[ingest_rml2018] unknown flag: $1" >&2; exit 1 ;;
        *)            DATA_DIR="$1"; shift ;;
    esac
done

RML_DIR="$DATA_DIR/rml2018"
HDF5="$RML_DIR/GOLD_XYZ_OSC.0001_1024.hdf5"
ARCHIVE="$RML_DIR/rml2018.hdf5.zst"
MANIFEST="$RML_DIR/manifest_all.csv"

mkdir -p "$RML_DIR"

# ---------------------------------------------------------------------------
# step 1: check Kaggle credentials
# ---------------------------------------------------------------------------
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "[ingest_rml2018] ERROR: Kaggle credentials not found at ~/.kaggle/kaggle.json" >&2
    echo "[ingest_rml2018] To set up:" >&2
    echo "  1. Go to https://www.kaggle.com/settings and click 'Create New Token'" >&2
    echo "  2. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json" >&2
    echo "  3. chmod 600 ~/.kaggle/kaggle.json" >&2
    exit 1
fi

if ! command -v kaggle &>/dev/null; then
    echo "[ingest_rml2018] ERROR: kaggle CLI not found. Install with: pip install kaggle" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# step 2: download HDF5 via Kaggle CLI
# ---------------------------------------------------------------------------
if [ -f "$HDF5" ]; then
    echo "[ingest_rml2018] HDF5 already present at $HDF5, skipping download"
else
    echo "[ingest_rml2018] downloading RadioML 2018.01A (~21 GB) ..."
    kaggle datasets download \
        --dataset pinxau1000/radioml2018 \
        --path "$RML_DIR" \
        --unzip
    # kaggle unzips to a subdirectory or in-place; locate the HDF5
    FOUND=$(find "$RML_DIR" -name "GOLD_XYZ_OSC.0001_1024.hdf5" | head -1)
    if [ -z "$FOUND" ]; then
        echo "[ingest_rml2018] ERROR: HDF5 not found after download" >&2
        exit 1
    fi
    if [ "$FOUND" != "$HDF5" ]; then
        mv "$FOUND" "$HDF5"
    fi
    echo "[ingest_rml2018] download complete: $HDF5"
fi

# ---------------------------------------------------------------------------
# step 3: compress to zstd archive (optional, for HPC transport)
# ---------------------------------------------------------------------------
if [ "$DO_COMPRESS" -eq 1 ]; then
    if [ -f "$ARCHIVE" ]; then
        echo "[ingest_rml2018] zstd archive already present at $ARCHIVE, skipping"
    else
        if ! command -v zstd &>/dev/null; then
            echo "[ingest_rml2018] ERROR: zstd not found. Install with:" >&2
            echo "  NixOS/nix-shell: nix-shell -p zstd" >&2
            echo "  HPC:             module load zstd" >&2
            exit 1
        fi
        echo "[ingest_rml2018] compressing to zstd archive (threads=$THREADS) ..."
        zstd -T"$THREADS" -19 "$HDF5" -o "$ARCHIVE"
        echo "[ingest_rml2018] archive written: $ARCHIVE ($(du -sh "$ARCHIVE" | cut -f1))"
    fi
fi

# ---------------------------------------------------------------------------
# step 4: write split manifests
# ---------------------------------------------------------------------------
if [ -f "$MANIFEST" ]; then
    echo "[ingest_rml2018] manifests already present, skipping"
else
    echo "[ingest_rml2018] writing split manifests ..."
    PYTHONPATH="$REPO_ROOT/src" python -m rf.preprocess.manifests \
        --hdf5 "$HDF5" \
        --output-dir "$RML_DIR"
    echo "[ingest_rml2018] manifests written"
fi

# ---------------------------------------------------------------------------
# step 5: integrity / prefetch check (optional)
# ---------------------------------------------------------------------------
if [ "$DO_PREFETCH_CHECK" -eq 1 ]; then
    echo "[ingest_rml2018] running prefetch integrity check ..."
    INGEST_HDF5="$HDF5" PYTHONPATH="$REPO_ROOT/src" python - <<'PYEOF'
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
print(f"[prefetch-check] OK  X={x_shape} Y={y_shape} Z={z_shape}")
PYEOF
fi

echo "[ingest_rml2018] DONE  data_dir=$RML_DIR"
