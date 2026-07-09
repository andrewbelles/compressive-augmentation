#!/usr/bin/env bash
set -euo pipefail

# ingest_fma.sh  --  download, unzip, and preprocess FMA Small
#
# Usage:
#   bash scripts/ingest_fma.sh [DATA_DIR] [--mel] [--sample-images]
#
# DATA_DIR defaults to data/ relative to the repo root.
# --mel           also generate mel-spectrogram .pt tensors (needed for mel-PCA baseline only)
# --sample-images generate one preview image per genre (implies --mel)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="$REPO_ROOT/data"
DO_MEL=0
DO_IMAGES=0

for arg in "$@"; do
    case "$arg" in
        --mel)           DO_MEL=1 ;;
        --sample-images) DO_MEL=1; DO_IMAGES=1 ;;
        --*)             echo "[ingest_fma] unknown flag: $arg" >&2; exit 1 ;;
        *)               DATA_DIR="$arg" ;;
    esac
done

mkdir -p "$DATA_DIR/downloads"

FMA_AUDIO_URL="https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_META_URL="https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
FMA_AUDIO_ZIP="$DATA_DIR/downloads/fma_small.zip"
FMA_META_ZIP="$DATA_DIR/downloads/fma_metadata.zip"
FMA_AUDIO_DIR="$DATA_DIR/fma_small"
FMA_META_DIR="$DATA_DIR/fma_metadata"

# step 1: download audio
if [ -f "$FMA_AUDIO_ZIP" ]; then
    echo "[ingest_fma] fma_small.zip already present, skipping download"
else
    echo "[ingest_fma] downloading FMA Small audio (~8 GB) ..."
    wget -c -O "$FMA_AUDIO_ZIP" "$FMA_AUDIO_URL"
    echo "[ingest_fma] download complete"
fi

# step 2: download metadata
if [ -f "$FMA_META_ZIP" ]; then
    echo "[ingest_fma] fma_metadata.zip already present, skipping download"
else
    echo "[ingest_fma] downloading FMA metadata (~342 MB) ..."
    wget -c -O "$FMA_META_ZIP" "$FMA_META_URL"
    echo "[ingest_fma] metadata download complete"
fi

# step 3: unzip audio
if [ -d "$FMA_AUDIO_DIR" ]; then
    echo "[ingest_fma] $FMA_AUDIO_DIR already exists, skipping unzip"
else
    echo "[ingest_fma] unzipping audio to $FMA_AUDIO_DIR ..."
    unzip -q "$FMA_AUDIO_ZIP" -d "$DATA_DIR"
    echo "[ingest_fma] audio unzip complete"
fi

# step 4: unzip metadata
if [ -d "$FMA_META_DIR" ]; then
    echo "[ingest_fma] $FMA_META_DIR already exists, skipping unzip"
else
    echo "[ingest_fma] unzipping metadata to $FMA_META_DIR ..."
    unzip -q "$FMA_META_ZIP" -d "$DATA_DIR"
    echo "[ingest_fma] metadata unzip complete"
fi

# step 5: decode mp3 -> .npy
FIRST_NPY=$(find "$FMA_AUDIO_DIR" -name "*.npy" -maxdepth 3 | head -1)
if [ -n "$FIRST_NPY" ]; then
    echo "[ingest_fma] .npy files already present, skipping decode"
else
    echo "[ingest_fma] decoding mp3 -> .npy (this may take 10-20 min) ..."
    PYTHONPATH="$REPO_ROOT/src" python -m audio.preprocess.decode_audio \
        -d "$FMA_AUDIO_DIR"
    echo "[ingest_fma] decode complete"
fi

# step 6: write manifests
MANIFEST="$DATA_DIR/fma_small_mel/manifest_all.csv"
if [ -f "$MANIFEST" ]; then
    echo "[ingest_fma] manifests already present, skipping"
else
    echo "[ingest_fma] writing split manifests ..."
    PYTHONPATH="$REPO_ROOT/src" python -m audio.preprocess.manifests \
        -d "$FMA_AUDIO_DIR"
    echo "[ingest_fma] manifests written"
fi

# step 7: mel tensors (optional)
if [ "$DO_MEL" -eq 1 ]; then
    FIRST_PT=$(find "$DATA_DIR/fma_small_mel" -name "*.pt" -maxdepth 3 | head -1)
    if [ -n "$FIRST_PT" ]; then
        echo "[ingest_fma] mel tensors already present, skipping"
    else
        echo "[ingest_fma] generating mel tensors (GPU recommended) ..."
        MEL_ARGS="-d $FMA_AUDIO_DIR"
        if [ "$DO_IMAGES" -eq 1 ]; then
            MEL_ARGS="$MEL_ARGS --sample-images"
        fi
        PYTHONPATH="$REPO_ROOT/src" python -m audio.preprocess.mel $MEL_ARGS
        echo "[ingest_fma] mel tensors complete"
    fi
fi

echo "[ingest_fma] DONE  data_dir=$DATA_DIR"
