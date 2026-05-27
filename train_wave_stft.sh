#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="preprocess/data/fma_small_mel"
AUDIO_ROOT="preprocess/data"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

run_mode() {
    local mode="$1"
    local log="$LOG_DIR/wave_stft_${mode}.log"
    echo "[$(date '+%H:%M:%S')] Starting mode=$mode  log=$log"
    python -m representation.wave_barlow \
        -d "$DATA_DIR" \
        --audio-root "$AUDIO_ROOT" \
        --mode "$mode" \
        2>&1 | tee "$log"
    echo "[$(date '+%H:%M:%S')] Done mode=$mode"
}

run_mode cs
run_mode traditional
run_mode hybrid

echo "[$(date '+%H:%M:%S')] All modes complete."
