#!/usr/bin/env python3
#
# decode_audio.py  Andrew Belles  May 2026
#
# Pre-decode mp3 files to float32 .npy waveforms for fast training data loading.
# Writes {track_id}.npy alongside each mp3 in the audio directory.
#

import argparse
import subprocess
from pathlib import Path

import numpy as np

DEFAULT_AUDIO_DIR   = Path("preprocess/data/fma_small")
DEFAULT_SAMPLE_RATE = 22050


def decode_mp3(mp3_path: Path, sr: int) -> np.ndarray:
    """
    Decode and peak-normalize one mp3 to a float32 mono waveform.

    Assumptions:
    - ffmpeg is installed and produces f32le samples on stdout.
    """
    cmd = ["ffmpeg", "-y", "-i", str(mp3_path), "-ar", str(sr), "-ac", "1", "-f", "f32le", "-"]
    result = subprocess.run(cmd, capture_output=True)
    y = np.frombuffer(result.stdout, dtype=np.float32)
    if len(y) == 0:
        raise ValueError(f"ffmpeg produced no output for {mp3_path}")
    peak = np.abs(y).max()
    if peak > 1e-8:
        y = y / peak
    return y


def main() -> int:
    """
    CLI entry point for writing cached .npy waveforms beside mp3 files.

    Assumptions:
    - Existing .npy files are valid caches and can be skipped.
    """
    parser = argparse.ArgumentParser(description="Pre-decode mp3 files to float32 .npy waveforms.")
    parser.add_argument("-d", "--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--sr", type=int, default=DEFAULT_SAMPLE_RATE)
    args = parser.parse_args()

    audio_dir = args.audio_dir.expanduser().resolve()
    if not audio_dir.is_dir():
        raise NotADirectoryError(f"directory not found: {audio_dir}")

    mp3_paths = sorted(audio_dir.rglob("*.mp3"))
    total = len(mp3_paths)
    processed = skipped = errors = 0
    print(f"START audio_dir={audio_dir} total={total} sr={args.sr}", flush=True)

    for i, mp3_path in enumerate(mp3_paths, 1):
        npy_path = mp3_path.with_suffix(".npy")
        if npy_path.exists():
            skipped += 1
            continue
        try:
            np.save(npy_path, decode_mp3(mp3_path, args.sr))
            processed += 1
        except Exception as exc:
            print(f"[decode] error {mp3_path}: {exc}", flush=True)
            errors += 1
        if i % 500 == 0 or i == total:
            print(f"[decode] {i}/{total}  processed={processed} skipped={skipped} errors={errors}", flush=True)

    print(f"DONE processed={processed} skipped={skipped} errors={errors}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
