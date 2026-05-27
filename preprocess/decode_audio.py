#!/usr/bin/env python3
#
# decode_audio.py  Andrew Belles  May 2026
#
# Pre-decode mp3 files to float32 .npy waveforms for fast training data loading.
# Writes {track_id}.npy alongside each mp3 in the fma_small directory.
#

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

DEFAULT_AUDIO_DIR = Path("preprocess/data/fma_small")
DEFAULT_SAMPLE_RATE = 22050


def report(message: str) -> None:
    print(message, flush=True)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-decode mp3 files to float32 .npy waveforms."
    )
    parser.add_argument(
        "-d", "--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR,
        help=f"Root directory containing mp3 files. Defaults to {DEFAULT_AUDIO_DIR}.",
    )
    parser.add_argument(
        "--sr", type=int, default=DEFAULT_SAMPLE_RATE,
        help=f"Target sample rate. Default {DEFAULT_SAMPLE_RATE}.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip tracks that already have a .npy file (default: True).",
    )
    parser.add_argument(
        "--no-skip-existing", dest="skip_existing", action="store_false",
        help="Recompute even if .npy already exists.",
    )
    return parser.parse_args()


def decode_mp3(mp3_path: Path, sr: int) -> np.ndarray:
    cmd = [
        "ffmpeg", "-y", "-i", str(mp3_path),
        "-ar", str(sr), "-ac", "1",
        "-f", "f32le", "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    y = np.frombuffer(result.stdout, dtype=np.float32)
    if len(y) == 0:
        raise ValueError(f"ffmpeg produced no output for {mp3_path}")
    peak = np.abs(y).max()
    if peak > 1e-8:
        y = y / peak
    return y


def main() -> int:
    args = parse_args()
    audio_dir = args.audio_dir.expanduser().resolve()
    if not audio_dir.is_dir():
        raise NotADirectoryError(f"directory not found: {audio_dir}")

    mp3_paths = sorted(audio_dir.rglob("*.mp3"))
    total = len(mp3_paths)
    processed = skipped = errors = 0

    report(f"START audio_dir={audio_dir} total={total} sr={args.sr}")

    for i, mp3_path in enumerate(mp3_paths, 1):
        npy_path = mp3_path.with_suffix(".npy")
        if args.skip_existing and npy_path.exists():
            skipped += 1
            continue
        try:
            y = decode_mp3(mp3_path, args.sr)
            np.save(npy_path, y)
            processed += 1
        except Exception as exc:
            log(f"[decode] error {mp3_path}: {exc}")
            errors += 1

        if i % 500 == 0 or i == total:
            log(f"[decode] {i}/{total}  processed={processed} skipped={skipped} errors={errors}")

    report(f"DONE processed={processed} skipped={skipped} errors={errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
