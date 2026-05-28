"""FMA preprocessing utilities."""

from pathlib import Path

from preprocess.decode_audio import decode_mp3
from preprocess.mel import MelConfig, convert_directory

import numpy as np
import sys


def run_decode_audio(audio_dir: str | Path, sr: int = 22050, skip_existing: bool = True) -> None:
    audio_dir = Path(audio_dir).expanduser().resolve()
    if not audio_dir.is_dir():
        raise NotADirectoryError(f"directory not found: {audio_dir}")
    mp3_paths = sorted(audio_dir.rglob("*.mp3"))
    total = len(mp3_paths)
    processed = skipped = errors = 0
    print(f"[preprocess] decode_audio: {total} tracks in {audio_dir}", flush=True)
    for mp3_path in mp3_paths:
        npy_path = mp3_path.with_suffix(".npy")
        if skip_existing and npy_path.exists():
            skipped += 1
            continue
        try:
            y = decode_mp3(mp3_path, sr)
            np.save(npy_path, y)
            processed += 1
        except Exception as exc:
            print(f"[preprocess] decode error {mp3_path}: {exc}", file=sys.stderr, flush=True)
            errors += 1
    print(f"[preprocess] decode_audio done: processed={processed} skipped={skipped} errors={errors}", flush=True)


def run_mel(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir).expanduser().resolve()
    output_dir = data_dir.parent / f"{data_dir.name}_mel"
    if output_dir.exists() and any(output_dir.rglob("*.pt")):
        print(f"[preprocess] mel: output already exists at {output_dir}, skipping", flush=True)
        return output_dir
    print(f"[preprocess] mel: converting {data_dir} -> {output_dir}", flush=True)
    out, processed, skipped, _ = convert_directory(data_dir, MelConfig())
    print(f"[preprocess] mel done: processed={processed} skipped={skipped}", flush=True)
    return out


__all__ = ["run_decode_audio", "run_mel"]
