"""FMA preprocessing utilities."""

import sys
import zipfile
from pathlib import Path

import numpy as np

from preprocess.decode_audio import decode_mp3
from preprocess.mel import MelConfig, convert_directory


def run_decode_audio(audio_dir: str | Path, sr: int = 22050) -> None:
    audio_dir = Path(audio_dir).expanduser().resolve()
    if not audio_dir.is_dir():
        raise NotADirectoryError(f"directory not found: {audio_dir}")
    mp3_paths = sorted(audio_dir.rglob("*.mp3"))
    total = len(mp3_paths)
    processed = errors = 0
    print(f"[preprocess] decode_audio: {total} tracks in {audio_dir}", flush=True)
    for mp3_path in mp3_paths:
        try:
            y = decode_mp3(mp3_path, sr)
            np.save(mp3_path.with_suffix(".npy"), y)
            processed += 1
        except Exception as exc:
            print(f"[preprocess] decode error {mp3_path}: {exc}", file=sys.stderr, flush=True)
            errors += 1
    print(f"[preprocess] decode_audio done: processed={processed} errors={errors}", flush=True)


def run_audio_unzip(zip_path: str | Path, data_dir: str | Path) -> None:
    zip_path = Path(zip_path).expanduser().resolve()
    data_dir = Path(data_dir).expanduser().resolve()
    audio_dir = data_dir
    mel_dir = data_dir.parent / f"{data_dir.name}_mel"
    has_npy = audio_dir.exists() and any(audio_dir.rglob("*.npy"))
    has_mel = mel_dir.exists() and any(mel_dir.rglob("*.pt"))
    if has_npy and has_mel:
        print(f"[preprocess] audio_unzip: data already present, skipping", flush=True)
        return
    if not zip_path.is_file():
        raise FileNotFoundError(f"zip not found: {zip_path}")
    print(f"[preprocess] audio_unzip: extracting {zip_path} -> {data_dir.parent}", flush=True)
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir.parent)
    print(f"[preprocess] audio_unzip done", flush=True)


def run_mel(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir).expanduser().resolve()
    output_dir = data_dir.parent / f"{data_dir.name}_mel"
    if output_dir.exists() and any(output_dir.rglob("*.pt")):
        print(f"[preprocess] mel: already exists at {output_dir}, skipping", flush=True)
        return output_dir
    print(f"[preprocess] mel: converting {data_dir} -> {output_dir}", flush=True)
    out, processed, skipped, _ = convert_directory(data_dir, MelConfig())
    print(f"[preprocess] mel done: processed={processed} skipped={skipped}", flush=True)
    return out


__all__ = ["run_decode_audio", "run_audio_unzip", "run_mel"]
