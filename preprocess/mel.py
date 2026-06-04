#!/usr/bin/env python3
#
# mel.py  Andrew Belles  April 10th 2026
#
# Generates Mel-Spectrogram matching WaveSTFTEncoder exactly:
# n_fft=1024, hop=256, n_mels=128, f_min=80 Hz, Slaney norm, HTK scale,
# log1p + per-track z-score normalization.  GPU-accelerated batch processing.
#

import argparse
import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image


EPS = 1e-12 

@dataclass(frozen=True)
class MelConfig:
    """
    Configuration for GPU batched mel-spectrogram conversion.

    Assumptions:
    - Values match the training encoder's STFT and mel settings.
    """
    sample_rate: int   = 22_050
    n_mels: int        = 128
    n_fft: int         = 1_024
    hop_length: int    = 256
    f_min: float       = 80.0
    batch_size: int    = 32
    power: float       = 2.0
    device: str        = "auto"


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_audio(audio_path: Path, sr: int) -> torch.Tensor:
    """
    Decode one audio file to mono float32 samples with ffmpeg.

    Assumptions:
    - ffmpeg is installed and can decode the input file format.
    """
    cmd = ["ffmpeg", "-v", "error", "-i", str(audio_path),
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "pipe:1"]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        lines = result.stderr.decode("utf-8", errors="replace").splitlines()
        raise RuntimeError(f"ffmpeg failed for {audio_path}: {lines[-1] if lines else result.returncode}")
    waveform = torch.frombuffer(bytearray(result.stdout), dtype=torch.float32).clone()
    if waveform.numel() == 0:
        raise RuntimeError(f"ffmpeg decoded zero samples from {audio_path}")
    return waveform


def load_batch(paths: list[Path], sr: int) -> tuple[torch.Tensor | None, list[Path], list[int], list[tuple]]:
    """
    Decode and pad a batch of audio files for mel conversion.

    Assumptions:
    - Failed decodes should be reported and skipped rather than aborting the batch.
    """
    waveforms, lengths, valid_paths, skipped = [], [], [], []
    for p in paths:
        try:
            w = load_audio(p, sr)
            waveforms.append(w)
            lengths.append(w.numel())
            valid_paths.append(p)
        except Exception as exc:
            skipped.append((p, str(exc)))
    if not waveforms:
        return None, [], [], skipped
    max_len = max(lengths)
    batch = torch.stack([F.pad(w, (0, max_len - w.numel())) for w in waveforms])
    return batch, valid_paths, lengths, skipped


def log_normalize(mel: torch.Tensor) -> torch.Tensor:
    """
    Apply log compression and per-track z-score normalization to mel tensors.

    Assumptions:
    - The final two dimensions are mel frequency and time.
    """
    mel = torch.log1p(mel)
    mean = mel.mean(dim=(-2, -1), keepdim=True)
    std  = mel.std(dim=(-2, -1), keepdim=True).clamp_min(EPS)
    return (mel - mean) / std


def find_tracks_csv(data_dir: Path) -> Path:
    """
    Locate FMA tracks.csv relative to an audio directory.

    Assumptions:
    - Metadata is either a sibling file or inside sibling fma_metadata.
    """
    for candidate in [data_dir.parent / "tracks.csv",
                      data_dir.parent / "fma_metadata" / "tracks.csv"]:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("could not find tracks.csv")


def load_track_metadata(tracks_csv: Path) -> dict[int, dict]:
    """
    Read FMA metadata needed for manifests from the two-row-header CSV.

    Assumptions:
    - tracks.csv uses the original FMA multi-header column layout.
    """
    result = {}
    with tracks_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        top = next(reader)
        bot = next(reader)
        idx = {(t.strip(), b.strip()): i for i, (t, b) in enumerate(zip(top, bot))}
        split_i  = idx[("set", "split")]
        subset_i = idx[("set", "subset")]
        genre_i  = idx[("track", "genre_top")]
        dur_i    = idx[("track", "duration")]
        title_i  = idx[("track", "title")]
        for row in reader:
            if not row:
                continue
            try:
                tid = int(row[0])
            except ValueError:
                continue
            result[tid] = {
                "split":     row[split_i].strip(),
                "subset":    row[subset_i].strip(),
                "genre_top": row[genre_i].strip(),
                "duration":  row[dur_i].strip(),
                "title":     row[title_i].strip(),
            }
    return result


def write_manifests(data_dir: Path, output_dir: Path) -> dict[str, Path]:
    """
    Write split manifests that connect track ids, audio paths, and mel tensors.

    Assumptions:
    - Mel tensor filenames are numeric FMA track ids.
    """
    tracks_csv = find_tracks_csv(data_dir)
    metadata   = load_track_metadata(tracks_csv)
    fields = ["track_id", "split", "subset", "genre_top", "duration", "title", "audio_path", "mel_path"]
    rows: dict[str, list] = {"all": [], "training": [], "validation": [], "test": []}

    for tensor_path in sorted(output_dir.rglob("*.pt")):
        try:
            tid = int(tensor_path.stem)
        except ValueError:
            continue
        meta = metadata.get(tid)
        if meta is None:
            continue
        audio_rel = (data_dir.parent / data_dir.name / tensor_path.relative_to(output_dir)).with_suffix(".mp3").relative_to(data_dir.parent)
        mel_rel   = tensor_path.relative_to(data_dir.parent)
        split     = meta["split"] or "unknown"
        row = {"track_id": str(tid), "split": split, "subset": meta["subset"],
               "genre_top": meta["genre_top"], "duration": meta["duration"],
               "title": meta["title"], "audio_path": audio_rel.as_posix(),
               "mel_path": mel_rel.as_posix()}
        rows["all"].append(row)
        if split in rows:
            rows[split].append(row)

    manifest_paths = {}
    for name, row_list in rows.items():
        path = output_dir / f"manifest_{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(row_list)
        manifest_paths[name] = path
    return manifest_paths


def convert_directory(data_dir: Path, config: MelConfig = MelConfig()) -> tuple[Path, int, int, dict]:
    """
    Convert an FMA audio directory into mel tensors and split manifests.

    Assumptions:
    - data_dir contains FMA-style mp3 paths and sibling metadata is available.
    """
    audio_files = sorted(p for p in data_dir.rglob("*.mp3") if p.is_file())
    if not audio_files:
        raise FileNotFoundError(f"no mp3 files found under {data_dir}")

    output_dir = data_dir.parent / f"{data_dir.name}_mel"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config.device)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate, n_fft=config.n_fft, win_length=config.n_fft,
        hop_length=config.hop_length, f_min=config.f_min, n_mels=config.n_mels,
        power=config.power, norm="slaney", mel_scale="htk", center=True,
    ).to(device)

    processed = skipped = 0
    print(f"START data_dir={data_dir} output_dir={output_dir} files={len(audio_files)}", flush=True)

    with torch.inference_mode():
        for i in range(0, len(audio_files), config.batch_size):
            batch_paths = audio_files[i : i + config.batch_size]
            batch, valid_paths, lengths, skip_list = load_batch(batch_paths, config.sample_rate)
            skipped += len(skip_list)
            for p, reason in skip_list:
                print(f"[mel] skipped {p}: {reason}", flush=True)
            if batch is None:
                continue
            mel_batch = log_normalize(mel_transform(batch.to(device))).cpu()
            for mel, src, n_samp in zip(mel_batch, valid_paths, lengths):
                rel  = src.relative_to(data_dir).with_suffix(".pt")
                out  = output_dir / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                frames = max(1, 1 + (n_samp // config.hop_length))
                torch.save(mel[:, :frames].contiguous(), out)
                processed += 1

    manifest_paths = write_manifests(data_dir, output_dir)
    print(f"DONE output_dir={output_dir} processed={processed} skipped={skipped}", flush=True)
    return output_dir, processed, skipped, manifest_paths


def write_sample_images(data_dir: Path) -> Path:
    """
    Write one grayscale mel preview image per top-level genre.

    Assumptions:
    - Mel tensors have already been generated for data_dir.
    """
    output_dir = data_dir.parent / f"{data_dir.name}_mel"
    tracks_csv = find_tracks_csv(data_dir)
    genres = {tid: m["genre_top"] for tid, m in load_track_metadata(tracks_csv).items() if m["genre_top"]}
    image_dir = Path(__file__).resolve().parent / "images" / output_dir.name
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for tp in sorted(output_dir.rglob("*.pt")):
        try:
            tid = int(tp.stem)
        except ValueError:
            continue
        genre = genres.get(tid)
        if genre and genre not in seen:
            seen.add(genre)
            mel = torch.load(tp, map_location="cpu").float()
            view = torch.flip(mel, [0])
            view = view - view.min()
            mx = float(view.max())
            if mx > 0:
                view = view / mx
            Image.fromarray((view * 255).to(torch.uint8).numpy(), mode="L").save(image_dir / f"{genre}.png")
    return image_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert FMA audio directory to mel-spectrogram tensors.")
    parser.add_argument("-d", "--data-dir", type=Path, required=True)
    parser.add_argument("--sample-images", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    """
    CLI entry point for converting audio to mel tensors and optional previews.

    Assumptions:
    - The provided data directory exists and contains mp3 files.
    """
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(f"input directory does not exist: {data_dir}")
    config = MelConfig(device=args.device, batch_size=args.batch_size)
    convert_directory(data_dir, config)
    if args.sample_images:
        image_dir = write_sample_images(data_dir)
        print(f"[mel] wrote sample images to {image_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
