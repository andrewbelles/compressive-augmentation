from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audio.dataset import load_manifest, load_waveform
from audio.encoder import AudioBarlowModel, AudioSTFTEncoder
from common.extract import write_frames_to_parquet
from csmath.losses import supcon_loss  # noqa: F401

SPLITS = ("training", "validation", "test")


@torch.no_grad()
def extract_embeddings(
    data_dir: Path,
    audio_root: Path,
    ckpt_path: Path,
    output_dir: Path,
    config: dict,
    device: torch.device,
) -> Path:
    """Extract full-track averaged encoder embeddings into the shared parquet."""
    payload      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    source       = str(payload["source_name"])
    dataset_name = str(payload["dataset"])
    out_path     = output_dir / f"wave_barlow_{dataset_name}.parquet"

    if out_path.exists():
        existing = pd.read_parquet(out_path, columns=["method"])
        if source in existing["method"].tolist():
            print(f"SKIP extract source={source} already in parquet", flush=True)
            return out_path

    m_cfg = payload["model"]
    model = AudioBarlowModel(
        embedding_dim         = int(payload["embedding_dim"]),
        base_channels         = int(m_cfg["base_channels"]),
        projection_hidden_dim = int(m_cfg["projection_hidden_dim"]),
        projection_dim        = int(m_cfg["projection_dim"]),
        n_fft                 = int(m_cfg.get("n_fft", 1024)),
        hop_length            = int(m_cfg.get("hop_length", 256)),
        n_blocks              = int(m_cfg.get("n_blocks", 3)),
        n_mels                = int(m_cfg.get("n_mels", 128)),
        sample_rate           = int(m_cfg.get("sample_rate", 22050)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    ckpt_ratio    = payload.get("ratio", None)
    ckpt_srht     = bool(payload.get("srht", False))
    ckpt_uniform  = bool(payload.get("uniform", False))
    ckpt_policy   = payload.get("policy", None)
    ckpt_mode     = str(payload.get("mode", "cs"))
    ckpt_seed     = int(payload.get("seed", 0))
    embedding_dim = int(payload["embedding_dim"])
    sr            = int(payload["sample_rate"])
    seg           = float(payload["segment_seconds"])

    if ckpt_mode == "cs":
        sensing_pair = "srht_srht" if ckpt_srht else ("dct_uniform_dct_uniform" if ckpt_uniform else "dct_dct")
        augmentation = ""
    else:
        sensing_pair = ""
        augmentation = str(ckpt_policy) if ckpt_policy else ""

    seg_samples  = int(sr * seg)
    full_samples = int(sr * float(config.get("full_track_seconds", 30.0)))
    all_frames: list[pd.DataFrame] = []

    for split in SPLITS:
        manifest = load_manifest(data_dir, split)
        embeddings, track_ids, genre_tops = [], [], []
        for _, row in manifest.iterrows():
            audio_path = audio_root / Path(row["audio_path"])
            try:
                y_full = load_waveform(audio_path, sr, 0.0, float(config.get("full_track_seconds", 30.0)))
            except Exception:
                continue
            crops = [y_full[s : s + seg_samples]
                     for s in range(0, full_samples - seg_samples + 1, seg_samples)]
            if not crops:
                crops = [y_full[:seg_samples]]
            batch = torch.from_numpy(np.stack(crops)).unsqueeze(1).to(device)
            h     = model.encoder(batch).mean(dim=0)
            embeddings.append(h.cpu().numpy())
            track_ids.append(row["track_id"])
            genre_tops.append(row.get("genre_top", None))

        if not embeddings:
            continue
        Z      = np.stack(embeddings, axis=0)
        emb_df = pd.DataFrame(Z, columns=[f"embedding_{i:04d}" for i in range(embedding_dim)])
        meta   = pd.DataFrame({"track_id": track_ids, "genre_top": genre_tops})
        frame  = pd.concat([meta, emb_df], axis=1)
        frame["method"]        = source
        frame["family"]        = "wave_barlow"
        frame["split"]         = split
        frame["ratio_percent"] = None if ckpt_ratio is None else int(ckpt_ratio)
        frame["sensing_pair"]  = sensing_pair
        frame["augmentation"]  = augmentation
        frame["encoder_seed"]  = ckpt_seed
        frame["dataset"]       = dataset_name
        all_frames.append(frame)
        print(f"extracted source={source} split={split} n={len(Z)}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_frames_to_parquet(all_frames, out_path, ["method", "split", "track_id"])
    return out_path


@torch.no_grad()
def extract_supcon_embeddings(
    data_dir: Path,
    audio_root: Path,
    ckpt_path: Path,
    output_dir: Path,
    config: dict,
    device: torch.device,
) -> Path:
    """Extract full-track averaged SupCon encoder embeddings into the shared parquet."""
    payload      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    source       = str(payload["source_name"])
    dataset_name = str(payload["dataset"])
    out_path     = output_dir / f"wave_barlow_{dataset_name}.parquet"

    if out_path.exists():
        existing = pd.read_parquet(out_path, columns=["method"])
        if source in existing["method"].tolist():
            print(f"SKIP extract source={source} already in parquet", flush=True)
            return out_path

    m_cfg         = payload["model"]
    embedding_dim = int(payload["embedding_dim"])
    sr            = int(payload["sample_rate"])
    seg           = float(payload["segment_seconds"])
    ckpt_seed     = int(payload["seed"])

    encoder = AudioSTFTEncoder(
        embedding_dim = embedding_dim,
        base_channels = int(m_cfg["base_channels"]),
        n_fft         = int(m_cfg["n_fft"]),
        hop_length    = int(m_cfg["hop_length"]),
        n_blocks      = int(m_cfg["n_blocks"]),
        n_mels        = int(m_cfg["n_mels"]),
        sample_rate   = int(m_cfg["sample_rate"]),
    ).to(device)
    encoder.load_state_dict(payload["encoder_state_dict"])
    encoder.eval()

    seg_samples  = int(sr * seg)
    full_samples = int(sr * float(config.get("full_track_seconds", 30.0)))
    all_frames: list[pd.DataFrame] = []

    for split in SPLITS:
        manifest = load_manifest(data_dir, split)
        embeddings, track_ids, genre_tops = [], [], []
        for _, row in manifest.iterrows():
            audio_path = audio_root / Path(row["audio_path"])
            try:
                y_full = load_waveform(audio_path, sr, 0.0, float(config.get("full_track_seconds", 30.0)))
            except Exception:
                continue
            crops = [y_full[s : s + seg_samples]
                     for s in range(0, full_samples - seg_samples + 1, seg_samples)]
            if not crops:
                crops = [y_full[:seg_samples]]
            batch = torch.from_numpy(np.stack(crops)).unsqueeze(1).to(device)
            h     = encoder(batch).mean(dim=0)
            embeddings.append(h.cpu().numpy())
            track_ids.append(row["track_id"])
            genre_tops.append(row.get("genre_top", None))

        if not embeddings:
            continue
        Z      = np.stack(embeddings)
        emb_df = pd.DataFrame(Z, columns=[f"embedding_{i:04d}" for i in range(embedding_dim)])
        meta   = pd.DataFrame({"track_id": track_ids, "genre_top": genre_tops})
        frame  = pd.concat([meta, emb_df], axis=1)
        frame["method"]        = source
        frame["family"]        = "supcon"
        frame["split"]         = split
        frame["ratio_percent"] = None
        frame["sensing_pair"]  = ""
        frame["augmentation"]  = "w3"
        frame["encoder_seed"]  = ckpt_seed
        frame["dataset"]       = dataset_name
        all_frames.append(frame)
        print(f"extracted source={source} split={split} n={len(Z)}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_frames_to_parquet(all_frames, out_path, ["method", "split", "track_id"])
    return out_path
