import fcntl
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common.data import load_manifest, load_waveform
from common.model import WaveSTFTEncoder

SPLITS = ("training", "validation", "test")


def supcon_loss(feats: torch.Tensor, labels: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    feats  = F.normalize(feats, dim=1)
    sim    = feats @ feats.T / temp
    n      = feats.size(0)
    labels = labels.view(-1, 1)
    mask   = (labels == labels.T).float()
    mask.fill_diagonal_(0)
    pos_sum = mask.sum(1).clamp_min(1)
    exp_sim = torch.exp(sim) * (1 - torch.eye(n, device=feats.device))
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True).clamp_min(1e-9))
    return (-(mask * log_prob).sum(1) / pos_sum).mean()


@torch.no_grad()
def extract_supcon_embeddings(
    data_dir: Path,
    audio_root: Path,
    ckpt_path: Path,
    output_dir: Path,
    config: dict,
    device: torch.device,
) -> Path:
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

    encoder = WaveSTFTEncoder(
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
            crops = [y_full[s : s + seg_samples] for s in range(0, full_samples - seg_samples + 1, seg_samples)]
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
    lock_path = out_path.with_suffix(".lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
        combined = pd.concat([existing, *all_frames], ignore_index=True)
        combined = combined.drop_duplicates(subset=["method", "split", "track_id"], keep="last")
        combined.to_parquet(out_path, index=False)
    print(f"wrote path={out_path} total_rows={len(combined)}", flush=True)
    return out_path
