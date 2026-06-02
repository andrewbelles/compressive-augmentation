#!/usr/bin/env python3
#
# supcon.py  Andrew Belles  June 2026
#
# Supervised Contrastive encoder training using WaveSTFTEncoder.
# Genre labels define positives across different tracks.
# Intended as a semantically grounded reference manifold for perturbation analysis.
#
# Usage:
#   python -m representation.supcon [--seed N] [--epochs N] [--exclude-genres Pop]
#

import argparse
import math
import signal
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from representation.audio import (
    WaveSTFTEncoder,
    _load_waveform,
    apply_wave_policy,
    load_manifest,
)
from representation.utils import load_config, resolve_device, set_seed
from representation.wave_barlow import DEFAULT_CONFIG, DEFAULT_CONFIG_PATH


DEFAULT_MEL_DIR    = Path("preprocess/data/fma_small_mel")
DEFAULT_AUDIO_ROOT = Path("preprocess/data")

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


class SupConDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        segment_seconds: float,
        sample_rate: int,
        audio_root: Path,
        augment_config: dict,
        seed: int = 0,
        exclude_genres: list[str] | None = None,
    ) -> None:
        manifest = load_manifest(data_dir.resolve(), split)
        if exclude_genres:
            manifest = manifest[~manifest["genre_top"].isin(exclude_genres)]
        manifest = manifest.dropna(subset=["genre_top"])
        self.rows            = manifest.to_dict("records")
        self.segment_seconds = float(segment_seconds)
        self.sample_rate     = int(sample_rate)
        self.audio_root      = audio_root.resolve()
        self.augment_config  = augment_config
        self.seed            = seed
        self.is_train        = (split == "training")
        genres               = sorted({r["genre_top"] for r in self.rows})
        self.genre_to_idx    = {g: i for i, g in enumerate(genres)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row        = self.rows[index]
        audio_path = self.audio_root / Path(row["audio_path"])
        epoch_seed = int(torch.initial_seed()) % (2 ** 31) if self.is_train else 0
        rng        = np.random.default_rng([self.seed, index, epoch_seed])
        offset     = float(rng.uniform(10.0, 25.0))
        y          = _load_waveform(audio_path, self.sample_rate, offset, self.segment_seconds)
        rng1       = np.random.default_rng([self.seed, index, epoch_seed, 1])
        rng2       = np.random.default_rng([self.seed, index, epoch_seed, 2])
        v1  = torch.from_numpy(apply_wave_policy(y, "w3", self.augment_config, rng1)).unsqueeze(0)
        v2  = torch.from_numpy(apply_wave_policy(y, "w3", self.augment_config, rng2)).unsqueeze(0)
        lbl = self.genre_to_idx[row["genre_top"]]
        return v1, v2, lbl


def _run_epoch(
    encoder: WaveSTFTEncoder,
    proj: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    temp: float,
    train: bool,
) -> float:
    encoder.train(train)
    proj.train(train)
    total, n = 0.0, 0
    use_amp  = device.type == "cuda"
    ctx      = torch.enable_grad if train else torch.no_grad
    with ctx():
        for v1, v2, labels in loader:
            v1     = v1.to(device, non_blocking=True)
            v2     = v2.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                feats = torch.cat([proj(encoder(v1)), proj(encoder(v2))], dim=0)
                lbls  = labels.repeat(2)
                loss  = supcon_loss(feats, lbls, temp)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            bs     = v1.size(0)
            total += loss.item() * bs
            n     += bs
    if n == 0:
        raise ValueError("loader produced no valid batches")
    return total / n


def train_supcon(
    data_dir: Path,
    audio_root: Path,
    checkpoint_dir: Path,
    config: dict,
    device: torch.device,
    seed: int,
    temp: float = 0.07,
    proj_dim: int = 128,
) -> Path:
    exclude_genres = list(config.get("exclude_genres", []))
    dataset_name   = str(config.get("dataset", "fma_small"))
    sr             = int(config["sample_rate"])
    seg            = float(config["segment_seconds"])
    wave_augment   = dict(config.get("wave_augment", {}))
    embedding_dim  = int(config["embedding_dims"][0])
    suffix         = "_nopop" if "Pop" in exclude_genres else ""
    source         = f"supcon_w3_d{embedding_dim}{suffix}_s{seed}"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{source}_{dataset_name}.pt"
    if ckpt_path.exists():
        print(f"SKIP source={source} checkpoint exists at {ckpt_path}", flush=True)
        return ckpt_path

    print(f"START source={source}", flush=True)

    ds_kw = dict(segment_seconds=seg, sample_rate=sr, audio_root=audio_root,
                 augment_config=wave_augment, seed=seed, exclude_genres=exclude_genres)
    train_ds = SupConDataset(data_dir, "training",   **ds_kw)
    val_ds   = SupConDataset(data_dir, "validation", **ds_kw)

    nw = int(config["num_workers"])
    bs = int(config["batch_size"])
    loader_kw = dict(batch_size=bs, num_workers=nw, pin_memory=device.type == "cuda",
                     persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None)
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **loader_kw)

    encoder = WaveSTFTEncoder(
        embedding_dim=embedding_dim,
        base_channels=int(config["base_channels"]),
        n_fft=int(config.get("n_fft", 1024)),
        hop_length=int(config.get("hop_length", 256)),
        n_blocks=int(config.get("n_blocks", 3)),
        n_mels=int(config.get("n_mels", 128)),
        sample_rate=sr,
    ).to(device)

    proj = nn.Sequential(
        nn.Linear(embedding_dim, embedding_dim, bias=False),
        nn.BatchNorm1d(embedding_dim), nn.ReLU(inplace=True),
        nn.Linear(embedding_dim, proj_dim),
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(proj.parameters()),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler  = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    epochs  = int(config["epochs"])
    warmup  = int(config["warmup_epochs"])
    base_lr = float(config["learning_rate"])

    stop_requested = False

    def _handle_sigquit(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"source={source} SIGQUIT — stopping after this epoch", flush=True)

    signal.signal(signal.SIGQUIT, _handle_sigquit)

    best_val_loss = float("inf")
    best_encoder_state: dict = {}
    best_epoch = 0
    epoch_history: list[dict] = []

    for epoch in range(epochs):
        if epoch < warmup:
            lr = base_lr * (epoch + 1) / max(warmup, 1)
        else:
            progress = (epoch - warmup) / max(epochs - warmup, 1)
            lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        train_loss = _run_epoch(encoder, proj, train_loader, optimizer, scaler, device, temp, train=True)
        val_loss   = _run_epoch(encoder, proj, val_loader,   optimizer, scaler, device, temp, train=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_encoder_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
            best_epoch = epoch + 1

        epoch_history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"source={source} epoch={epoch+1}/{epochs} train={train_loss:.6f} val={val_loss:.6f}", flush=True)

        if stop_requested:
            break

    torch.save({
        "encoder_state_dict": best_encoder_state,
        "source_name":        source,
        "embedding_dim":      embedding_dim,
        "seed":               seed,
        "dataset":            dataset_name,
        "sample_rate":        sr,
        "segment_seconds":    seg,
        "best_epoch":         best_epoch,
        "best_val_loss":      best_val_loss,
        "epoch_history":      epoch_history,
        "model": {
            "base_channels": int(config["base_channels"]),
            "n_fft":         int(config.get("n_fft", 1024)),
            "hop_length":    int(config.get("hop_length", 256)),
            "n_blocks":      int(config.get("n_blocks", 3)),
            "n_mels":        int(config.get("n_mels", 128)),
            "sample_rate":   sr,
        },
    }, ckpt_path)
    print(f"checkpoint source={source} best_epoch={best_epoch} best_val_loss={best_val_loss:.6f} path={ckpt_path}", flush=True)
    return ckpt_path


@torch.no_grad()
def extract_supcon_embeddings(
    data_dir: Path,
    audio_root: Path,
    ckpt_path: Path,
    output_dir: Path,
    config: dict,
    device: torch.device,
) -> Path:
    payload      = torch.load(ckpt_path, map_location=device, weights_only=False)
    m_cfg        = payload["model"]
    embedding_dim = int(payload["embedding_dim"])
    source        = str(payload["source_name"])
    dataset_name  = str(payload["dataset"])
    sr            = int(payload["sample_rate"])
    seg           = float(payload["segment_seconds"])
    ckpt_seed     = int(payload["seed"])

    encoder = WaveSTFTEncoder(
        embedding_dim=embedding_dim,
        base_channels=int(m_cfg["base_channels"]),
        n_fft=int(m_cfg["n_fft"]),
        hop_length=int(m_cfg["hop_length"]),
        n_blocks=int(m_cfg["n_blocks"]),
        n_mels=int(m_cfg["n_mels"]),
        sample_rate=int(m_cfg["sample_rate"]),
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
                y_full = _load_waveform(audio_path, sr, 0.0, float(config.get("full_track_seconds", 30.0)))
            except Exception:
                continue
            crops = [y_full[s: s + seg_samples] for s in range(0, full_samples - seg_samples + 1, seg_samples)]
            if not crops:
                crops = [y_full[:seg_samples]]
            batch = torch.from_numpy(np.stack(crops)).unsqueeze(1).to(device)
            h = encoder(batch).mean(dim=0)
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
    out_path = output_dir / f"wave_barlow_{dataset_name}.parquet"
    existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    combined = pd.concat([existing, *all_frames], ignore_index=True)
    combined = combined.drop_duplicates(subset=["method", "split", "track_id"], keep="last")
    combined.to_parquet(out_path, index=False)
    print(f"wrote path={out_path} total_rows={len(combined)}", flush=True)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SupCon encoder as semantic reference manifold.")
    parser.add_argument("-d", "--data-dir",     type=Path, default=DEFAULT_MEL_DIR)
    parser.add_argument("--audio-root",         type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("-c", "--config",       type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("-o", "--output-dir",   type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--checkpoint-dir",     type=Path, default=Path(__file__).resolve().parent / "checkpoints")
    parser.add_argument("--seed",               type=int,  default=0)
    parser.add_argument("--temp",               type=float, default=0.07)
    parser.add_argument("--proj-dim",           type=int,  default=128)
    parser.add_argument("--exclude-genres",     type=str,  nargs="*", default=None, metavar="GENRE")
    return parser.parse_args()


def main() -> int:
    args   = parse_args()
    config = load_config(args.config, DEFAULT_CONFIG)
    if args.exclude_genres is not None:
        config["exclude_genres"] = list(args.exclude_genres)

    device = resolve_device(str(config["device"]))
    set_seed(args.seed)

    data_dir       = args.data_dir.expanduser().resolve()
    audio_root     = args.audio_root.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    output_dir     = args.output_dir.expanduser().resolve()

    ckpt = train_supcon(data_dir, audio_root, checkpoint_dir, config, device,
                        seed=args.seed, temp=args.temp, proj_dim=args.proj_dim)
    extract_supcon_embeddings(data_dir, audio_root, ckpt, output_dir, config, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
