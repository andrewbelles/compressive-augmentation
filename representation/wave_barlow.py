#!/usr/bin/env python3
#
# wave_barlow.py  Andrew Belles  May 2026
#
# Train Waveform Barlow Twins encoders in three modes:
#   cs         -- two independent DCT CS backprojection views
#   traditional -- two independent waveform augmentation views (w1/w2/w3)
#   hybrid      -- view 1 = CS backprojection, view 2 = waveform augmentation
#

import argparse
import json
import math
import signal
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from compression.train_utils import load_config, resolve_device, set_seed
from representation.audio import (
    HybridWaveDataset,
    WaveABTDataset,
    WaveBarlowDataset,
    WaveBarlowModel,
    _load_waveform,
    barlow_twins_loss,
    load_manifest,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "wave_barlow.yaml"
DEFAULT_MEL_DIR = Path("preprocess/data/fma_small_mel")
DEFAULT_AUDIO_ROOT = Path("preprocess/data")
MODES = ("cs", "traditional", "hybrid")

DEFAULT_CONFIG: dict = {
    "device": "auto",
    "seed": 17,
    "dataset": "fma_small",
    "sample_rate": 22050,
    "segment_seconds": 5.0,
    "full_track_seconds": 30.0,
    "mode": "cs",
    "embedding_dims": [256],
    "ratios": [20],
    "policies": ["w2", "w3"],
    "exclude_genres": [],
    "n_fft": 1024,
    "hop_length": 256,
    "n_mels": 128,
    "base_channels": 16,
    "n_blocks": 3,
    "projection_hidden_dim": 4096,
    "projection_dim": 2048,
    "batch_size": 256,
    "num_workers": 8,
    "epochs": 300,
    "learning_rate": 1.2e-3,
    "weight_decay": 1e-4,
    "warmup_epochs": 20,
    "gl2_strip": 5,
    "gl5_threshold": 3.0,
    "gl5_strip": 5,
    "gl5_grace": 30,
    "up_k_min": 5.0,
    "up_k_strip": 10,
    "barlow_lambda": 5e-5,
    "force_retrain": False,
    "wave_augment": {
        "wave_stretch_scale": [0.8, 1.2],
        "wave_gain_strength": 0.25,
        "wave_n_masks": 2,
        "wave_mask_width": 4410,
        "wave_noise_std": 0.005,
    },
}

SPLITS = ("training", "validation", "test")


def report(message: str) -> None:
    print(message, flush=True)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Waveform Barlow Twins encoders.")
    parser.add_argument(
        "-d", "--data-dir", type=Path, default=DEFAULT_MEL_DIR,
        help="Mel manifest directory. Defaults to preprocess/data/fma_small_mel.",
    )
    parser.add_argument(
        "--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT,
        help="Root directory containing raw audio files. Defaults to preprocess/data.",
    )
    parser.add_argument(
        "-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help=f"YAML config path. Defaults to {DEFAULT_CONFIG_PATH}.",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Output directory for extracted embeddings.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path(__file__).resolve().parent / "checkpoints",
        help="Output directory for model checkpoints.",
    )
    parser.add_argument(
        "--mode", type=str, choices=MODES, default=None,
        help="Training mode override: cs, traditional, or hybrid. Overrides config.",
    )
    parser.add_argument(
        "--exclude-genres", type=str, nargs="*", default=None, metavar="GENRE",
        help="Genres to exclude from training (e.g. --exclude-genres Pop). Overrides config.",
    )
    return parser.parse_args()


def get_source_name(
    mode: str,
    embedding_dim: int,
    ratio: int | None,
    policy: str | None,
    exclude_genres: list[str] | None = None,
) -> str:
    suffix = "_nopop" if exclude_genres and "Pop" in exclude_genres else ""
    if mode == "cs":
        return f"wave_barlow_cs_r{ratio:02d}_d{embedding_dim}{suffix}"
    if mode == "traditional":
        return f"wave_barlow_abt_{policy}_d{embedding_dim}{suffix}"
    return f"wave_barlow_hybrid_{policy}_r{ratio:02d}_d{embedding_dim}{suffix}"


def cosine_lr(optimizer: torch.optim.Optimizer, epoch: int, epochs: int, warmup: int, base_lr: float) -> None:
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / max(warmup, 1)
    else:
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        lr = base_lr * 0.5 * (1.0 + torch.tensor(progress * math.pi).cos().item())
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def train_epoch(
    model: WaveBarlowModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    lambd: float,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {"loss": 0.0, "on_diag": 0.0, "off_diag": 0.0}
    n = 0
    use_amp = device.type == "cuda"
    for v1, v2 in loader:
        if v1.size(0) < 2:
            continue
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            _, _, z1, z2 = model(v1, v2)
            loss, on_diag, off_diag = barlow_twins_loss(z1, z2, lambd)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = v1.size(0)
        totals["loss"] += loss.item() * bs
        totals["on_diag"] += on_diag.item() * bs
        totals["off_diag"] += off_diag.item() * bs
        n += bs
    denom = max(n, 1)
    return {k: v / denom for k, v in totals.items()}


@torch.no_grad()
def validation_epoch(
    model: WaveBarlowModel,
    loader: DataLoader,
    device: torch.device,
    lambd: float,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {"loss": 0.0, "on_diag": 0.0, "off_diag": 0.0}
    n = 0
    use_amp = device.type == "cuda"
    for v1, v2 in loader:
        if v1.size(0) < 2:
            continue
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            _, _, z1, z2 = model(v1, v2)
            loss, on_diag, off_diag = barlow_twins_loss(z1, z2, lambd)
        bs = v1.size(0)
        totals["loss"] += loss.item() * bs
        totals["on_diag"] += on_diag.item() * bs
        totals["off_diag"] += off_diag.item() * bs
        n += bs
    if n == 0:
        raise ValueError("validation loader produced no valid batches")
    return {k: v / n for k, v in totals.items()}


def clone_state(model: WaveBarlowModel) -> dict[str, torch.Tensor]:
    src = getattr(model, "_orig_mod", model)
    return {k: v.detach().cpu().clone() for k, v in src.state_dict().items()}


def _build_dataset(
    mode: str,
    data_dir: Path,
    split: str,
    ratio: int | None,
    policy: str | None,
    seg: float,
    sr: int,
    audio_root: Path,
    wave_augment: dict,
    seed: int,
    exclude_genres: list[str] | None = None,
):
    if mode == "cs":
        return WaveBarlowDataset(data_dir, split, ratio, seg, sr, audio_root, seed=seed, exclude_genres=exclude_genres)
    if mode == "traditional":
        return WaveABTDataset(data_dir, split, policy, seg, sr, audio_root, wave_augment, seed=seed, exclude_genres=exclude_genres)
    return HybridWaveDataset(data_dir, split, ratio, policy, seg, sr, audio_root, wave_augment, seed=seed, exclude_genres=exclude_genres)


def train_one(
    data_dir: Path,
    audio_root: Path,
    checkpoint_dir: Path,
    mode: str,
    embedding_dim: int,
    ratio: int | None,
    policy: str | None,
    config: dict,
    device: torch.device,
) -> Path:
    exclude_genres = list(config.get("exclude_genres", []))
    source = get_source_name(mode, embedding_dim, ratio, policy, exclude_genres)
    dataset_name = str(config.get("dataset", "fma_small"))
    lambd = float(config["barlow_lambda"])
    sr = int(config["sample_rate"])
    seg = float(config["segment_seconds"])
    seed = int(config["seed"])
    wave_augment = dict(config.get("wave_augment", {}))

    train_ds = _build_dataset(mode, data_dir, "training",   ratio, policy, seg, sr, audio_root, wave_augment, seed, exclude_genres)
    val_ds   = _build_dataset(mode, data_dir, "validation", ratio, policy, seg, sr, audio_root, wave_augment, seed, exclude_genres)

    nw = int(config["num_workers"])
    bs = int(config["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                              pin_memory=device.type == "cuda", drop_last=True,
                              persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw,
                            pin_memory=device.type == "cuda", drop_last=False,
                            persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None)

    model = WaveBarlowModel(
        embedding_dim=embedding_dim,
        base_channels=int(config["base_channels"]),
        projection_hidden_dim=int(config["projection_hidden_dim"]),
        projection_dim=int(config["projection_dim"]),
        n_fft=int(config.get("n_fft", 1024)),
        hop_length=int(config.get("hop_length", 256)),
        n_blocks=int(config.get("n_blocks", 3)),
        n_mels=int(config.get("n_mels", 128)),
        sample_rate=int(config.get("sample_rate", 22050)),
    ).to(device)
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    epochs    = int(config["epochs"])
    warmup    = int(config["warmup_epochs"])
    base_lr   = float(config["learning_rate"])
    gl2_strip = int(config["gl2_strip"])
    gl5_thr   = float(config["gl5_threshold"])
    gl5_strip = int(config["gl5_strip"])
    gl5_grace = int(config["gl5_grace"])
    up_k_min  = float(config["up_k_min"])
    up_k_strip = int(config["up_k_strip"])

    stop_requested = False

    def _handle_sigquit(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        log(f"source={source} SIGQUIT received — will checkpoint after this epoch")

    signal.signal(signal.SIGQUIT, _handle_sigquit)

    best_val_loss = float("inf")
    best_state: dict = {}
    best_epoch = 0
    best_val_metrics: dict[str, float] = {}
    val_loss_history: list[float] = []
    gl2_history: list[float] = []
    p_k_history: list[float] = []
    epoch_history: list[dict] = []

    for epoch in range(epochs):
        cosine_lr(optimizer, epoch, epochs, warmup, base_lr)
        train_ds.epoch = epoch
        val_ds.epoch = epoch
        train_m = train_epoch(model, train_loader, optimizer, scaler, device, lambd)
        val_m   = validation_epoch(model, val_loader, device, lambd)
        vl = val_m["loss"]
        val_loss_history.append(vl)

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = clone_state(model)
            best_epoch = epoch + 1
            best_val_metrics = dict(val_m)

        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": train_m["loss"],
            "val_loss": vl,
            "val_on_diag": val_m["on_diag"],
            "val_off_diag": val_m["off_diag"],
        })

        gl2 = 100.0 * (vl / best_val_loss - 1.0)
        gl2_history.append(gl2)

        p_k = float("nan")
        if len(val_loss_history) >= gl2_strip:
            strip = val_loss_history[-gl2_strip:]
            strip_min = min(strip)
            p_k = max(1000.0 * (sum(strip) / (gl2_strip * strip_min) - 1.0), 1e-8)
            gl2_str = f" sGLt={gl2:.2f} P_k={p_k:.2f}"
        else:
            gl2_str = f" sGLt={gl2:.2f}"
        p_k_history.append(p_k)

        log(
            f"source={source} epoch={epoch+1}/{epochs} "
            f"train_loss={train_m['loss']:.6f} val_loss={vl:.6f} "
            f"on_diag={val_m['on_diag']:.4f} off_diag={val_m['off_diag']:.4f}"
            f"{gl2_str}"
        )

        if stop_requested:
            log(f"source={source} stopping at epoch={epoch+1} by request")
            break

        if epoch + 1 > gl5_grace:
            if len(gl2_history) >= gl5_strip and all(g > gl5_thr for g in gl2_history[-gl5_strip:]):
                log(f"source={source} early_stop=sGLt epoch={epoch+1}")
                break
            finite_pk = [p for p in p_k_history[-up_k_strip:] if p == p]
            if len(finite_pk) >= up_k_strip and all(p < up_k_min for p in finite_pk):
                log(f"source={source} early_stop=UP_k epoch={epoch+1}")
                break

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{source}_{dataset_name}.pt"
    torch.save({
        "state_dict": best_state,
        "source_name": source,
        "mode": mode,
        "embedding_dim": embedding_dim,
        "ratio": ratio,
        "policy": policy,
        "dataset": dataset_name,
        "sample_rate": sr,
        "segment_seconds": seg,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_on_diag": best_val_metrics.get("on_diag", float("nan")),
        "best_val_off_diag": best_val_metrics.get("off_diag", float("nan")),
        "epoch_history": epoch_history,
        "model": {
            "base_channels": int(config["base_channels"]),
            "projection_hidden_dim": int(config["projection_hidden_dim"]),
            "projection_dim": int(config["projection_dim"]),
            "n_fft": int(config.get("n_fft", 1024)),
            "hop_length": int(config.get("hop_length", 256)),
            "n_blocks": int(config.get("n_blocks", 3)),
            "n_mels": int(config.get("n_mels", 128)),
            "sample_rate": int(config.get("sample_rate", 22050)),
        },
    }, ckpt_path)
    report(f"checkpoint source={source} best_epoch={best_epoch} best_val_loss={best_val_loss:.6f} path={ckpt_path}")
    return ckpt_path


@torch.no_grad()
def extract_embeddings(
    data_dir: Path,
    audio_root: Path,
    ckpt_path: Path,
    output_dir: Path,
    config: dict,
    device: torch.device,
) -> Path:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    m_cfg = payload["model"]
    model = WaveBarlowModel(
        embedding_dim=int(payload["embedding_dim"]),
        base_channels=int(m_cfg["base_channels"]),
        projection_hidden_dim=int(m_cfg["projection_hidden_dim"]),
        projection_dim=int(m_cfg["projection_dim"]),
        n_fft=int(m_cfg.get("n_fft", 1024)),
        hop_length=int(m_cfg.get("hop_length", 256)),
        n_blocks=int(m_cfg.get("n_blocks", 3)),
        n_mels=int(m_cfg.get("n_mels", 128)),
        sample_rate=int(m_cfg.get("sample_rate", 22050)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    source = str(payload["source_name"])
    ckpt_mode = str(payload.get("mode", "cs"))
    ckpt_policy = payload.get("policy", None)
    ckpt_ratio = payload.get("ratio", None)
    embedding_dim = int(payload["embedding_dim"])
    ratio_val = None if ckpt_ratio is None else int(ckpt_ratio)
    dataset_name = str(payload["dataset"])
    sr = int(payload["sample_rate"])
    seg = float(payload["segment_seconds"])

    if ckpt_mode == "cs":
        sensing_pair_val = "dct_dct"
        augmentation_val = ""
    elif ckpt_mode == "traditional":
        sensing_pair_val = ""
        augmentation_val = str(ckpt_policy) if ckpt_policy else ""
    else:
        sensing_pair_val = "dct_wave"
        augmentation_val = str(ckpt_policy) if ckpt_policy else ""

    seg_samples = int(sr * seg)
    full_seconds = float(config.get("full_track_seconds", 30.0))
    full_samples = int(sr * full_seconds)

    all_frames: list[pd.DataFrame] = []
    for split in SPLITS:
        manifest = load_manifest(data_dir, split)
        embeddings, track_ids, genre_tops = [], [], []

        for _, row in manifest.iterrows():
            audio_path = audio_root / Path(row["audio_path"])
            try:
                y_full = _load_waveform(audio_path, sr, 0.0, full_seconds)
            except Exception:
                continue
            crops = []
            for start in range(0, full_samples - seg_samples + 1, seg_samples):
                crops.append(y_full[start : start + seg_samples])
            if not crops:
                crops = [y_full[:seg_samples]]
            batch = torch.from_numpy(np.stack(crops)).unsqueeze(1).to(device)
            h = model.encoder(batch).mean(dim=0)
            embeddings.append(h.cpu().numpy())
            track_ids.append(row["track_id"])
            genre_tops.append(row.get("genre_top", None))

        if not embeddings:
            continue

        Z = np.stack(embeddings, axis=0)
        emb_df = pd.DataFrame(Z, columns=[f"embedding_{i:04d}" for i in range(embedding_dim)])
        meta_df = pd.DataFrame({"track_id": track_ids, "genre_top": genre_tops})
        out_frame = pd.concat([meta_df, emb_df], axis=1)
        out_frame["method"] = source
        out_frame["family"] = "wave_barlow"
        out_frame["split"] = split
        out_frame["ratio_percent"] = ratio_val
        out_frame["sensing_pair"] = sensing_pair_val
        out_frame["augmentation"] = augmentation_val
        out_frame["seed"] = int(config.get("seed", 0))
        out_frame["dataset"] = dataset_name
        all_frames.append(out_frame)
        report(f"extracted source={source} split={split} n={len(Z)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"wave_barlow_{dataset_name}.parquet"
    existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    combined = pd.concat([existing, *all_frames], ignore_index=True)
    combined = combined.drop_duplicates(subset=["method", "split", "track_id"], keep="last")
    combined.to_parquet(out_path, index=False)
    report(f"wrote path={out_path} total_rows={len(combined)}")
    return out_path


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_CONFIG)
    device = resolve_device(str(config["device"]))
    set_seed(int(config["seed"]))

    data_dir    = args.data_dir.expanduser().resolve()
    audio_root  = args.audio_root.expanduser().resolve()
    output_dir  = args.output_dir.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    force_retrain = bool(config.get("force_retrain", False))

    report(f"START module=representation.wave_barlow data_dir={data_dir} audio_root={audio_root} device={device}")

    mode = str(args.mode if args.mode is not None else config.get("mode", "cs"))
    embedding_dims = [int(d) for d in config["embedding_dims"]]
    ratios = [int(r) for r in config.get("ratios", [20])]
    policies = [str(p) for p in config.get("policies", ["w2"])]
    dataset_name = str(config.get("dataset", "fma_small"))
    if args.exclude_genres is not None:
        exclude_genres = list(args.exclude_genres)
        config["exclude_genres"] = exclude_genres
    else:
        exclude_genres = list(config.get("exclude_genres", []))

    if mode == "cs":
        grid = [(dim, r, None) for dim in embedding_dims for r in ratios]
    elif mode == "traditional":
        grid = [(dim, None, p) for dim in embedding_dims for p in policies]
    else:
        grid = [(dim, r, p) for dim in embedding_dims for r in ratios for p in policies]

    written_ckpts: list[Path] = []
    for embedding_dim, ratio, policy in grid:
        source = get_source_name(mode, embedding_dim, ratio, policy, exclude_genres)
        ckpt_path = checkpoint_dir / f"{source}_{dataset_name}.pt"
        if ckpt_path.exists() and not force_retrain:
            log(f"skipping source={source} — checkpoint exists")
            ckpt = ckpt_path
        else:
            ckpt = train_one(data_dir, audio_root, checkpoint_dir, mode, embedding_dim, ratio, policy, config, device)
        written_ckpts.append(ckpt)
        extract_embeddings(data_dir, audio_root, ckpt, output_dir, config, device)

    manifest = {
        "dataset": dataset_name,
        "checkpoints": [p.as_posix() for p in written_ckpts],
    }
    manifest_path = checkpoint_dir / f"wave_barlow_{dataset_name}_checkpoints.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report(f"DONE module=representation.wave_barlow checkpoints={len(written_ckpts)} manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
