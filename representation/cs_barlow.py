#!/usr/bin/env python3
#
# cs_barlow.py  Andrew Belles  May 22nd, 2026
#
# Train CS-Barlow encoders: two independent compressive sensing views of the
# same log-mel spectrogram are fed through a shared encoder and projection head,
# trained with the Barlow Twins objective to learn structure stable across CS sketches.
#

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from compression.train_utils import load_config, resolve_device, set_seed
from representation.audio import (
    CSBarlowModel,
    CSBarlowDataset,
    barlow_twins_loss,
    crop_or_pad,
    load_manifest,
    resolve_relative_data_path,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "cs_barlow.yaml"
DEFAULT_MEL_DIR = Path("preprocess/data/fma_small_mel")
DEFAULT_CONFIG: dict = {
    "device": "auto",
    "seed": 17,
    "dataset": "fma_small_mel",
    "embedding_dims": [256],
    "sensing_pairs": ["srht_srht", "dct_dct"],
    "ratios": [1, 3, 7, 10],
    "base_channels": 32,
    "dropout": 0.0,
    "projection_hidden_dim": 1024,
    "projection_dim": 128,
    "batch_size": 128,
    "num_workers": 4,
    "epochs": 200,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "gl2_strip": 5,
    "gl5_threshold": 2.0,
    "gl5_strip": 5,
    "gl5_grace": 20,
    "up_k_min": 5.0,
    "up_k_strip": 10,
    "barlow_lambda": 0.05,
    "augment": {
        "crop_frames": 128,
        "time_mask_width": 0,
        "freq_mask_width": 0,
    },
}

SPLITS = ("training", "validation", "test")


def report(message: str) -> None:
    print(message, flush=True)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CS-Barlow encoders.")
    parser.add_argument(
        "-d", "--data-dir",
        type=Path,
        default=DEFAULT_MEL_DIR,
        help=f"Mel tensor directory with manifests. Defaults to {DEFAULT_MEL_DIR}.",
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config path. Defaults to {DEFAULT_CONFIG_PATH}.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Output directory for extracted embeddings.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints",
        help="Output directory for model checkpoints.",
    )
    return parser.parse_args()


def get_source_name(embedding_dim: int, sensing_pair: str, ratio: int) -> str:
    return f"cs_barlow_{sensing_pair}_r{ratio:02d}_d{embedding_dim}"


def cosine_lr(optimizer: torch.optim.Optimizer, epoch: int, epochs: int, warmup: int, base_lr: float) -> None:
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / max(warmup, 1)
    else:
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        lr = base_lr * 0.5 * (1.0 + torch.tensor(progress * 3.14159265358979).cos().item())
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def train_epoch(
    model: CSBarlowModel,
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
    model: CSBarlowModel,
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


def clone_state(model: CSBarlowModel) -> dict[str, torch.Tensor]:
    src = getattr(model, "_orig_mod", model)
    return {k: v.detach().cpu().clone() for k, v in src.state_dict().items()}


def train_one(
    data_dir: Path,
    checkpoint_dir: Path,
    embedding_dim: int,
    sensing_pair: str,
    ratio: int,
    config: dict,
    device: torch.device,
) -> Path:
    source = get_source_name(embedding_dim, sensing_pair, ratio)
    dataset_name = str(config.get("dataset", data_dir.name))
    lambd = float(config["barlow_lambda"])

    use_lr = bool(config.get("use_low_rank", False))
    train_ds = CSBarlowDataset(data_dir, "training", sensing_pair, ratio, config["augment"], use_low_rank=use_lr)
    val_ds = CSBarlowDataset(data_dir, "validation", sensing_pair, ratio, config["augment"], use_low_rank=use_lr)

    nw = int(config["num_workers"])
    bs = int(config["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                              pin_memory=device.type == "cuda", drop_last=True,
                              persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw,
                            pin_memory=device.type == "cuda", drop_last=False,
                            persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None)

    model = CSBarlowModel(
        embedding_dim=embedding_dim,
        base_channels=int(config["base_channels"]),
        dropout=float(config["dropout"]),
        projection_hidden_dim=int(config["projection_hidden_dim"]),
        projection_dim=int(config["projection_dim"]),
    ).to(device)
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    epochs = int(config["epochs"])
    warmup = int(config["warmup_epochs"])
    base_lr = float(config["learning_rate"])
    gl2_strip = int(config["gl2_strip"])
    gl5_threshold = float(config["gl5_threshold"])
    gl5_strip = int(config["gl5_strip"])
    gl5_grace = int(config["gl5_grace"])
    up_k_min = float(config["up_k_min"])
    up_k_strip = int(config["up_k_strip"])

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
        train_m = train_epoch(model, train_loader, optimizer, scaler, device, lambd)
        val_m = validation_epoch(model, val_loader, device, lambd)

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

        if epoch + 1 > gl5_grace:
            if (len(gl2_history) >= gl5_strip
                    and all(g > gl5_threshold for g in gl2_history[-gl5_strip:])):
                log(f"source={source} early_stop=sGLt epoch={epoch+1} sGLt={gl2:.2f} threshold={gl5_threshold}")
                break
            finite_pk = [p for p in p_k_history[-up_k_strip:] if p == p]
            if (len(finite_pk) >= up_k_strip
                    and all(p < up_k_min for p in finite_pk)):
                log(f"source={source} early_stop=UP_k epoch={epoch+1} P_k={p_k:.2f}")
                break

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{source}_{dataset_name}.pt"
    torch.save({
        "state_dict": best_state,
        "source_name": source,
        "embedding_dim": embedding_dim,
        "sensing_pair": sensing_pair,
        "ratio": ratio,
        "dataset": dataset_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_on_diag": best_val_metrics.get("on_diag", float("nan")),
        "best_val_off_diag": best_val_metrics.get("off_diag", float("nan")),
        "epoch_history": epoch_history,
        "model": {
            "base_channels": int(config["base_channels"]),
            "dropout": float(config["dropout"]),
            "projection_hidden_dim": int(config["projection_hidden_dim"]),
            "projection_dim": int(config["projection_dim"]),
        },
        "augment": dict(config["augment"]),
    }, ckpt_path)
    report(f"checkpoint source={source} best_epoch={best_epoch} best_val_loss={best_val_loss:.6f} path={ckpt_path}")
    return ckpt_path


@torch.no_grad()
def extract_embeddings(
    data_dir: Path,
    ckpt_path: Path,
    output_dir: Path,
    config: dict,
    device: torch.device,
) -> list[Path]:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    m_cfg = payload["model"]
    model = CSBarlowModel(
        embedding_dim=int(payload["embedding_dim"]),
        base_channels=int(m_cfg["base_channels"]),
        dropout=0.0,
        projection_hidden_dim=int(m_cfg["projection_hidden_dim"]),
        projection_dim=int(m_cfg["projection_dim"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    source = str(payload["source_name"])
    embedding_dim = int(payload["embedding_dim"])
    ratio = int(payload["ratio"])
    dataset_name = str(payload["dataset"])
    augment_cfg = dict(payload["augment"])
    crop_frames = int(augment_cfg["crop_frames"])

    all_frames: list[pd.DataFrame] = []
    for split in SPLITS:
        manifest = load_manifest(data_dir, split)

        all_embeddings: list[np.ndarray] = []
        track_ids: list = []
        genre_tops: list = []

        use_lr = bool(config.get("use_low_rank", False))
        for _, row in manifest.iterrows():
            mel_path = resolve_relative_data_path(data_dir, str(row["mel_path"]), use_lr)
            mel = torch.load(mel_path, map_location="cpu", weights_only=True).float()
            if mel.ndim != 2:
                continue
            mel = crop_or_pad(mel, crop_frames, random_crop=False)
            x = mel.unsqueeze(0).unsqueeze(0).to(device)
            h = model.encoder(x)
            all_embeddings.append(h.squeeze(0).cpu().numpy())
            track_ids.append(row["track_id"])
            genre_tops.append(row.get("genre_top", None))

        if not all_embeddings:
            continue

        Z = np.stack(all_embeddings, axis=0)
        emb_df = pd.DataFrame(Z, columns=[f"embedding_{i:04d}" for i in range(embedding_dim)])
        meta_df = pd.DataFrame({"track_id": track_ids, "genre_top": genre_tops})
        out_frame = pd.concat([meta_df, emb_df], axis=1)
        out_frame["method"] = source
        out_frame["family"] = "cs_barlow"
        out_frame["split"] = split
        out_frame["ratio_percent"] = ratio
        out_frame["sensing_pair"] = sensing_pair
        out_frame["m_dim"] = embedding_dim
        out_frame["input_dim"] = embedding_dim
        out_frame["seed"] = int(config.get("seed", 0))
        out_frame["dataset"] = dataset_name
        all_frames.append(out_frame)
        report(f"extracted source={source} split={split} n={len(Z)}")

    if not all_frames:
        return []
    out_path = output_dir / f"cs_barlow_{dataset_name}.parquet"
    existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    combined = pd.concat([existing, *all_frames], ignore_index=True)
    combined = combined.drop_duplicates(subset=["method", "split", "track_id"], keep="last")
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    report(f"wrote path={out_path} total_rows={len(combined)}")
    return [out_path]


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_CONFIG)
    device = resolve_device(str(config["device"]))
    set_seed(int(config["seed"]))

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()

    report(f"START module=representation.cs_barlow data_dir={data_dir} device={device} config={args.config}")

    embedding_dims = [int(d) for d in config["embedding_dims"]]
    sensing_pairs = [str(p) for p in config["sensing_pairs"]]
    ratios = [int(r) for r in config["ratios"]]

    written_ckpts: list[Path] = []
    for embedding_dim in embedding_dims:
        for sensing_pair in sensing_pairs:
            for ratio in ratios:
                source = get_source_name(embedding_dim, sensing_pair, ratio)
                dataset_name = str(config.get("dataset", data_dir.name))
                ckpt_path = checkpoint_dir / f"{source}_{dataset_name}.pt"
                if ckpt_path.exists():
                    log(f"skipping source={source} — checkpoint exists")
                    ckpt = ckpt_path
                else:
                    ckpt = train_one(data_dir, checkpoint_dir, embedding_dim, sensing_pair, ratio, config, device)
                written_ckpts.append(ckpt)
                extract_embeddings(data_dir, ckpt, output_dir, config, device)

    manifest = {
        "dataset": data_dir.name,
        "checkpoints": [p.as_posix() for p in written_ckpts],
    }
    manifest_path = checkpoint_dir / f"cs_barlow_{data_dir.name}_checkpoints.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report(f"DONE module=representation.cs_barlow checkpoints={len(written_ckpts)} manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
