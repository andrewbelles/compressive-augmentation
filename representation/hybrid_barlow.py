#!/usr/bin/env python3
#
# hybrid_barlow.py  Andrew Belles  May 22nd, 2026
#
# Hybrid CS-Barlow Twins: view 1 is standard augmentation A_1(X), view 2 is
# CS backprojection applied after standard augmentation C_2(A_2(X)).
#
# Grid axes: augmentation policy × CS ratio × projection_dim
# sensing_pair is fixed per run (not iterated) — set in config.
#

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from compression.train_utils import load_config, resolve_device, set_seed
from representation.audio import (
    BarlowTwinsModel,
    HybridBarlowDataset,
    barlow_twins_loss,
    crop_or_pad,
    load_manifest,
    resolve_relative_data_path,
)
from evaluation.linear import (
    build_estimator,
    build_features,
    compute_metrics,
    encode_labels,
    optimize_hyperparameters,
    split_frames_from_group,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "hybrid_barlow.yaml"
DEFAULT_MEL_DIR = Path("preprocess/data/fma_small_mel")
DEFAULT_CONFIG: dict = {
    "device": "auto",
    "seed": 17,
    "dataset": "fma_small_mel",
    "embedding_dims": [256],
    "sensing_pair": "dct_dct",
    "augmentations": ["a2"],
    "ratios": [3],
    "projection_dims": [128],
    "base_channels": 32,
    "dropout": 0.0,
    "projection_hidden_dim": 1024,
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
    "cs_prob": 1.0,
    "symmetric": False,
    "barlow_lambda": 0.05,
    "augment": {
        "crop_frames": 128,
        "resize_scale": [0.85, 1.0],
        "linear_fader_strength": 0.15,
        "time_mask_width": 24,
        "freq_mask_width": 8,
    },
}

SPLITS = ("training", "validation", "test")


def report(message: str) -> None:
    print(message, flush=True)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Hybrid CS-Barlow Twins encoders.")
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


def get_source_name(aug: str, sensing_pair: str, ratio: int, proj_dim: int, emb_dim: int, cs_prob: float = 1.0, symmetric: bool = False) -> str:
    name = f"hybrid_barlow_{sensing_pair}_{aug}_r{ratio:02d}_p{proj_dim}_d{emb_dim}"
    if symmetric:
        name += "_sym"
    elif cs_prob < 1.0:
        name += f"_cs{int(round(cs_prob * 100))}"
    return name


def cosine_lr(optimizer: torch.optim.Optimizer, epoch: int, epochs: int, warmup: int, base_lr: float) -> None:
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / max(warmup, 1)
    else:
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        lr = base_lr * 0.5 * (1.0 + torch.tensor(progress * 3.14159265358979).cos().item())
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def train_epoch(
    model: BarlowTwinsModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambd: float,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {"loss": 0.0, "on_diag": 0.0, "off_diag": 0.0}
    n = 0
    for v1, v2 in loader:
        if v1.size(0) < 2:
            continue
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)
        _, z1 = model(v1)
        _, z2 = model(v2)
        loss, on_diag, off_diag = barlow_twins_loss(z1, z2, lambd)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        bs = v1.size(0)
        totals["loss"] += loss.item() * bs
        totals["on_diag"] += on_diag.item() * bs
        totals["off_diag"] += off_diag.item() * bs
        n += bs
    denom = max(n, 1)
    return {k: v / denom for k, v in totals.items()}


@torch.no_grad()
def validation_epoch(
    model: BarlowTwinsModel,
    loader: DataLoader,
    device: torch.device,
    lambd: float,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {"loss": 0.0, "on_diag": 0.0, "off_diag": 0.0}
    n = 0
    for v1, v2 in loader:
        if v1.size(0) < 2:
            continue
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)
        _, z1 = model(v1)
        _, z2 = model(v2)
        loss, on_diag, off_diag = barlow_twins_loss(z1, z2, lambd)
        bs = v1.size(0)
        totals["loss"] += loss.item() * bs
        totals["on_diag"] += on_diag.item() * bs
        totals["off_diag"] += off_diag.item() * bs
        n += bs
    if n == 0:
        raise ValueError("validation loader produced no valid batches")
    return {k: v / n for k, v in totals.items()}


def clone_state(model: BarlowTwinsModel) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def _quick_probe(model: BarlowTwinsModel, data_dir: Path, crop_frames: int, device: torch.device) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    import sklearn.metrics as skm

    model.eval()
    splits_data: dict[str, tuple[list, list]] = {}
    for split in ("training", "validation"):
        manifest = load_manifest(data_dir, split)
        embs, labels = [], []
        for _, row in manifest.iterrows():
            rel = Path(str(row["mel_path"]))
            if rel.parts and rel.parts[0] == data_dir.name:
                rel = Path(*rel.parts[1:])
            mel_path = data_dir / rel
            if not mel_path.exists():
                continue
            label = row.get("genre_top", None)
            if label is None or (isinstance(label, float) and np.isnan(label)):
                continue
            mel = torch.load(mel_path, map_location="cpu", weights_only=True).float()
            if mel.ndim != 2:
                continue
            t = mel.shape[1]
            start = (t - crop_frames) // 2 if t >= crop_frames else 0
            mel = mel[:, start:start + crop_frames] if t >= crop_frames else torch.nn.functional.pad(mel, (0, crop_frames - t))
            x = mel.unsqueeze(0).unsqueeze(0).to(device)
            h, _ = model(x)
            embs.append(h.squeeze(0).cpu().numpy())
            labels.append(str(label))
        splits_data[split] = (embs, labels)

    tr_embs, tr_labels = splits_data["training"]
    va_embs, va_labels = splits_data["validation"]
    if not tr_embs or not va_embs:
        return float("nan")

    X_tr = np.stack(tr_embs)
    X_va = np.stack(va_embs)
    le = LabelEncoder().fit(tr_labels)
    y_tr = le.transform(tr_labels)
    y_va = le.transform([l for l in va_labels if l in le.classes_])
    X_va = X_va[[i for i, l in enumerate(va_labels) if l in le.classes_]]

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
        clf.fit(X_tr, y_tr)
    preds = clf.predict(X_va)
    return float(skm.f1_score(y_va, preds, average="macro", zero_division=0))


def train_one(
    data_dir: Path,
    checkpoint_dir: Path,
    aug: str,
    sensing_pair: str,
    ratio: int,
    proj_dim: int,
    embedding_dim: int,
    config: dict,
    device: torch.device,
) -> Path:
    cs_prob = float(config.get("cs_prob", 1.0))
    symmetric = bool(config.get("symmetric", False))
    source = get_source_name(aug, sensing_pair, ratio, proj_dim, embedding_dim, cs_prob, symmetric)
    dataset_name = str(config.get("dataset", data_dir.name))

    train_ds = HybridBarlowDataset(data_dir, "training", aug, config["augment"], sensing_pair, ratio, cs_prob=cs_prob, symmetric=symmetric)
    val_ds = HybridBarlowDataset(data_dir, "validation", aug, config["augment"], sensing_pair, ratio, cs_prob=cs_prob, symmetric=symmetric)

    nw = int(config["num_workers"])
    bs = int(config["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                              pin_memory=device.type == "cuda", drop_last=True,
                              persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw,
                            pin_memory=device.type == "cuda", drop_last=False,
                            persistent_workers=nw > 0)

    model = BarlowTwinsModel(
        embedding_dim=embedding_dim,
        base_channels=int(config["base_channels"]),
        dropout=float(config["dropout"]),
        projector_hidden_dim=int(config["projection_hidden_dim"]),
        projector_dim=proj_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    epochs = int(config["epochs"])
    warmup = int(config["warmup_epochs"])
    base_lr = float(config["learning_rate"])
    lambd = float(config["barlow_lambda"])
    gl2_strip = int(config["gl2_strip"])
    gl5_threshold = float(config["gl5_threshold"])
    gl5_strip = int(config["gl5_strip"])
    gl5_grace = int(config["gl5_grace"])
    crop_frames = int(config["augment"]["crop_frames"])

    best_val_loss = float("inf")
    best_state: dict = {}
    best_epoch = 0
    best_val_metrics: dict[str, float] = {}
    best_probe_f1: float = float("-inf")
    val_loss_history: list[float] = []
    gl2_history: list[float] = []
    epoch_history: list[dict] = []

    for epoch in range(epochs):
        cosine_lr(optimizer, epoch, epochs, warmup, base_lr)
        train_m = train_epoch(model, train_loader, optimizer, device, lambd)
        val_m = validation_epoch(model, val_loader, device, lambd)

        vl = val_m["loss"]
        val_loss_history.append(vl)

        if vl < best_val_loss:
            best_val_loss = vl
            best_val_metrics = dict(val_m)

        probe_f1 = _quick_probe(model, data_dir, crop_frames, device)
        model.train()

        new_best = math.isfinite(probe_f1) and probe_f1 > best_probe_f1
        if new_best:
            best_probe_f1 = probe_f1
            best_state = clone_state(model)
            best_epoch = epoch + 1
            log(f"source={source} new_best epoch={epoch+1} val_loss={vl:.6f} probe_f1={probe_f1:.4f}")

        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": train_m["loss"],
            "val_loss": vl,
            "val_on_diag": val_m["on_diag"],
            "val_off_diag": val_m["off_diag"],
            "probe_val_f1": probe_f1,
        })

        if math.isfinite(probe_f1) and probe_f1 > 0:
            gl2 = 100.0 * (best_probe_f1 / probe_f1 - 1.0)
        else:
            gl2 = 0.0
        gl2_history.append(gl2)

        if len(gl2_history) >= gl2_strip:
            strip_gl2 = gl2_history[-gl2_strip:]
            p_k = max(1000.0 * (sum(strip_gl2) / (gl2_strip * max(min(strip_gl2), 1e-8))), 1e-8)
            gl2_str = f" GL2={gl2:.2f} P_k={p_k:.2f}"
        else:
            gl2_str = f" GL2={gl2:.2f}"

        log(
            f"source={source} epoch={epoch+1}/{epochs} "
            f"train_loss={train_m['loss']:.6f} val_loss={vl:.6f} "
            f"on_diag={val_m['on_diag']:.4f} off_diag={val_m['off_diag']:.4f}"
            f"{gl2_str} probe_f1={probe_f1:.4f}"
        )

        if (epoch + 1 > gl5_grace
                and len(gl2_history) >= gl5_strip
                and all(g > gl5_threshold for g in gl2_history[-gl5_strip:])):
            log(f"source={source} early_stop=GL5 epoch={epoch+1} GL2={gl2:.2f} threshold={gl5_threshold}")
            break

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{source}_{dataset_name}.pt"
    torch.save({
        "state_dict": best_state,
        "source_name": source,
        "embedding_dim": embedding_dim,
        "sensing_pair": sensing_pair,
        "augmentation": aug,
        "ratio": ratio,
        "projection_dim": proj_dim,
        "dataset": dataset_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_probe_f1": best_probe_f1,
        "best_val_on_diag": best_val_metrics.get("on_diag", float("nan")),
        "best_val_off_diag": best_val_metrics.get("off_diag", float("nan")),
        "cs_prob": cs_prob,
        "epoch_history": epoch_history,
        "model": {
            "base_channels": int(config["base_channels"]),
            "dropout": float(config["dropout"]),
            "projection_hidden_dim": int(config["projection_hidden_dim"]),
            "projection_dim": proj_dim,
        },
        "augment": dict(config["augment"]),
    }, ckpt_path)
    report(f"checkpoint source={source} best_epoch={best_epoch} best_probe_f1={best_probe_f1:.4f} path={ckpt_path}")
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
    model = BarlowTwinsModel(
        embedding_dim=int(payload["embedding_dim"]),
        base_channels=int(m_cfg["base_channels"]),
        dropout=0.0,
        projector_hidden_dim=int(m_cfg["projection_hidden_dim"]),
        projector_dim=int(m_cfg["projection_dim"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    source = str(payload["source_name"])
    embedding_dim = int(payload["embedding_dim"])
    ratio = int(payload["ratio"])
    aug = str(payload["augmentation"])
    sensing_pair = str(payload["sensing_pair"])
    proj_dim = int(payload["projection_dim"])
    cs_prob = float(payload.get("cs_prob", 1.0))
    dataset_name = str(payload["dataset"])
    augment_cfg = dict(payload["augment"])
    crop_frames = int(augment_cfg["crop_frames"])

    written: list[Path] = []
    for split in SPLITS:
        manifest = load_manifest(data_dir, split)

        all_embeddings: list[np.ndarray] = []
        track_ids: list = []
        genre_tops: list = []

        for _, row in manifest.iterrows():
            mel_path = resolve_relative_data_path(data_dir, str(row["mel_path"]))
            mel = torch.load(mel_path, map_location="cpu", weights_only=True).float()
            if mel.ndim != 2:
                continue
            mel = crop_or_pad(mel, crop_frames, random_crop=False)
            x = mel.unsqueeze(0).unsqueeze(0).to(device)
            h, _ = model(x)
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
        out_frame["family"] = "hybrid_barlow"
        out_frame["split"] = split
        out_frame["ratio_percent"] = ratio
        out_frame["augmentation"] = aug
        out_frame["sensing_pair"] = sensing_pair
        out_frame["projection_dim"] = proj_dim
        out_frame["m_dim"] = embedding_dim
        out_frame["input_dim"] = embedding_dim
        out_frame["cs_prob"] = cs_prob
        out_frame["seed"] = int(config.get("seed", 0))
        out_frame["dataset"] = dataset_name

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{source}_{dataset_name}_{split}.parquet"
        out_frame.to_parquet(out_path, index=False)
        written.append(out_path)
        report(f"extracted source={source} split={split} n={len(Z)} path={out_path}")

    return written


def _probe_val_f1(parquet_paths: list[Path], probe_config: dict) -> float:
    frames = [pd.read_parquet(p) for p in parquet_paths]
    group = pd.concat(frames, ignore_index=True)
    split_frames, columns = split_frames_from_group(group)
    features = build_features(split_frames, columns)
    _, labels = encode_labels(split_frames)
    best_params, best_val_score = optimize_hyperparameters(
        "logistic", probe_config,
        features["training"], labels["training"],
        features["validation"], labels["validation"],
    )
    estimator = build_estimator(
        "logistic", float(best_params["C"]),
        max_iter=int(probe_config["max_iter"]),
        tol=float(probe_config["tol"]),
        config=probe_config,
    )
    estimator.fit(features["training"], labels["training"])
    n_classes = len(np.unique(labels["training"]))
    val_metrics = compute_metrics(estimator, features["validation"], labels["validation"], n_classes=n_classes)
    return float(val_metrics["f1_macro"])


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_CONFIG)
    device = resolve_device(str(config["device"]))
    set_seed(int(config["seed"]))

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()

    report(f"START module=representation.hybrid_barlow data_dir={data_dir} device={device} config={args.config}")

    sensing_pair = str(config["sensing_pair"])
    augmentations = [str(a) for a in config["augmentations"]]
    ratios = [int(r) for r in config["ratios"]]
    projection_dims = [int(p) for p in config["projection_dims"]]
    embedding_dims = [int(d) for d in config["embedding_dims"]]

    dataset_name = str(config.get("dataset", data_dir.name))

    probe_config = {
        "device": str(config["device"]),
        "seed": int(config["seed"]),
        "torch_epochs": 100,
        "torch_lr": 0.05,
        "torch_batch_size": 2048,
        "max_iter": 1000,
        "tol": 1e-3,
        "c_min": 1e-4,
        "c_max": 1.0,
        "optuna": {"trials": 10, "target_metric": "f1_macro"},
        "knn_neighbors": [5],
    }

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    written_ckpts: list[Path] = []
    ratio_candidates: dict[int, list[tuple[float, Path, list[Path]]]] = {}

    for embedding_dim in embedding_dims:
        for ratio in ratios:
            for aug in augmentations:
                for proj_dim in projection_dims:
                    cs_prob = float(config.get("cs_prob", 1.0))
                    symmetric = bool(config.get("symmetric", False))
                    source = get_source_name(aug, sensing_pair, ratio, proj_dim, embedding_dim, cs_prob, symmetric)
                    dataset_name = str(config.get("dataset", data_dir.name))
                    ckpt_path = checkpoint_dir / f"{source}_{dataset_name}.pt"

                    force_retrain = bool(config.get("force_retrain", False))
                    if not force_retrain and ckpt_path.exists():
                        log(f"skipping source={source} — checkpoint exists")
                        ckpt = ckpt_path
                    else:
                        ckpt = train_one(
                            data_dir, checkpoint_dir,
                            aug, sensing_pair, ratio, proj_dim, embedding_dim,
                            config, device,
                        )
                    written_ckpts.append(ckpt)

                    split_parquets = [output_dir / f"{source}_{dataset_name}_{split}.parquet" for split in SPLITS]
                    if all(p.exists() for p in split_parquets):
                        log(f"skipping extract source={source} — parquets exist")
                        parquet_paths = split_parquets
                    else:
                        parquet_paths = extract_embeddings(data_dir, ckpt, output_dir, config, device)

                    log(f"probing source={source}...")
                    val_f1 = _probe_val_f1(parquet_paths, probe_config)
                    log(f"probe source={source} val_f1={val_f1:.4f}")
                    ratio_candidates.setdefault(ratio, []).append((val_f1, ckpt, parquet_paths))

    report("=== per-ratio best by val_f1 ===")
    best_records = []
    for ratio in sorted(ratio_candidates):
        candidates = ratio_candidates[ratio]
        best_val_f1, best_ckpt, best_parquets = max(candidates, key=lambda t: t[0])
        best_source = torch.load(best_ckpt, map_location="cpu", weights_only=False)["source_name"]
        report(f"ratio={ratio}% best_source={best_source} val_f1={best_val_f1:.4f}")
        best_records.append({
            "ratio": ratio,
            "source": best_source,
            "val_f1": best_val_f1,
            "ckpt": str(best_ckpt),
            "parquets": [str(p) for p in best_parquets],
        })

    selection_path = checkpoint_dir / f"hybrid_barlow_{sensing_pair}_{dataset_name}_best_by_ratio.json"
    selection_path.write_text(json.dumps(best_records, indent=2), encoding="utf-8")
    report(f"saved selection={selection_path}")

    manifest = {
        "dataset": dataset_name,
        "sensing_pair": sensing_pair,
        "checkpoints": [p.as_posix() for p in written_ckpts],
    }
    manifest_path = checkpoint_dir / f"hybrid_barlow_{sensing_pair}_{dataset_name}_checkpoints.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report(f"DONE module=representation.hybrid_barlow checkpoints={len(written_ckpts)} manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
