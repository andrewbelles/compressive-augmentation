#!/usr/bin/env python3
#
# barlow.py  Andrew Belles  May 8th, 2026
#
# Train Audio Barlow Twins encoders over log-mel crops.
#

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from compression.train_utils import load_config, resolve_device, set_seed
from representation.audio import (
    AUGMENTATION_POLICIES,
    BarlowCropDataset,
    BarlowTwinsModel,
    barlow_twins_loss,
    mixup_batch,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "barlow.yaml"
DEFAULT_MEL_DIR = Path("preprocess/data/fma_small_mel")
DEFAULT_CONFIG = {
    "device": "cuda",
    "seed": 17,
    "embedding_dims": [256],
    "augmentations": ["a0", "a1", "a2", "a3", "a4"],
    "batch_size": 128,
    "num_workers": 4,
    "epochs": 200,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "base_channels": 32,
    "dropout": 0.0,
    "projector_hidden_dim": 1024,
    "projector_dim": 1024,
    "barlow_lambda": 0.005,
    "augment": {
        "crop_frames": 256,
        "resize_scale": [0.85, 1.0],
        "mixup_alpha": 0.2,
        "linear_fader_strength": 0.15,
        "time_mask_width": 24,
        "freq_mask_width": 8,
    },
}


def report(message: str) -> None:
    print(message, flush=True)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Audio Barlow Twins encoders.")
    parser.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        default=DEFAULT_MEL_DIR,
        help=f"Mel tensor directory with manifests. Defaults to {DEFAULT_MEL_DIR}.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config path. Defaults to {DEFAULT_CONFIG_PATH}.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints",
        help="Output directory for model checkpoints. Defaults to representation/checkpoints.",
    )
    return parser.parse_args()


def validate_config(config: dict) -> None:
    invalid = sorted(set(str(value) for value in config["augmentations"]) - set(AUGMENTATION_POLICIES))
    if invalid:
        raise ValueError(f"unsupported augmentation policies: {', '.join(invalid)}")
    if int(config["batch_size"]) <= 1:
        raise ValueError("batch_size must be greater than 1 for Barlow Twins")
    if int(config["epochs"]) <= 0:
        raise ValueError("epochs must be positive")
    if int(config["augment"]["crop_frames"]) <= 0:
        raise ValueError("augment.crop_frames must be positive")


def source_name(embedding_dim: int, policy: str) -> str:
    return f"barlow_d{int(embedding_dim)}_{policy}"


def checkpoint_path(output_dir: Path, embedding_dim: int, policy: str, dataset_name: str) -> Path:
    return output_dir / f"{source_name(embedding_dim, policy)}_{dataset_name}.pt"


def train_epoch(
    model: BarlowTwinsModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict,
    policy: str,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_on_diag = 0.0
    total_off_diag = 0.0
    total_items = 0

    for left, right in loader:
        left = left.to(device, non_blocking=device.type == "cuda")
        right = right.to(device, non_blocking=device.type == "cuda")
        if policy in {"a2", "a3", "a4"}:
            left, right = mixup_batch(left, right, float(config["augment"]["mixup_alpha"]))

        _, left_projection = model(left)
        _, right_projection = model(right)
        loss, on_diag, off_diag = barlow_twins_loss(left_projection, right_projection, float(config["barlow_lambda"]))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = left.size(0)
        total_loss += float(loss.item()) * batch_size
        total_on_diag += float(on_diag.item()) * batch_size
        total_off_diag += float(off_diag.item()) * batch_size
        total_items += batch_size

    denominator = max(1, total_items)
    return {
        "loss": total_loss / denominator,
        "on_diag": total_on_diag / denominator,
        "off_diag": total_off_diag / denominator,
    }


@torch.no_grad()
def validation_epoch(
    model: BarlowTwinsModel,
    loader: DataLoader,
    device: torch.device,
    config: dict,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_on_diag = 0.0
    total_off_diag = 0.0
    total_items = 0

    for left, right in loader:
        if left.size(0) < 2:
            continue
        left = left.to(device, non_blocking=device.type == "cuda")
        right = right.to(device, non_blocking=device.type == "cuda")

        _, left_projection = model(left)
        _, right_projection = model(right)
        loss, on_diag, off_diag = barlow_twins_loss(left_projection, right_projection, float(config["barlow_lambda"]))

        batch_size = left.size(0)
        total_loss += float(loss.item()) * batch_size
        total_on_diag += float(on_diag.item()) * batch_size
        total_off_diag += float(off_diag.item()) * batch_size
        total_items += batch_size

    if total_items == 0:
        raise ValueError("validation loader produced no batches with at least two items")
    return {
        "loss": total_loss / total_items,
        "on_diag": total_on_diag / total_items,
        "off_diag": total_off_diag / total_items,
    }


def clone_state_dict(model: BarlowTwinsModel) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_one(data_dir: Path, output_dir: Path, embedding_dim: int, policy: str, config: dict, device: torch.device) -> Path:
    train_dataset = BarlowCropDataset(data_dir, "training", policy, config["augment"], paired=True)
    validation_dataset = BarlowCropDataset(data_dir, "validation", policy, config["augment"], paired=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["num_workers"]) > 0,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["num_workers"]) > 0,
        drop_last=False,
    )
    model = BarlowTwinsModel(
        embedding_dim=int(embedding_dim),
        base_channels=int(config["base_channels"]),
        dropout=float(config["dropout"]),
        projector_hidden_dim=int(config["projector_hidden_dim"]),
        projector_dim=int(config["projector_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    best_val_loss = float("inf")
    best_state = clone_state_dict(model)
    best_epoch = 0
    best_val_metrics: dict[str, float] = {}

    for epoch in range(1, int(config["epochs"]) + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, config, policy)
        val_metrics = validation_epoch(model, validation_loader, device, config)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_val_metrics = val_metrics
            best_epoch = epoch
            best_state = clone_state_dict(model)

        log(
            f"source={source_name(embedding_dim, policy)} epoch={epoch} "
            f"train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} "
            f"on_diag={val_metrics['on_diag']:.4f} off_diag={val_metrics['off_diag']:.4f} "
            f"best_val_loss={best_val_loss:.6f} best_epoch={best_epoch}"
        )

    output_path = checkpoint_path(output_dir, embedding_dim, policy, data_dir.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "embedding_dim": int(embedding_dim),
        "augmentation": str(policy),
        "source_name": source_name(embedding_dim, policy),
        "dataset": data_dir.name,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_on_diag": float(best_val_metrics.get("on_diag", float("nan"))),
        "best_val_off_diag": float(best_val_metrics.get("off_diag", float("nan"))),
        "model": {
            "base_channels": int(config["base_channels"]),
            "dropout": float(config["dropout"]),
            "projector_hidden_dim": int(config["projector_hidden_dim"]),
            "projector_dim": int(config["projector_dim"]),
        },
        "augment": dict(config["augment"]),
    }, output_path)
    report(
        f"checkpoint source={source_name(embedding_dim, policy)} best_epoch={best_epoch} "
        f"best_val_loss={best_val_loss:.6f} path={output_path}"
    )
    return output_path


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_CONFIG)
    validate_config(config)
    device = resolve_device(str(config["device"]))
    set_seed(int(config["seed"]))

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    report(f"START module=representation.barlow data_dir={data_dir} device={device} config={args.config}")

    written: list[Path] = []
    for embedding_dim in [int(value) for value in config["embedding_dims"]]:
        for policy in [str(value) for value in config["augmentations"]]:
            written.append(train_one(data_dir, output_dir, embedding_dim, policy, config, device))

    manifest = {
        "dataset": data_dir.name,
        "checkpoints": [path.as_posix() for path in written],
    }
    manifest_path = output_dir / f"barlow_{data_dir.name}_checkpoints.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report(f"DONE module=representation.barlow checkpoints={len(written)} manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
