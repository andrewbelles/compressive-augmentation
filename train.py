#!/usr/bin/env python3
#
# train.py  Andrew Belles  June 2026
#
# Full-sweep training script: 
# - Trains all CS, traditional, and SupCon encoders on FMA Small.
#
# Usage (two parallel processes, one per H200):
#   CUDA_VISIBLE_DEVICES=0 python train.py --half 0 \
#       --scratch-dir /dartfs-hpc/scratch/$USER/compressive-augmentation &
#   CUDA_VISIBLE_DEVICES=1 python train.py --half 1 \
#       --scratch-dir /dartfs-hpc/scratch/$USER/compressive-augmentation &
#   wait
#

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common.data import WaveBarlowDataset, WaveABTDataset, SupConDataset
from common.model import WaveBarlowModel, WaveSTFTEncoder, barlow_twins_loss
from common.ops import gpu_dct_cs_view_batch, gpu_srht_batch, gpu_wave_policy_batch
from common.supcon import supcon_loss
from common.extract import extract_embeddings
from common.supcon import extract_supcon_embeddings
from common.utils import set_seed


SEEDS          = [0, 7, 17, 31, 42, 53]
SEEDS_TRAD     = [0, 7, 17, 31]
RATIOS         = [1, 5, 10, 20, 40, 60, 70, 80]
POLICIES       = ["w2", "w3", "w4"]
EXCLUDE_GENRES = ["Pop"]
EPOCHS         = 250
WARMUP_EPOCHS  = 10
PEAK_LR        = 4e-3
MIN_LR         = 1e-5
WEIGHT_DECAY   = 1e-4
EMBEDDING_DIM  = 256
BASE_CHANNELS  = 16
N_BLOCKS       = 3
N_FFT          = 1024
HOP_LENGTH     = 256
N_MELS         = 128
SAMPLE_RATE    = 22050
SEGMENT_SEC    = 5.0
FULL_TRACK_SEC = 30.0
PROJ_HIDDEN    = 4096
PROJ_DIM       = 2048
BARLOW_LAMBDA  = 5e-5
SUPCON_TEMP    = 0.07
SUPCON_PROJ    = 128
DATASET_NAME   = "fma_small"

WAVE_AUGMENT = {
    "wave_stretch_scale": [0.8, 1.2],
    "wave_gain_strength": 0.25,
    "wave_n_masks":       2,
    "wave_mask_width":    4410,
    "wave_noise_std":     0.005,
}


@dataclass
class RunSpec:
    """
    Describe one training/extraction run in the experimental sweep.

    Assumptions:
    - kind selects exactly one augmentation or sensing family.
    """
    kind:    str
    seed:    int
    ratio:   Optional[float] = None
    policy:  Optional[str]   = None
    uniform: bool            = False
    srht:    bool            = False


def build_run_list() -> list[RunSpec]:
    """
    Construct the full set of training runs for the current sweep.

    Assumptions:
    - SEEDS, RATIOS, and POLICIES define the canonical experiment grid.
    """
    runs = []
    for seed in SEEDS:
        for ratio in RATIOS:
            runs.append(RunSpec("cs_biased",  seed, ratio=ratio))
            runs.append(RunSpec("cs_uniform", seed, ratio=ratio, uniform=True))
            runs.append(RunSpec("cs_srht",    seed, ratio=ratio, srht=True))
    for seed in SEEDS_TRAD:
        for policy in POLICIES:
            runs.append(RunSpec("traditional", seed, policy=policy))
    for seed in SEEDS_TRAD:
        runs.append(RunSpec("supcon", seed))
    return runs


def source_name(spec: RunSpec) -> str:
    """
    Map a run specification to the stable checkpoint/parquet method name.

    Assumptions:
    - Downstream analysis parses these names to recover family, ratio, and seed.
    """
    suffix = "_nopop"
    seed_tag = f"_s{spec.seed}"
    if spec.kind == "supcon":
        return f"supcon_w3_d{EMBEDDING_DIM}{suffix}{seed_tag}"
    if spec.kind.startswith("cs"):
        sampling = "_srht" if spec.srht else ("_uniform" if spec.uniform else "")
        r = spec.ratio
        ratio_tag = f"{int(r):02d}" if r == int(r) else f"{r:g}".replace(".", "p")
        return f"wave_barlow_cs{sampling}_r{ratio_tag}_d{EMBEDDING_DIM}{suffix}{seed_tag}"
    return f"wave_barlow_abt_{spec.policy}_d{EMBEDDING_DIM}{suffix}{seed_tag}"


def cosine_lr(epoch: int) -> float:
    """
    Compute the warmup plus cosine-decay learning rate for an epoch.

    Assumptions:
    - EPOCHS and WARMUP_EPOCHS define the full schedule length.
    """
    if epoch < WARMUP_EPOCHS:
        return PEAK_LR * (epoch + 1) / max(WARMUP_EPOCHS, 1)
    progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def cache_raw_on_gpu(dataset, device: torch.device) -> torch.Tensor:
    """
    Materialize all raw dataset crops as one tensor on the target device.

    Assumptions:
    - The dataset has been configured to return raw waveform crops.
    """
    tensors = [dataset[i][0] for i in range(len(dataset))]
    return torch.stack(tensors).to(device)


def cache_supcon_on_gpu(dataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Materialize all SupCon raw crops and labels on the target device.

    Assumptions:
    - The dataset has been configured to return raw waveform crops and labels.
    """
    waveforms, labels = [], []
    for i in range(len(dataset)):
        y, lbl = dataset[i]
        waveforms.append(y)
        labels.append(lbl)
    return torch.stack(waveforms).to(device), torch.tensor(labels, device=device)


def _make_gen(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _clone_state(model: nn.Module) -> dict:
    """
    Clone a CPU checkpoint state from regular or torch.compile-wrapped modules.

    Assumptions:
    - Compiled modules expose their original module through _orig_mod.
    """
    src = getattr(model, "_orig_mod", model)
    return {k: v.detach().cpu().clone() for k, v in src.state_dict().items()}


def run_barlow_epoch(
    model, raw: torch.Tensor, optimizer, scaler, device, spec: RunSpec, epoch: int, train: bool,
) -> dict:
    """
    Run one full-batch Barlow Twins train or validation epoch.

    Assumptions:
    - raw already fits on device and view generation is deterministic from epoch.
    """
    model.train(train)
    ctx = torch.enable_grad if train else torch.no_grad
    with ctx():
        y  = raw.squeeze(1) if raw.dim() == 2 else raw
        g1 = _make_gen(device, epoch * 1000 + 1)
        g2 = _make_gen(device, epoch * 1000 + 2)
        if spec.srht:
            v1 = gpu_srht_batch(y, spec.ratio, g1).unsqueeze(1)
            v2 = gpu_srht_batch(y, spec.ratio, g2).unsqueeze(1)
        elif spec.kind.startswith("cs"):
            v1 = gpu_dct_cs_view_batch(y, spec.ratio, g1, uniform=spec.uniform).unsqueeze(1)
            v2 = gpu_dct_cs_view_batch(y, spec.ratio, g2, uniform=spec.uniform).unsqueeze(1)
        else:
            v1 = gpu_wave_policy_batch(y, spec.policy, WAVE_AUGMENT, g1).unsqueeze(1)
            v2 = gpu_wave_policy_batch(y, spec.policy, WAVE_AUGMENT, g2).unsqueeze(1)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, _, z1, z2 = model(v1, v2)
            loss, on_diag, off_diag = barlow_twins_loss(z1, z2, BARLOW_LAMBDA)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    return {"loss": loss.item(), "on_diag": on_diag.item(), "off_diag": off_diag.item()}


def run_supcon_epoch(
    encoder, proj, raw: torch.Tensor, labels: torch.Tensor,
    optimizer, scaler, device, epoch: int, train: bool,
) -> float:
    """
    Run one full-batch SupCon train or validation epoch.

    Assumptions:
    - raw and labels already fit on device and labels align with waveform rows.
    """
    encoder.train(train)
    proj.train(train)
    ctx = torch.enable_grad if train else torch.no_grad
    with ctx():
        g1 = _make_gen(device, epoch * 1000 + 1)
        g2 = _make_gen(device, epoch * 1000 + 2)
        v1 = gpu_wave_policy_batch(raw, "w3", WAVE_AUGMENT, g1).unsqueeze(1)
        v2 = gpu_wave_policy_batch(raw, "w3", WAVE_AUGMENT, g2).unsqueeze(1)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            feats = torch.cat([proj(encoder(v1)), proj(encoder(v2))], dim=0)
            loss  = supcon_loss(feats, labels.repeat(2), SUPCON_TEMP)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    return loss.item()


def train_barlow(
    spec: RunSpec,
    data_dir: Path,
    audio_root: Path,
    checkpoint_dir: Path,
    device: torch.device,
) -> Path:
    """
    Train one Barlow-style encoder and save the best validation checkpoint.

    Assumptions:
    - Predecoded waveforms and split manifests are available under the provided roots.
    """
    source    = source_name(spec)
    ckpt_path = checkpoint_dir / f"{source}_{DATASET_NAME}.pt"
    if ckpt_path.exists():
        print(f"SKIP {source}", flush=True)
        return ckpt_path

    set_seed(spec.seed)
    print(f"START {source}", flush=True)

    ds_kw = dict(
        segment_seconds=SEGMENT_SEC,
        sample_rate=SAMPLE_RATE,
        audio_root=audio_root,
        seed=spec.seed,
        exclude_genres=EXCLUDE_GENRES,
    )
    if spec.kind.startswith("cs"):
        train_ds = WaveBarlowDataset(data_dir, "training",   spec.ratio, **ds_kw,
                                     uniform=spec.uniform, srht=spec.srht, preload=True)
        val_ds   = WaveBarlowDataset(data_dir, "validation", spec.ratio, **ds_kw,
                                     uniform=spec.uniform, srht=spec.srht, preload=True)
    else:
        train_ds = WaveABTDataset(data_dir, "training",   spec.policy, **ds_kw,
                                  augment_config=WAVE_AUGMENT, preload=True)
        val_ds   = WaveABTDataset(data_dir, "validation", spec.policy, **ds_kw,
                                  augment_config=WAVE_AUGMENT, preload=True)
    train_ds._raw_only = True
    val_ds._raw_only   = True

    train_raw = cache_raw_on_gpu(train_ds, device)
    val_raw   = cache_raw_on_gpu(val_ds,   device)

    model = WaveBarlowModel(
        embedding_dim         = EMBEDDING_DIM,
        base_channels         = BASE_CHANNELS,
        projection_hidden_dim = PROJ_HIDDEN,
        projection_dim        = PROJ_DIM,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_blocks=N_BLOCKS, n_mels=N_MELS,
        sample_rate=SAMPLE_RATE,
    ).to(device)
    model     = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
    scaler    = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best_val   = float("inf")
    best_state: dict = {}
    best_epoch = 0
    history: list[dict] = []

    for epoch in range(EPOCHS):
        lr = cosine_lr(epoch)
        set_lr(optimizer, lr)
        tr = run_barlow_epoch(model, train_raw, optimizer, scaler, device, spec, epoch, train=True)
        va = run_barlow_epoch(model, val_raw,   optimizer, scaler, device, spec, epoch, train=False)
        if va["loss"] < best_val:
            best_val   = va["loss"]
            best_state = _clone_state(model)
            best_epoch = epoch + 1
        history.append({"epoch": epoch + 1, "train_loss": tr["loss"], "val_loss": va["loss"]})
        print(
            f"{source} epoch={epoch+1}/{EPOCHS} "
            f"train={tr['loss']:.6f} val={va['loss']:.6f} "
            f"best={best_val:.6f} lr={lr:.2e}",
            flush=True,
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict":      best_state,
        "source_name":     source,
        "mode":            "cs" if spec.kind.startswith("cs") else "traditional",
        "embedding_dim":   EMBEDDING_DIM,
        "ratio":           spec.ratio,
        "policy":          spec.policy,
        "uniform":         spec.uniform,
        "srht":            spec.srht,
        "supervised":      False,
        "seed":            spec.seed,
        "dataset":         DATASET_NAME,
        "sample_rate":     SAMPLE_RATE,
        "segment_seconds": SEGMENT_SEC,
        "best_epoch":      best_epoch,
        "best_val_loss":   best_val,
        "epoch_history":   history,
        "model": {
            "base_channels":         BASE_CHANNELS,
            "projection_hidden_dim": PROJ_HIDDEN,
            "projection_dim":        PROJ_DIM,
            "n_fft":                 N_FFT,
            "hop_length":            HOP_LENGTH,
            "n_blocks":              N_BLOCKS,
            "n_mels":                N_MELS,
            "sample_rate":           SAMPLE_RATE,
        },
    }, ckpt_path)
    print(f"SAVED {source} best_epoch={best_epoch} best_val={best_val:.6f}", flush=True)
    return ckpt_path


def train_supcon(
    spec: RunSpec,
    data_dir: Path,
    audio_root: Path,
    checkpoint_dir: Path,
    device: torch.device,
) -> Path:
    """
    Train one supervised contrastive encoder and save the best validation checkpoint.

    Assumptions:
    - Genre labels in the manifest are the intended supervised classes.
    """
    source    = source_name(spec)
    ckpt_path = checkpoint_dir / f"{source}_{DATASET_NAME}.pt"
    if ckpt_path.exists():
        print(f"SKIP {source}", flush=True)
        return ckpt_path

    set_seed(spec.seed)
    print(f"START {source}", flush=True)

    ds_kw = dict(
        segment_seconds=SEGMENT_SEC,
        sample_rate=SAMPLE_RATE,
        audio_root=audio_root,
        augment_config=WAVE_AUGMENT,
        seed=spec.seed,
        exclude_genres=EXCLUDE_GENRES,
    )
    train_ds = SupConDataset(data_dir, "training",   **ds_kw, preload=True)
    val_ds   = SupConDataset(data_dir, "validation", **ds_kw, preload=True)
    train_ds._raw_only = True
    val_ds._raw_only   = True

    train_raw, train_labels = cache_supcon_on_gpu(train_ds, device)
    val_raw,   val_labels   = cache_supcon_on_gpu(val_ds,   device)

    encoder = WaveSTFTEncoder(
        embedding_dim = EMBEDDING_DIM,
        base_channels = BASE_CHANNELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_blocks=N_BLOCKS, n_mels=N_MELS,
        sample_rate=SAMPLE_RATE,
    ).to(device)

    proj = nn.Sequential(
        nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False),
        nn.BatchNorm1d(EMBEDDING_DIM), nn.ReLU(inplace=True),
        nn.Linear(EMBEDDING_DIM, SUPCON_PROJ),
    ).to(device)

    params    = list(encoder.parameters()) + list(proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
    scaler    = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best_val   = float("inf")
    best_state: dict = {}
    best_epoch = 0
    history: list[dict] = []

    for epoch in range(EPOCHS):
        lr = cosine_lr(epoch)
        set_lr(optimizer, lr)
        tr_loss = run_supcon_epoch(encoder, proj, train_raw, train_labels, optimizer, scaler, device, epoch, train=True)
        va_loss = run_supcon_epoch(encoder, proj, val_raw,   val_labels,   optimizer, scaler, device, epoch, train=False)
        if va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
            best_epoch = epoch + 1
        history.append({"epoch": epoch + 1, "train_loss": tr_loss, "val_loss": va_loss})
        print(
            f"{source} epoch={epoch+1}/{EPOCHS} "
            f"train={tr_loss:.6f} val={va_loss:.6f} "
            f"best={best_val:.6f} lr={lr:.2e}",
            flush=True,
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_state_dict": best_state,
        "source_name":        source,
        "embedding_dim":      EMBEDDING_DIM,
        "seed":               spec.seed,
        "dataset":            DATASET_NAME,
        "sample_rate":        SAMPLE_RATE,
        "segment_seconds":    SEGMENT_SEC,
        "best_epoch":         best_epoch,
        "best_val_loss":      best_val,
        "epoch_history":      history,
        "model": {
            "base_channels": BASE_CHANNELS,
            "n_fft":         N_FFT,
            "hop_length":    HOP_LENGTH,
            "n_blocks":      N_BLOCKS,
            "n_mels":        N_MELS,
            "sample_rate":   SAMPLE_RATE,
        },
    }, ckpt_path)
    print(f"SAVED {source} best_epoch={best_epoch} best_val={best_val:.6f}", flush=True)
    return ckpt_path


def extract(
    spec: RunSpec,
    ckpt_path: Path,
    data_dir: Path,
    audio_root: Path,
    output_dir: Path,
    device: torch.device,
    config: dict,
) -> None:
    """
    Append embeddings for a completed run to the consolidated parquet if needed.

    Assumptions:
    - source_name(spec) matches the checkpoint payload source_name.
    """
    source   = source_name(spec)
    out_path = output_dir / f"wave_barlow_{DATASET_NAME}.parquet"
    if out_path.exists():
        existing = pd.read_parquet(out_path, columns=["method"])
        if source in existing["method"].tolist():
            print(f"SKIP extract {source} already in parquet", flush=True)
            return
    if spec.kind == "supcon":
        extract_supcon_embeddings(data_dir, audio_root, ckpt_path, output_dir, config, device)
    else:
        extract_embeddings(data_dir, audio_root, ckpt_path, output_dir, config, device)


def main() -> None:
    """
    Run the assigned half of the training sweep and extract embeddings.

    Assumptions:
    - Two worker processes split the same ordered run list by even and odd indices.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--half",        type=int, required=True, choices=[0, 1])
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--data-dir",    type=Path, default=Path("preprocess/data/fma_small_mel"))
    parser.add_argument("--audio-root",  type=Path, default=Path("preprocess/data"))
    parser.add_argument("--num-workers", type=int,  default=8)
    parser.add_argument("--kinds",       nargs="+", default=None,
                        help="e.g. cs_biased cs_uniform cs_srht")
    args = parser.parse_args()

    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scratch        = args.scratch_dir.expanduser().resolve()
    checkpoint_dir = scratch / "checkpoints"
    output_dir     = scratch / "data"
    data_dir       = args.data_dir.expanduser().resolve()
    audio_root     = args.audio_root.expanduser().resolve()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset":            DATASET_NAME,
        "sample_rate":        SAMPLE_RATE,
        "segment_seconds":    SEGMENT_SEC,
        "full_track_seconds": FULL_TRACK_SEC,
        "embedding_dims":     [EMBEDDING_DIM],
        "exclude_genres":     EXCLUDE_GENRES,
        "num_workers":        args.num_workers,
        "epochs":             EPOCHS,
        "barlow_lambda":      BARLOW_LAMBDA,
        "base_channels":      BASE_CHANNELS,
        "projection_hidden_dim": PROJ_HIDDEN,
        "projection_dim":     PROJ_DIM,
        "n_fft":              N_FFT,
        "hop_length":         HOP_LENGTH,
        "n_blocks":           N_BLOCKS,
        "n_mels":             N_MELS,
        "wave_augment":       WAVE_AUGMENT,
    }

    all_runs = build_run_list()
    if args.kinds:
        all_runs = [r for r in all_runs if r.kind in args.kinds]
    half_runs = [r for i, r in enumerate(all_runs) if i % 2 == args.half]

    print(f"GPU half={args.half} device={device} total_assigned={len(half_runs)}", flush=True)

    for i, spec in enumerate(half_runs):
        print(f"[{i+1}/{len(half_runs)}] kind={spec.kind} seed={spec.seed} "
              f"ratio={spec.ratio} policy={spec.policy}", flush=True)
        if spec.kind == "supcon":
            ckpt = train_supcon(spec, data_dir, audio_root, checkpoint_dir, device)
        else:
            ckpt = train_barlow(spec, data_dir, audio_root, checkpoint_dir, device)
        extract(spec, ckpt, data_dir, audio_root, output_dir, device, config)

    print(f"DONE half={args.half}", flush=True)


if __name__ == "__main__":
    main()
