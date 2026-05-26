#!/usr/bin/env python3
#
# cs_reconstruction_error.py  Andrew Belles  May 2026
#
# Measures backprojection PSNR vs m/N ratio on mel crops from the
# training split.  Produces a table (stdout) and a PSNR-vs-ratio plot.
#

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

DEFAULT_MEL_DIR = Path("preprocess/data/fma_small_mel")
DEFAULT_CONFIG = Path("configs/cs_reconstruction_error.yaml")
DEFAULT_OUT = Path("evaluation/images/cs_reconstruction_psnr.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backprojection PSNR vs m/N.")
    p.add_argument("-d", "--data-dir", type=Path, default=DEFAULT_MEL_DIR)
    p.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--method", type=str, default=None,
                   help="Override config method: dct | gaussian")
    return p.parse_args()


def load_config(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_manifest(data_dir: Path, split: str):
    import pandas as pd
    for name in (f"{split}.csv", f"manifest_{split}.csv"):
        p = data_dir / name
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(f"no manifest for split={split} in {data_dir}")


def crop_or_pad(mel: torch.Tensor, frames: int) -> torch.Tensor:
    t = mel.shape[1]
    if t >= frames:
        start = (t - frames) // 2
        return mel[:, start:start + frames]
    return torch.nn.functional.pad(mel, (0, frames - t))


def dct_backproject(x_flat: np.ndarray, m: int) -> np.ndarray:
    from scipy.fft import dct, idct
    d = len(x_flat)
    m = min(m, d)
    coeffs = dct(x_flat.astype(np.float64), norm="ortho")
    probs = 1.0 / (np.arange(1, d + 1) ** 0.5)
    probs /= probs.sum()
    idx = np.random.choice(d, size=m, replace=False, p=probs)
    z = np.zeros(d, dtype=np.float64)
    z[idx] = coeffs[idx]
    return idct(z, norm="ortho").astype(np.float32)



def gaussian_backproject(x_flat: np.ndarray, m: int) -> np.ndarray:
    d = len(x_flat)
    m = min(m, d)
    Phi = np.random.randn(m, d) / math.sqrt(m)
    y = Phi @ x_flat.astype(np.float64)
    return (Phi.T @ y).astype(np.float32)


METHODS = {"dct": dct_backproject, "gaussian": gaussian_backproject}


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    peak = float(np.abs(original).max())
    if peak == 0.0:
        return float("nan")
    mse = float(np.mean((original - reconstructed) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(peak ** 2 / mse)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    method_name = args.method or str(cfg.get("method", "dct"))
    if method_name not in METHODS:
        print(f"ERROR: unknown method {method_name!r}, choose from {list(METHODS)}", file=sys.stderr)
        return 1
    backproject = METHODS[method_name]

    n_samples = int(cfg.get("n_samples", 200))
    crop_frames = int(cfg.get("crop_frames", 256))
    seed = int(cfg.get("seed", 0))
    ratios = [int(r) for r in cfg.get("ratios", list(range(1, 100)))]
    train_ratios = [int(r) for r in cfg.get("train_ratios", [])]

    np.random.seed(seed)
    data_dir = args.data_dir.expanduser().resolve()
    manifest = load_manifest(data_dir, "training")
    manifest = manifest.sample(
        n=min(n_samples, len(manifest)),
        random_state=seed,
    ).reset_index(drop=True)

    crops: list[np.ndarray] = []
    for _, row in manifest.iterrows():
        rel = Path(str(row["mel_path"]))
        if rel.parts and rel.parts[0] == data_dir.name:
            rel = Path(*rel.parts[1:])
        mel_path = data_dir / rel
        if not mel_path.exists():
            continue
        mel = torch.load(mel_path, map_location="cpu", weights_only=True).float()
        if mel.ndim != 2:
            continue
        mel = crop_or_pad(mel, crop_frames)
        crops.append(mel.numpy().ravel())
        if len(crops) >= n_samples:
            break

    if not crops:
        print("ERROR: no crops loaded", file=sys.stderr)
        return 1

    print(f"method={method_name}  crops={len(crops)}  d={len(crops[0])}", flush=True)

    mean_psnr: list[float] = []
    std_psnr: list[float] = []

    print(f"\n{'ratio':>6}  {'m':>6}  {'mean_psnr':>10}  {'std_psnr':>9}")
    print("-" * 40)

    d = len(crops[0])
    for r in ratios:
        m = max(1, int(round(d * r / 100.0)))
        vals = [psnr(x, backproject(x, m)) for x in crops]
        vals = [v for v in vals if math.isfinite(v)]
        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        mean_psnr.append(mu)
        std_psnr.append(sigma)
        print(f"{r:>5}%  {m:>6}  {mu:>10.2f}  {sigma:>9.2f}")

    out_path = args.output.parent / f"{args.output.stem}_{method_name}{args.output.suffix}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mu_arr = np.array(mean_psnr)
    sd_arr = np.array(std_psnr)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ratios, mu_arr, color="steelblue", linewidth=2, marker="o", markersize=4)
    ax.fill_between(ratios, mu_arr - sd_arr, mu_arr + sd_arr, alpha=0.2, color="steelblue")
    for tr in train_ratios:
        ax.axvline(tr, color="firebrick", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("Measurement rate m/N (%)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(
        Rf"{method_name.upper()} Backprojection $pSNR$ versus Measurement Ratio $m/N$" 
    )
    ax.set_xticks([r for r in ratios if r % 10 == 0 or r == 1])
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nplot saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
