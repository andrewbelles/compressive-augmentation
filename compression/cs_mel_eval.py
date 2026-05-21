#!/usr/bin/env python3
#
# cs_mel_eval.py  Andrew Belles  May 13th, 2026
#
# Evaluation suite for CS mel-spectrogram sensing methods.
# CoSaMP reconstruction runs here (not in cs_mel.py) for diagnostic purposes only.
#
# Experiments are controlled by the eval_experiments config key:
#   psnr          -- reconstruction quality vs ratio (with oracle upper bounds)
#   linear_probe  -- logistic regression genre accuracy from raw measurement codes
#   patch_predict -- masked patch R² (MAE objective proxy, on reconstructions)
#   intrinsic_dim -- PCA fraction of dims for 90% variance (compressibility)
#
# All plots saved to compression/images/.
#

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fma-cs-mel-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.fft import dct, idct
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch

from compression.cs_mel import (
    DEFAULT_CONFIG,
    _build_cqt_kernels,
    fit_patch_dictionary,
    fwht_torch,
    load_cqt_tensor,
    load_manifest,
    load_mel_tensor,
    ratio_to_m,
    report,
    sensing_gaussian_random_batch,
    sensing_sfrt_batch,
    sensing_shrt_batch,
    sensing_shrt_cqt_batch,
    sensing_dct_random_batch,
    sensing_patch_dictionary,
    _DEVICE,
)
from compression.train_utils import load_config


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "cs_mel.yaml"

ALL_EXPERIMENTS = ["psnr", "linear_probe", "patch_predict", "intrinsic_dim"]

METHOD_LABELS = {
    "gaussian_random": "Gaussian Random",
    "sfrt": "SFRT",
    "shrt": "SHRT",
    "dct_random": "Partial DCT (Random)",
    "patch_dictionary": "Patch Dictionary",
    "shrt_cqt": "SHRT (CQT domain)",
}

MEL_METHODS = {"gaussian_random", "sfrt", "shrt", "dct_random", "patch_dictionary"}
CQT_METHODS = {"shrt_cqt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CS mel sensing methods.")
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def sample_training_vectors(
    mel_dir: Path,
    mel_bins: int,
    mel_frames: int,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict]]:
    rows = load_manifest(mel_dir, "training")
    rng.shuffle(rows)
    vectors: list[np.ndarray] = []
    meta: list[dict] = []
    for row in rows:
        if len(vectors) >= n_samples:
            break
        x = load_mel_tensor(mel_dir, row, mel_bins, mel_frames)
        if x is not None:
            vectors.append(x)
            meta.append(row)
    if not vectors:
        raise RuntimeError("no training mel tensors found")
    return np.stack(vectors, axis=0), meta


def oracle_psnr(X: np.ndarray, k: int, domain: str = "dct") -> float:
    psnr_vals: list[float] = []
    for x in X:
        if domain == "dct":
            coeffs = dct(x.astype(np.float64), norm="ortho")
            idx = np.argpartition(np.abs(coeffs), -k)[-k:]
            z = np.zeros_like(coeffs)
            z[idx] = coeffs[idx]
            x_hat = idct(z, norm="ortho").astype(np.float32)
        else:
            idx = np.argpartition(np.abs(x), -k)[-k:]
            x_hat = np.zeros_like(x)
            x_hat[idx] = x[idx]
        sp = float(np.mean(x ** 2))
        np_ = float(np.mean((x - x_hat) ** 2))
        psnr_vals.append(100.0 if np_ == 0.0 else 10.0 * np.log10(sp / np_))
    return float(np.mean(psnr_vals))


def cosamp_explicit(
    Y: torch.Tensor,
    Phi: torch.Tensor,
    k: int,
    n_iter: int,
) -> torch.Tensor:
    N, d = Y.shape[0], Phi.shape[1]
    X_hat = torch.zeros(N, d, device=Y.device, dtype=Phi.dtype)
    for _ in range(n_iter):
        r = Y - X_hat @ Phi.T
        proxy = r @ Phi
        _, idx_2k = proxy.abs().topk(min(2 * k, d), dim=1)
        _, idx_cur = X_hat.abs().topk(min(k, d), dim=1)
        S = torch.zeros(N, d, device=Y.device, dtype=torch.bool)
        S.scatter_(1, idx_2k, True)
        S.scatter_(1, idx_cur, True)
        X_ls = torch.zeros(N, d, device=Y.device, dtype=Phi.dtype)
        for n in range(N):
            s = S[n].nonzero(as_tuple=True)[0]
            Phi_s = Phi[:, s]
            c = torch.linalg.lstsq(Phi_s, Y[n]).solution
            X_ls[n, s] = c
        _, topk_idx = X_ls.abs().topk(min(k, d), dim=1)
        X_hat = torch.zeros_like(X_hat)
        X_hat.scatter_(1, topk_idx, X_ls.gather(1, topk_idx))
    return X_hat


def reconstruct(
    method: str,
    X_np: np.ndarray,
    m: int,
    k: int,
    n_iter: int,
    rng: np.random.Generator,
    dictionary: np.ndarray | None = None,
) -> np.ndarray:
    N, d = X_np.shape

    if method == "gaussian_random":
        Phi_np = rng.standard_normal((m, d)).astype(np.float32) / math.sqrt(m)
        Phi = torch.from_numpy(Phi_np).to(_DEVICE)
        X_gpu = torch.from_numpy(X_np).to(_DEVICE)
        Y = X_gpu @ Phi.T
        return cosamp_explicit(Y, Phi, k, n_iter).cpu().numpy()

    elif method == "sfrt":
        n_rfft = d // 2 + 1
        m_c = min(m, n_rfft)
        signs_np = (rng.integers(0, 2, size=d).astype(np.float32) * 2 - 1)
        row_idx_np = np.sort(rng.choice(n_rfft, size=m_c, replace=False))
        signs = torch.from_numpy(signs_np).to(_DEVICE)
        row_idx = torch.from_numpy(row_idx_np).to(_DEVICE)
        X_gpu = torch.from_numpy(X_np).to(_DEVICE)

        def fwd(x: torch.Tensor) -> torch.Tensor:
            return torch.fft.rfft(x * signs.unsqueeze(0), dim=-1)[:, row_idx] / math.sqrt(d)

        def adj(y: torch.Tensor) -> torch.Tensor:
            z = torch.zeros(y.shape[0], n_rfft, device=y.device, dtype=torch.complex64)
            z[:, row_idx] = y
            return torch.fft.irfft(z, n=d, dim=-1) * math.sqrt(d) * signs.unsqueeze(0)

        Y = fwd(X_gpu)
        X_hat = torch.zeros(N, d, device=_DEVICE, dtype=torch.float32)
        for _ in range(n_iter):
            r = Y - fwd(X_hat)
            proxy = adj(r)
            _, idx_2k = proxy.abs().topk(min(2 * k, d), dim=1)
            _, idx_cur = X_hat.abs().topk(min(k, d), dim=1)
            S = torch.zeros(N, d, device=_DEVICE, dtype=torch.bool)
            S.scatter_(1, idx_2k, True)
            S.scatter_(1, idx_cur, True)
            X_ls = torch.zeros(N, d, device=_DEVICE, dtype=torch.float32)
            for n in range(N):
                s = S[n].nonzero(as_tuple=True)[0]
                e_s = torch.zeros(len(s), d, device=_DEVICE, dtype=torch.float32)
                e_s.scatter_(1, s.unsqueeze(1), 1.0)
                Phi_s_c = fwd(e_s).T
                Phi_s_real = torch.cat([Phi_s_c.real, Phi_s_c.imag], dim=0)
                y_n_real = torch.cat([Y[n].real, Y[n].imag], dim=0)
                c = torch.linalg.lstsq(Phi_s_real, y_n_real).solution
                X_ls[n, s] = c
            _, topk_idx = X_ls.abs().topk(min(k, d), dim=1)
            X_hat = torch.zeros_like(X_hat)
            X_hat.scatter_(1, topk_idx, X_ls.gather(1, topk_idx))
        if _DEVICE.type == "cuda":
            torch.cuda.synchronize()
        return X_hat.cpu().numpy()

    elif method in ("shrt", "shrt_cqt"):
        d_pad = 1 << math.ceil(math.log2(max(d, 2)))
        signs_np = (rng.integers(0, 2, size=d_pad).astype(np.float32) * 2 - 1)
        row_idx_np = np.sort(rng.choice(d_pad, size=min(m, d_pad), replace=False))
        signs = torch.from_numpy(signs_np).to(_DEVICE)
        row_idx = torch.from_numpy(row_idx_np).to(_DEVICE)
        X_pad = torch.zeros(N, d_pad, device=_DEVICE, dtype=torch.float32)
        X_pad[:, :d] = torch.from_numpy(X_np).to(_DEVICE)

        def fwd(x: torch.Tensor) -> torch.Tensor:
            xp = torch.zeros(*x.shape[:-1], d_pad, device=x.device, dtype=x.dtype)
            xp[..., :x.shape[-1]] = x
            return fwht_torch(xp * signs.unsqueeze(0))[:, row_idx] / math.sqrt(d_pad)

        def adj(y: torch.Tensor) -> torch.Tensor:
            z = torch.zeros(y.shape[0], d_pad, device=y.device, dtype=y.dtype)
            z[:, row_idx] = y
            return (fwht_torch(z) / math.sqrt(d_pad) * signs.unsqueeze(0))[:, :d]

        Y = fwd(X_pad)
        X_hat = torch.zeros(N, d, device=_DEVICE, dtype=torch.float32)
        for _ in range(n_iter):
            r = Y - fwd(X_hat)
            proxy = adj(r)
            _, idx_2k = proxy.abs().topk(min(2 * k, d), dim=1)
            _, idx_cur = X_hat.abs().topk(min(k, d), dim=1)
            S = torch.zeros(N, d, device=_DEVICE, dtype=torch.bool)
            S.scatter_(1, idx_2k, True)
            S.scatter_(1, idx_cur, True)
            X_ls = torch.zeros(N, d, device=_DEVICE, dtype=torch.float32)
            for n in range(N):
                s = S[n].nonzero(as_tuple=True)[0]
                e_s = torch.zeros(len(s), d, device=_DEVICE, dtype=torch.float32)
                e_s.scatter_(1, s.unsqueeze(1), 1.0)
                Phi_s = fwd(e_s).T
                c = torch.linalg.lstsq(Phi_s, Y[n]).solution
                X_ls[n, s] = c
            _, topk_idx = X_ls.abs().topk(min(k, d), dim=1)
            X_hat = torch.zeros_like(X_hat)
            X_hat.scatter_(1, topk_idx, X_ls.gather(1, topk_idx))
        if _DEVICE.type == "cuda":
            torch.cuda.synchronize()
        return X_hat.cpu().numpy()

    elif method == "dct_random":
        row_idx_np = np.sort(rng.choice(d, size=min(m, d), replace=False))

        def fwd_dct(x: torch.Tensor) -> torch.Tensor:
            x_np = x.cpu().numpy().astype(np.float64)
            c = dct(x_np, norm="ortho", axis=-1)
            return torch.from_numpy(c[..., row_idx_np].astype(np.float32)).to(_DEVICE)

        def adj_dct(y: torch.Tensor) -> torch.Tensor:
            y_np = y.cpu().numpy().astype(np.float64)
            z = np.zeros((*y_np.shape[:-1], d), dtype=np.float64)
            z[..., row_idx_np] = y_np
            return torch.from_numpy(idct(z, norm="ortho", axis=-1).astype(np.float32)).to(_DEVICE)

        X_gpu = torch.from_numpy(X_np).to(_DEVICE)
        Y = fwd_dct(X_gpu)
        X_hat = torch.zeros(N, d, device=_DEVICE, dtype=torch.float32)
        for _ in range(n_iter):
            r = Y - fwd_dct(X_hat)
            proxy = adj_dct(r)
            _, idx_2k = proxy.abs().topk(min(2 * k, d), dim=1)
            _, idx_cur = X_hat.abs().topk(min(k, d), dim=1)
            S = torch.zeros(N, d, device=_DEVICE, dtype=torch.bool)
            S.scatter_(1, idx_2k, True)
            S.scatter_(1, idx_cur, True)
            X_ls = torch.zeros(N, d, device=_DEVICE, dtype=torch.float32)
            for n in range(N):
                s = S[n].nonzero(as_tuple=True)[0]
                e_s = torch.zeros(len(s), d, device=_DEVICE, dtype=torch.float32)
                e_s.scatter_(1, s.unsqueeze(1), 1.0)
                Phi_s = fwd_dct(e_s).T
                c = torch.linalg.lstsq(Phi_s, Y[n]).solution
                X_ls[n, s] = c
            _, topk_idx = X_ls.abs().topk(min(k, d), dim=1)
            X_hat = torch.zeros_like(X_hat)
            X_hat.scatter_(1, topk_idx, X_ls.gather(1, topk_idx))
        if _DEVICE.type == "cuda":
            torch.cuda.synchronize()
        return X_hat.cpu().numpy()

    elif method == "patch_dictionary":
        assert dictionary is not None
        nc = dictionary.shape[0]
        k_target = max(1, min(nc, round(m / d * nc)))
        return np.stack([sensing_patch_dictionary(x, dictionary, k_target)[2] for x in X_np])

    raise ValueError(f"unknown method: {method}")


def oracle_reconstruct(
    method: str,
    X_np: np.ndarray,
    k: int,
    dictionary: np.ndarray | None = None,
) -> np.ndarray:
    N, d = X_np.shape
    if method in CQT_METHODS:
        X_hat = np.zeros_like(X_np)
        for i, x in enumerate(X_np):
            idx = np.argpartition(np.abs(x), -min(k, d))[-min(k, d):]
            X_hat[i, idx] = x[idx]
        return X_hat
    elif method == "patch_dictionary":
        assert dictionary is not None
        nc = dictionary.shape[0]
        k_target = max(1, min(nc, k))
        return np.stack([sensing_patch_dictionary(x, dictionary, k_target)[2] for x in X_np])
    else:
        X_hat = np.zeros_like(X_np)
        for i, x in enumerate(X_np):
            coeffs = dct(x.astype(np.float64), norm="ortho")
            idx = np.argpartition(np.abs(coeffs), -min(k, d))[-min(k, d):]
            z = np.zeros_like(coeffs)
            z[idx] = coeffs[idx]
            X_hat[i] = idct(z, norm="ortho").astype(np.float32)
        return X_hat


def plot_psnr(
    methods: list[str],
    ratios: list[int],
    results: dict[str, list[float]],
    stderrs: dict[str, list[float]],
    image_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("tab10", n_colors=len(methods))
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    x_pos = np.array(ratios, dtype=float)
    for i, method in enumerate(methods):
        ax.errorbar(
            x_pos, results[method], yerr=stderrs[method],
            marker="o", linewidth=1.8, markersize=5, capsize=3,
            label=METHOD_LABELS.get(method, method), color=palette[i],
        )
    ax.set_xlabel("m/d (%)", fontsize=12)
    ax.set_ylabel("Mean pSNR (dB)", fontsize=12)
    ax.set_title("CS Mel pSNR vs Compression Ratio", fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in ratios])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Method", frameon=True, fontsize=9)
    out = image_dir / "cs_mel_psnr.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    report(f"saved {out}")


def plot_linear_probe(
    methods: list[str],
    ratios: list[int],
    codes: dict[str, dict[int, np.ndarray]],
    labels: np.ndarray,
    X_mel_baseline: np.ndarray,
    image_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("tab10", n_colors=len(methods))
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    x_pos = np.array(ratios, dtype=float)
    scaler_mel = StandardScaler()
    X_mel_s = scaler_mel.fit_transform(X_mel_baseline)
    clf_mel = LogisticRegression(max_iter=500, C=0.1, solver="lbfgs", multi_class="multinomial")
    baseline_acc = float(np.mean(cross_val_score(clf_mel, X_mel_s, labels, cv=5, scoring="accuracy")))
    ax.axhline(baseline_acc, color="black", linestyle="--", linewidth=1.2, label=f"Raw mel baseline ({baseline_acc:.2f})")
    for i, method in enumerate(methods):
        accs: list[float] = []
        for ratio in ratios:
            Y = codes[method][ratio]
            scaler = StandardScaler()
            Y_s = scaler.fit_transform(Y)
            clf = LogisticRegression(max_iter=500, C=0.1, solver="lbfgs", multi_class="multinomial")
            acc = float(np.mean(cross_val_score(clf, Y_s, labels, cv=5, scoring="accuracy")))
            accs.append(acc)
            log(f"linear_probe method={method} ratio={ratio}% acc={acc:.3f}")
        ax.plot(x_pos, accs, marker="o", linewidth=1.8, markersize=5,
                label=METHOD_LABELS.get(method, method), color=palette[i])
    ax.set_xlabel("m/d (%)", fontsize=12)
    ax.set_ylabel("5-fold CV Genre Accuracy", fontsize=12)
    ax.set_title("Linear Probe Accuracy vs Compression Ratio", fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in ratios])
    ax.set_ylim(0, 1)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Method", frameon=True, fontsize=9)
    out = image_dir / "cs_mel_linear_probe.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    report(f"saved {out}")


def plot_patch_predictability(
    methods: list[str],
    ratios: list[int],
    xhats: dict[str, dict[int, np.ndarray]],
    mel_bins: int,
    mel_frames: int,
    cqt_n_bins: int,
    cqt_n_frames: int,
    image_dir: Path,
    mask_ratio: float = 0.75,
    n_patches: int = 16,
    seed: int = 0,
) -> None:
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("tab10", n_colors=len(methods))
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    x_pos = np.array(ratios, dtype=float)
    rng = np.random.default_rng(seed)
    for i, method in enumerate(methods):
        total_d = cqt_n_bins * cqt_n_frames if method in CQT_METHODS else mel_bins * mel_frames
        patch_dim = total_d // n_patches
        r2_per_ratio: list[float] = []
        for ratio in ratios:
            X_hat = xhats[method][ratio]
            N = X_hat.shape[0]
            X_patched = X_hat[:, : n_patches * patch_dim].reshape(N, n_patches, patch_dim)
            r2_vals: list[float] = []
            for _ in range(10):
                mask = rng.random(n_patches) < mask_ratio
                if mask.all() or not mask.any():
                    continue
                visible = X_patched[:, ~mask, :].reshape(N, -1)
                masked = X_patched[:, mask, :].reshape(N, -1)
                ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
                ridge.fit(visible, masked)
                pred = ridge.predict(visible)
                ss_res = float(np.sum((masked - pred) ** 2))
                ss_tot = float(np.sum((masked - masked.mean(axis=0)) ** 2))
                r2_vals.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
            r2_per_ratio.append(float(np.mean(r2_vals)) if r2_vals else 0.0)
            log(f"patch_predict method={method} ratio={ratio}% R²={r2_per_ratio[-1]:.3f}")
        ax.plot(x_pos, r2_per_ratio, marker="o", linewidth=1.8, markersize=5,
                label=METHOD_LABELS.get(method, method), color=palette[i])
    ax.set_xlabel("m/d (%)", fontsize=12)
    ax.set_ylabel("Masked Patch R²", fontsize=12)
    ax.set_title("Patch Predictability (MAE proxy, 75% mask) vs Compression Ratio", fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in ratios])
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Method", frameon=True, fontsize=9)
    out = image_dir / "cs_mel_patch_predict.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    report(f"saved {out}")


def plot_oracle_psnr(
    methods: list[str],
    ratios: list[int],
    oracle_results: dict[str, list[float]],
    oracle_stderrs: dict[str, list[float]],
    image_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("tab10", n_colors=len(methods))
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    x_pos = np.array(ratios, dtype=float)
    for i, method in enumerate(methods):
        ax.errorbar(
            x_pos, oracle_results[method], yerr=oracle_stderrs[method],
            marker="o", linewidth=1.8, markersize=5, capsize=3,
            label=METHOD_LABELS.get(method, method), color=palette[i],
        )
    ax.set_xlabel("m/d (%)", fontsize=12)
    ax.set_ylabel("Mean pSNR (dB)", fontsize=12)
    ax.set_title("Oracle pSNR vs Compression Ratio (best k-sparse in native domain)", fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in ratios])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Method", frameon=True, fontsize=9)
    out = image_dir / "cs_mel_oracle_psnr.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    report(f"saved {out}")


def plot_oracle_linear_probe(
    methods: list[str],
    ratios: list[int],
    oracle_xhats: dict[str, dict[int, np.ndarray]],
    labels: np.ndarray,
    X_mel_baseline: np.ndarray,
    image_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("tab10", n_colors=len(methods))
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    x_pos = np.array(ratios, dtype=float)
    scaler_mel = StandardScaler()
    X_mel_s = scaler_mel.fit_transform(X_mel_baseline)
    clf_mel = LogisticRegression(max_iter=500, C=0.1, solver="lbfgs", multi_class="multinomial")
    baseline_acc = float(np.mean(cross_val_score(clf_mel, X_mel_s, labels, cv=5, scoring="accuracy")))
    ax.axhline(baseline_acc, color="black", linestyle="--", linewidth=1.2, label=f"Raw mel baseline ({baseline_acc:.2f})")
    for i, method in enumerate(methods):
        accs: list[float] = []
        for ratio in ratios:
            X_hat = oracle_xhats[method][ratio]
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X_hat)
            clf = LogisticRegression(max_iter=500, C=0.1, solver="lbfgs", multi_class="multinomial")
            acc = float(np.mean(cross_val_score(clf, X_s, labels, cv=5, scoring="accuracy")))
            accs.append(acc)
            log(f"oracle_linear_probe method={method} ratio={ratio}% acc={acc:.3f}")
        ax.plot(x_pos, accs, marker="o", linewidth=1.8, markersize=5,
                label=METHOD_LABELS.get(method, method), color=palette[i])
    ax.set_xlabel("m/d (%)", fontsize=12)
    ax.set_ylabel("5-fold CV Genre Accuracy", fontsize=12)
    ax.set_title("Oracle Linear Probe Accuracy vs Compression Ratio (best k-sparse reconstruction)", fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in ratios])
    ax.set_ylim(0, 1)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Method", frameon=True, fontsize=9)
    out = image_dir / "cs_mel_oracle_linear_probe.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    report(f"saved {out}")


def plot_oracle_patch_predictability(
    methods: list[str],
    ratios: list[int],
    oracle_xhats: dict[str, dict[int, np.ndarray]],
    mel_bins: int,
    mel_frames: int,
    cqt_n_bins: int,
    cqt_n_frames: int,
    image_dir: Path,
    mask_ratio: float = 0.75,
    n_patches: int = 16,
    seed: int = 0,
) -> None:
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("tab10", n_colors=len(methods))
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    x_pos = np.array(ratios, dtype=float)
    rng = np.random.default_rng(seed)
    for i, method in enumerate(methods):
        total_d = cqt_n_bins * cqt_n_frames if method in CQT_METHODS else mel_bins * mel_frames
        patch_dim = total_d // n_patches
        r2_per_ratio: list[float] = []
        for ratio in ratios:
            X_hat = oracle_xhats[method][ratio]
            N = X_hat.shape[0]
            X_patched = X_hat[:, : n_patches * patch_dim].reshape(N, n_patches, patch_dim)
            r2_vals: list[float] = []
            for _ in range(10):
                mask = rng.random(n_patches) < mask_ratio
                if mask.all() or not mask.any():
                    continue
                visible = X_patched[:, ~mask, :].reshape(N, -1)
                masked = X_patched[:, mask, :].reshape(N, -1)
                ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
                ridge.fit(visible, masked)
                pred = ridge.predict(visible)
                ss_res = float(np.sum((masked - pred) ** 2))
                ss_tot = float(np.sum((masked - masked.mean(axis=0)) ** 2))
                r2_vals.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
            r2_per_ratio.append(float(np.mean(r2_vals)) if r2_vals else 0.0)
            log(f"oracle_patch_predict method={method} ratio={ratio}% R²={r2_per_ratio[-1]:.3f}")
        ax.plot(x_pos, r2_per_ratio, marker="o", linewidth=1.8, markersize=5,
                label=METHOD_LABELS.get(method, method), color=palette[i])
    ax.set_xlabel("m/d (%)", fontsize=12)
    ax.set_ylabel("Masked Patch R²", fontsize=12)
    ax.set_title("Oracle Patch Predictability (MAE proxy, 75% mask, best k-sparse)", fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in ratios])
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Method", frameon=True, fontsize=9)
    out = image_dir / "cs_mel_oracle_patch_predict.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    report(f"saved {out}")


def plot_intrinsic_dim(
    methods: list[str],
    ratios: list[int],
    codes: dict[str, dict[int, np.ndarray]],
    image_dir: Path,
    var_threshold: float = 0.90,
) -> None:
    sns.set_theme(style="whitegrid")
    data = np.zeros((len(methods), len(ratios)))
    for i, method in enumerate(methods):
        for j, ratio in enumerate(ratios):
            Y = codes[method][ratio]
            n, m = Y.shape
            n_components = min(n - 1, m, 128)
            pca = PCA(n_components=n_components)
            pca.fit(Y)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            k90 = int(np.searchsorted(cumvar, var_threshold) + 1)
            data[i, j] = k90 / m
            log(f"intrinsic_dim method={method} ratio={ratio}% k90={k90} m={m} frac={k90/m:.3f}")
    fig, ax = plt.subplots(figsize=(10, max(3, len(methods) * 0.7 + 1.5)), constrained_layout=True)
    sns.heatmap(
        data, ax=ax,
        xticklabels=[str(r) for r in ratios],
        yticklabels=[METHOD_LABELS.get(m, m) for m in methods],
        annot=True, fmt=".2f", cmap="YlOrRd_r", vmin=0, vmax=1,
        cbar_kws={"label": f"k90 / m  (fraction of dims for {int(var_threshold*100)}% var)"},
    )
    ax.set_xlabel("m/d (%)", fontsize=12)
    ax.set_ylabel("Method", fontsize=12)
    ax.set_title("Intrinsic Dimensionality (lower = more structured)", fontsize=13)
    out = image_dir / "cs_mel_intrinsic_dim.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    report(f"saved {out}")


def run_eval(config: dict) -> None:
    mel_dir = Path(str(config["mel_dir"])).expanduser().resolve()
    mel_bins = int(config["mel_bins"])
    mel_frames = int(config["mel_frames"])
    ratios = [int(r) for r in config["ratios"]]
    methods = [str(m) for m in config["methods"]]
    n_samples = int(config["n_eval_samples"])
    seed = int(config["seed"])
    dict_cfg = config.get("dictionary", {})
    n_components = int(dict_cfg.get("n_components", 128))
    max_iter = int(dict_cfg.get("max_iter", 500))
    cqt_cfg: dict = config.get("cqt", DEFAULT_CONFIG["cqt"])
    cosamp_sparsity = int(config.get("cosamp_sparsity", 128))
    cosamp_iters = int(config.get("cosamp_iters", 20))
    experiments: list[str] = [str(e) for e in config.get("eval_experiments", ALL_EXPERIMENTS)]

    needs_recon = any(e in experiments for e in ("psnr", "patch_predict", "linear_probe"))

    d = mel_bins * mel_frames
    rng = np.random.default_rng(seed)

    report(f"experiments={experiments}")
    report(f"sampling {n_samples} training vectors (d={d})...")
    X, meta = sample_training_vectors(mel_dir, mel_bins, mel_frames, n_samples, rng)
    report(f"loaded n={len(X)} samples")

    genre_raw = [row.get("genre_top", "unknown") for row in meta]
    le = LabelEncoder()
    labels = le.fit_transform(genre_raw)
    report(f"genres: {list(le.classes_)}")

    dictionary: np.ndarray | None = None
    if "patch_dictionary" in methods:
        report("fitting patch dictionary...")
        all_rows = load_manifest(mel_dir, "training")
        train_vecs: list[np.ndarray] = []
        for row in all_rows:
            v = load_mel_tensor(mel_dir, row, mel_bins, mel_frames)
            if v is not None:
                train_vecs.append(v)
        dictionary = fit_patch_dictionary(np.stack(train_vecs), n_components, max_iter, seed)

    X_cqt: np.ndarray | None = None
    cqt_n_bins = int(cqt_cfg.get("n_bins", 84))
    cqt_n_frames = int(cqt_cfg.get("n_frames", 128))
    if any(m in CQT_METHODS for m in methods):
        sr = int(cqt_cfg.get("sample_rate", 22050))
        fmin = float(cqt_cfg.get("fmin", 32.7))
        bpo = int(cqt_cfg.get("bins_per_octave", 12))
        hop = int(cqt_cfg.get("hop_length", 512))
        cqt_kernels, cqt_Nks, cqt_max_Nk = _build_cqt_kernels(sr, fmin, cqt_n_bins, bpo)
        report(f"sampling {n_samples} CQT training vectors...")
        rows_all = load_manifest(mel_dir, "training")
        rng_cqt = np.random.default_rng(seed)
        rng_cqt.shuffle(rows_all)
        cqt_vecs: list[np.ndarray] = []
        for row in rows_all:
            if len(cqt_vecs) >= n_samples:
                break
            v = load_cqt_tensor(mel_dir, row, cqt_kernels, cqt_Nks, cqt_max_Nk, hop, cqt_n_frames, sr, _DEVICE)
            if v is not None:
                cqt_vecs.append(v)
        if not cqt_vecs:
            raise RuntimeError("no CQT tensors found")
        X_cqt = np.stack(cqt_vecs, axis=0)
        report(f"loaded n={len(X_cqt)} CQT samples (d={X_cqt.shape[1]})")

    results: dict[str, list[float]] = {m: [] for m in methods}
    stderrs: dict[str, list[float]] = {m: [] for m in methods}
    oracle_results: dict[str, list[float]] = {m: [] for m in methods}
    oracle_stderrs: dict[str, list[float]] = {m: [] for m in methods}
    codes: dict[str, dict[int, np.ndarray]] = {m: {} for m in methods}
    xhats: dict[str, dict[int, np.ndarray]] = {m: {} for m in methods}
    oracle_xhats: dict[str, dict[int, np.ndarray]] = {m: {} for m in methods}

    for method in methods:
        X_ref = X_cqt if method in CQT_METHODS else X
        assert X_ref is not None
        d_ref = X_ref.shape[1]

        for ratio in ratios:
            m_dim = ratio_to_m(d_ref, ratio)
            k = max(1, min(cosamp_sparsity, round(ratio / 100.0 * cosamp_sparsity)))
            method_rng = np.random.default_rng(seed + ratio)

            if method == "gaussian_random":
                Y_np, _ = sensing_gaussian_random_batch(X_ref, m_dim, method_rng)
            elif method == "sfrt":
                Y_np, _, _ = sensing_sfrt_batch(X_ref, m_dim, method_rng)
            elif method == "shrt":
                Y_np, _, _ = sensing_shrt_batch(X_ref, m_dim, method_rng)
            elif method == "dct_random":
                Y_np, _ = sensing_dct_random_batch(X_ref, m_dim, method_rng)
            elif method == "shrt_cqt":
                Y_np, _, _ = sensing_shrt_cqt_batch(X_ref, m_dim, method_rng)
            elif method == "patch_dictionary":
                assert dictionary is not None
                nc = dictionary.shape[0]
                k_target = max(1, min(nc, round(ratio / 100.0 * nc)))
                results_list = [sensing_patch_dictionary(x, dictionary, k_target) for x in X_ref]
                Y_np = np.stack([r[0] for r in results_list])

            codes[method][ratio] = Y_np.reshape(len(Y_np), -1).astype(np.float32)

            X_hat_oracle = oracle_reconstruct(method, X_ref, k, dictionary)
            oracle_xhats[method][ratio] = X_hat_oracle.astype(np.float32)

            if "psnr" in experiments:
                oracle_psnr_vals: list[float] = []
                for i, x in enumerate(X_ref):
                    sp = float(np.mean(x ** 2))
                    np_ = float(np.mean((x - X_hat_oracle[i]) ** 2))
                    oracle_psnr_vals.append(100.0 if np_ == 0.0 else 10.0 * np.log10(sp / np_))
                oracle_results[method].append(float(np.mean(oracle_psnr_vals)))
                oracle_stderrs[method].append(float(np.std(oracle_psnr_vals) / np.sqrt(len(oracle_psnr_vals))))
                log(f"oracle_psnr method={method} ratio={ratio}% pSNR={oracle_results[method][-1]:.2f} dB")

            if needs_recon:
                recon_rng = np.random.default_rng(seed + ratio)
                X_hat_np = reconstruct(method, X_ref, m_dim, k, cosamp_iters, recon_rng, dictionary)
                xhats[method][ratio] = X_hat_np.astype(np.float32)

                if "psnr" in experiments:
                    psnr_vals: list[float] = []
                    for i, x in enumerate(X_ref):
                        sp = float(np.mean(x ** 2))
                        np_ = float(np.mean((x - X_hat_np[i]) ** 2))
                        psnr_vals.append(100.0 if np_ == 0.0 else 10.0 * np.log10(sp / np_))
                    mean_psnr = float(np.mean(psnr_vals))
                    stderr = float(np.std(psnr_vals) / np.sqrt(len(psnr_vals)))
                    results[method].append(mean_psnr)
                    stderrs[method].append(stderr)
                    log(f"psnr method={method} ratio={ratio}% pSNR={mean_psnr:.2f} dB (+/-{stderr:.2f})")

        report(f"method={method} done")

    image_dir = Path(__file__).resolve().parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    if "psnr" in experiments:
        plot_psnr(methods, ratios, results, stderrs, image_dir)
        plot_oracle_psnr(methods, ratios, oracle_results, oracle_stderrs, image_dir)

    if "linear_probe" in experiments:
        plot_linear_probe(methods, ratios, codes, labels, X, image_dir)
        plot_oracle_linear_probe(methods, ratios, oracle_xhats, labels, X, image_dir)

    if "patch_predict" in experiments:
        plot_patch_predictability(
            methods, ratios, xhats,
            mel_bins, mel_frames, cqt_n_bins, cqt_n_frames,
            image_dir, seed=seed,
        )
        plot_oracle_patch_predictability(
            methods, ratios, oracle_xhats,
            mel_bins, mel_frames, cqt_n_bins, cqt_n_frames,
            image_dir, seed=seed,
        )

    if "intrinsic_dim" in experiments:
        plot_intrinsic_dim(methods, ratios, codes, image_dir)


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_CONFIG)
    report(
        f"START module=compression.cs_mel_eval n_samples={config['n_eval_samples']} "
        f"ratios={config['ratios']}"
    )
    run_eval(config)
    report("DONE module=compression.cs_mel_eval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
