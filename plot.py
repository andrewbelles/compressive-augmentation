#!/usr/bin/env python3
#
# plot.py  Andrew Belles  June 2026
#
# Generates the paper figures from analysis CSVs produced by analyze.py.
#
#   Figure 2: Test F1-macro vs measurement ratio (+ baselines)
#   Figure 3: Nuisance perturbation magnitude vs Test F1-macro and Between-view alignment
#   Figure 4: Alignment and Uniformity vs Test F1-macro
#   Figure 5: Mel-spectrogram views: original + W3 / DCT-U / DCT-B / SRHT at r=20
#
# Usage:
#   python plot.py --analysis-dir analysis/ --output-dir images/
#

import argparse
import math
import os
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/spiky-matplotlib")

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.fft import dct, idct
from scipy.signal import resample as scipy_resample


FIG_BG = "#FFFFFF"
AX_BG  = "#F7F7F7"
GRID_C = "#DDDDDD"
TEXT   = "#222222"

mpl.rcParams.update({
    "axes.spines.top":    False, "axes.spines.right":  False,
    "axes.spines.left":   True,  "axes.spines.bottom": True,
    "axes.edgecolor":     "#AAAAAA", "axes.facecolor":  AX_BG,
    "figure.facecolor":   FIG_BG,   "axes.grid":       True,
    "grid.color":         GRID_C,   "grid.linewidth":  0.6,
    "axes.labelsize":     9,         "xtick.labelsize": 8,   "ytick.labelsize": 8,
    "axes.labelcolor":    TEXT,      "xtick.color":     TEXT, "ytick.color":     TEXT,
    "text.color":         TEXT,
    "legend.fontsize":    8,  "legend.framealpha": 0.9,
    "legend.facecolor":   "#FFFFFF", "legend.edgecolor": "#CCCCCC",
    "figure.dpi":         120, "axes.titlesize": 9, "axes.titlepad": 4,
    "axes.labelpad":      4,   "xtick.major.pad": 3, "ytick.major.pad": 3,
})

CS_STYLES = {
    "dct_biased":  {"color": "#2166AC", "marker": "o", "ls": "-",  "label": "DCT-B"},
    "dct_uniform": {"color": "#D6604D", "marker": "s", "ls": "--", "label": "DCT-U"},
    "srht":        {"color": "#4DAC26", "marker": "^", "ls": "--", "label": "SRHT"},
}
TRAD_STYLES = {
    "w2":     {"color": "#E69F00", "ls": "--", "lw": 1.6, "label": "W2"},
    "w3":     {"color": "#CC79A7", "ls": "--", "lw": 1.6, "label": "W3"},
    "w4":     {"color": "#009E73", "ls": "--", "lw": 1.6, "label": "W4"},
    "supcon": {"color": "#000000", "ls": "-.", "lw": 2.0, "label": "SupCon-W3"},
    "other":  {"color": "#56B4E9", "ls": ":",  "lw": 1.6, "label": "Mel PCA-256"},
}


def _theme(ax: plt.Axes) -> None:
    """
    Apply the shared figure, axes, grid, label, and spine styling.

    Assumptions:
    - ax belongs to a Matplotlib figure that should use the paper theme.
    """
    ax.figure.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    ax.grid(True, color=GRID_C, linewidth=0.6, alpha=0.65)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color("#AAAAAA")


def _style_legend(ax: plt.Axes) -> None:
    """
    Apply shared legend styling if an axes already has a legend.

    Assumptions:
    - The legend object may be absent and should be left unchanged.
    """
    leg = ax.get_legend()
    if leg is None:
        return
    leg.get_frame().set_facecolor("#FFFFFF")
    leg.get_frame().set_edgecolor("#CCCCCC")
    for t in leg.get_texts():
        t.set_color(TEXT)


def plot_f1_vs_ratio(linear: pd.DataFrame, out: Path) -> None:
    """
    Plot test macro-F1 across compression ratios with baseline reference lines.

    Assumptions:
    - linear contains one row per method base with seed-aggregated confidence intervals.
    """
    cs        = linear[linear["family"].isin(CS_STYLES)].copy()
    baselines = linear[~linear["family"].isin(CS_STYLES)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(FIG_BG)
    _theme(ax)

    for _, row in baselines.iterrows():
        st = TRAD_STYLES.get(row["family"])
        if st is None:
            continue
        f  = row["test_f1_mean"]
        lo = row["test_ci_lo"]
        hi = row["test_ci_hi"]
        ax.axhline(f, color=st["color"], linestyle=st["ls"], linewidth=st["lw"],
                   alpha=0.85, label=st["label"])
        ax.axhspan(lo, hi, alpha=0.08, color=st["color"])

    for fam, st in CS_STYLES.items():
        sub = cs[cs["family"] == fam].sort_values("ratio")
        if sub.empty:
            continue
        ratios = sub["ratio"].tolist()
        f1s    = sub["test_f1_mean"].tolist()
        lo     = sub["test_ci_lo"].tolist()
        hi     = sub["test_ci_hi"].tolist()
        yerr   = np.array([(f - l, h - f) for f, l, h in zip(f1s, lo, hi)]).T
        ax.errorbar(ratios, f1s, yerr=yerr,
                    color=st["color"], marker=st["marker"], linestyle=st["ls"],
                    linewidth=2.0, markersize=5, capsize=3, elinewidth=1.2, label=st["label"])

    ax.set_xlabel("Compression ratio (%)")
    ax.set_ylabel("Macro F1 (test)")
    ax.legend(framealpha=0.92, fontsize=8)
    _style_legend(ax)
    fig.tight_layout(pad=0.5)
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_nuisance_perturbation(pert: pd.DataFrame, linear: pd.DataFrame,
                               align: pd.DataFrame, out: Path) -> None:
    """
    Plot nuisance magnitude against downstream F1 and between-view alignment.

    Assumptions:
    - Input CSVs share the same method base identifiers.
    """
    merged = pert.merge(linear[["method", "test_f1_mean"]], on="method", how="inner")
    merged = merged.merge(align[["method", "between_views_mean"]], on="method", how="left")
    cs     = merged[merged["family"].isin(CS_STYLES)].copy()

    panels = [
        ("test_f1_mean",       "Test F1 (macro)"),
        ("between_views_mean", "Between-view alignment"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor(FIG_BG)

    for ax, (ycol, ylabel) in zip(axes, panels):
        _theme(ax)
        ax.set_xscale("log")
        for fam, st in CS_STYLES.items():
            sub = cs[cs["family"] == fam].sort_values("ratio")
            if sub.empty:
                continue
            ax.scatter(sub["nuis_norm"], sub[ycol],
                       color=st["color"], marker=st["marker"],
                       s=40, alpha=0.9, label=st["label"], zorder=3)
            for _, row in sub.iterrows():
                ratio_val = row.get("ratio")
                lbl = f"r{int(ratio_val)}" if pd.notna(ratio_val) else ""
                ax.annotate(lbl, (row["nuis_norm"], row[ycol]),
                            fontsize=6, color=st["color"], alpha=0.8,
                            xytext=(4, 2), textcoords="offset points")
        ax.set_xlabel("Nuisance perturbation magnitude (log scale)")
        ax.set_ylabel(ylabel)
        ax.legend(framealpha=0.92, fontsize=8)
        _style_legend(ax)

    fig.tight_layout(pad=0.5)
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_alignment_vs_f1(align: pd.DataFrame, linear: pd.DataFrame, out: Path) -> None:
    """
    Plot alignment and uniformity metrics against test macro-F1.

    Assumptions:
    - align and linear are outputs from the same analysis run.
    """
    merged  = align.merge(linear[["method", "test_f1_mean"]], on="method", how="inner")
    metrics = [
        ("between_views_mean", "between_views_std", "Between-view alignment"),
        ("uniformity_mean",    "uniformity_std",    "Uniformity"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor(FIG_BG)

    for ax, (col, std_col, xlabel) in zip(axes, metrics):
        _theme(ax)

        for fam, st in CS_STYLES.items():
            sub = merged[merged["family"] == fam].sort_values("ratio")
            if sub.empty:
                continue
            ax.errorbar(sub[col], sub["test_f1_mean"], xerr=sub[std_col],
                        color=st["color"], marker=st["marker"], linestyle="none",
                        markersize=6, capsize=3, elinewidth=1.0,
                        alpha=0.9, label=st["label"], zorder=3)
            for _, row in sub.iterrows():
                ratio_val = row.get("ratio")
                lbl = f"r{int(ratio_val)}" if pd.notna(ratio_val) else ""
                ax.annotate(lbl, (row[col], row["test_f1_mean"]),
                            fontsize=6, color=st["color"], alpha=0.85,
                            xytext=(4, 2), textcoords="offset points")

        for fam, st in TRAD_STYLES.items():
            sub = merged[merged["family"] == fam]
            if sub.empty:
                continue
            ax.errorbar(sub[col], sub["test_f1_mean"], xerr=sub[std_col],
                        color=st["color"], marker="D", linestyle="none",
                        markersize=6, capsize=3, elinewidth=1.0,
                        alpha=0.9, label=st["label"], zorder=3)
            for _, row in sub.iterrows():
                ax.annotate(str(row.get("label", "")), (row[col], row["test_f1_mean"]),
                            fontsize=6, color=st["color"], alpha=0.85,
                            xytext=(4, 2), textcoords="offset points")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Test F1 (macro)")
        ax.legend(framealpha=0.92, fontsize=7.5)
        _style_legend(ax)

    fig.tight_layout(pad=0.5)
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  saved {out.name}")


_MEL_SR      = 22_050
_MEL_N_FFT   = 1_024
_MEL_HOP     = 256
_MEL_N_MELS  = 128
_MEL_F_MIN   = 80.0
_MEL_EPS     = 1e-12
_MEL_SEG_SEC = 30.0
_MEL_OFFSET  = 0.0
_MEL_RATIO   = 20.0
_MEL_WAVE_CFG = {
    "wave_stretch_scale": [0.7, 1.3],
    "wave_gain_strength": 0.4,
    "wave_n_masks":       6,
    "wave_mask_width":    22050,
    "wave_noise_std":     0.02,
}


def _mel_transform() -> "torchaudio.transforms.MelSpectrogram":
    """
    Build the mel-spectrogram transform matching the AudioSTFTEncoder config.

    Assumptions:
    - Parameters must stay in sync with the training encoder.
    """
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=_MEL_SR, n_fft=_MEL_N_FFT, win_length=_MEL_N_FFT,
        hop_length=_MEL_HOP, f_min=_MEL_F_MIN, n_mels=_MEL_N_MELS,
        power=2.0, norm="slaney", mel_scale="htk", center=True,
    )


def _load_segment(audio_path: Path) -> np.ndarray:
    """
    Decode a fixed-length mono segment from an audio file via ffmpeg.

    Assumptions:
    - ffmpeg is on PATH and the file is decodable to float32 PCM.
    """
    cmd = ["ffmpeg", "-v", "error", "-i", str(audio_path),
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(_MEL_SR),
           "-ss", str(_MEL_OFFSET), "-t", str(_MEL_SEG_SEC), "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    y   = np.frombuffer(raw, dtype=np.float32).copy()
    n   = int(_MEL_SR * _MEL_SEG_SEC)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y[:n]


def _to_mel(y: np.ndarray, tf: "torchaudio.transforms.MelSpectrogram") -> np.ndarray:
    """
    Convert a waveform to a log-normalized mel spectrogram.

    Assumptions:
    - tf was built with _mel_transform() and shares the repo mel config.
    """
    mel = torch.log1p(tf(torch.from_numpy(y).unsqueeze(0)).squeeze(0))
    return ((mel - mel.mean()) / mel.std().clamp_min(_MEL_EPS)).numpy()


def _aug_w3(y: np.ndarray, seed: int = 31) -> np.ndarray:
    """
    Apply the W3 waveform augmentation policy (stretch + gain + mask + noise).

    Assumptions:
    - Parameters are drawn from _MEL_WAVE_CFG to match training exactly.
    """
    rng  = np.random.default_rng(seed)
    lo, hi = _MEL_WAVE_CFG["wave_stretch_scale"]
    n    = len(y)
    n_res = max(1, int(round(n * float(rng.uniform(lo, hi)))))
    y2   = scipy_resample(y.astype(np.float64), n_res).astype(np.float32)
    if n_res >= n:
        s  = int(rng.integers(0, n_res - n + 1))
        y2 = y2[s : s + n]
    else:
        pad = n - n_res
        pl  = int(rng.integers(0, pad + 1))
        tmp = np.zeros(n, dtype=np.float32)
        tmp[pl : pl + n_res] = y2
        y2  = tmp
    y2 = y2 * float(rng.uniform(1 - _MEL_WAVE_CFG["wave_gain_strength"],
                                 1 + _MEL_WAVE_CFG["wave_gain_strength"]))
    for _ in range(int(_MEL_WAVE_CFG["wave_n_masks"])):
        w = int(rng.integers(1, _MEL_WAVE_CFG["wave_mask_width"] + 1))
        s = int(rng.integers(0, max(1, n - w)))
        y2[s : s + w] = 0.0
    return (y2 + rng.standard_normal(n).astype(np.float32)
            * _MEL_WAVE_CFG["wave_noise_std"]).astype(np.float32)


def _aug_dct(y: np.ndarray, uniform: bool, seed: int = 1) -> np.ndarray:
    """
    Apply a DCT compressive-sensing reconstruction view at _MEL_RATIO percent.

    Assumptions:
    - Non-uniform sampling uses the same 1/sqrt(k) frequency prior as training.
    """
    rng    = np.random.default_rng(seed)
    n      = len(y)
    m      = max(1, int(round(n * _MEL_RATIO / 100.0)))
    coeffs = dct(y, norm="ortho", workers=1)
    if uniform:
        idx = rng.choice(n, m, replace=False)
    else:
        p   = 1.0 / np.sqrt(np.arange(1, n + 1, dtype=np.float32))
        p  /= p.sum()
        idx = rng.choice(n, m, replace=False, p=p)
    z      = np.zeros(n, dtype=np.float32)
    z[idx] = coeffs[idx] * math.sqrt(n / m)
    return idct(z, norm="ortho", workers=1).astype(np.float32)


def _aug_srht(y: np.ndarray, seed: int = 2) -> np.ndarray:
    """
    Apply an SRHT compressive-sensing reconstruction view at _MEL_RATIO percent.

    Assumptions:
    - Input length is padded internally to the next power of two and stripped on return.
    """
    rng     = np.random.default_rng(seed)
    n       = len(y)
    m       = max(1, int(round(n * _MEL_RATIO / 100.0)))
    p2      = 1 << math.ceil(math.log2(max(n, 2)))
    signs   = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=p2)
    yp      = np.zeros(p2, dtype=np.float32)
    yp[:n]  = y * signs[:n]
    h = 1
    while h < p2:
        yp = yp.reshape(-1, h * 2)
        u, v = yp[:, :h].copy(), yp[:, h:].copy()
        yp[:, :h] = u + v; yp[:, h:] = u - v
        yp = yp.ravel(); h *= 2
    yp /= math.sqrt(p2)
    support   = np.sort(rng.choice(p2, m, replace=False))
    z         = np.zeros(p2, dtype=np.float32)
    z[support] = yp[support] * math.sqrt(p2 / m)
    h = 1
    while h < p2:
        z = z.reshape(-1, h * 2)
        u, v = z[:, :h].copy(), z[:, h:].copy()
        z[:, :h] = u + v; z[:, h:] = u - v
        z = z.ravel(); h *= 2
    z /= math.sqrt(p2)
    return (z[:n] * signs[:n]).astype(np.float32)


def plot_mel_views(audio_path: Path, out: Path) -> None:
    """
    Plot mel-spectrogram views of one track under the original signal and four augmentations.

    Assumptions:
    - audio_path is a decodable audio file; track id is inferred from the stem.
    """
    tf  = _mel_transform()
    y   = _load_segment(audio_path)
    tid = audio_path.stem.lstrip("0") or "0"
    panels = [
        ("Original",  _to_mel(y, tf)),
        ("W3",        _to_mel(_aug_w3(y), tf)),
        ("DCT-U r20", _to_mel(_aug_dct(y, uniform=True), tf)),
        ("DCT-B r20", _to_mel(_aug_dct(y, uniform=False), tf)),
        ("SRHT r20",  _to_mel(_aug_srht(y), tf)),
    ]
    vmin = min(m.min() for _, m in panels)
    vmax = max(m.max() for _, m in panels)

    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    fig.patch.set_facecolor(FIG_BG)
    for ax, (label, mel) in zip(axes, panels):
        ax.imshow(np.flipud(mel), aspect="auto", origin="lower",
                  cmap="magma", vmin=vmin, vmax=vmax,
                  extent=[0, mel.shape[1], 0, mel.shape[0]])
        ax.set_xlabel(label)
        ax.set_xticks([])
        ax.set_yticks([])
        _theme(ax)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_color("#AAAAAA")
    axes[0].set_ylabel(tid)

    fig.tight_layout(pad=0.4)
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  saved {out.name}")


def load_csv(path: Path, name: str) -> pd.DataFrame | None:
    """
    Load an analysis CSV and report whether it is available.

    Assumptions:
    - Missing inputs should be handled by downstream plot skipping.
    """
    if not path.exists():
        print(f"  SKIP {name}: {path} not found")
        return None
    df = pd.read_csv(path)
    print(f"  loaded {name}: {len(df)} rows")
    return df


def main() -> None:
    """
    CLI entry point for generating all paper figures from analysis CSVs.

    Assumptions:
    - Missing CSVs should skip only the figures that require them.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"),
                        help="Directory containing CSVs from analyze.py")
    parser.add_argument("--output-dir",   type=Path, default=Path("data/images"),
                        help="Directory to write output figures")
    parser.add_argument("--audio-path",   type=Path,
                        default=Path("data/fma_small/021/021085.mp3"),
                        help="Audio file for mel-spectrogram augmentation views (Figure 5)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adir = args.analysis_dir
    odir = args.output_dir

    print("Loading CSVs...")
    linear = load_csv(adir / "linear_results.csv",      "linear")
    pert   = load_csv(adir / "perturbation_results.csv", "perturbation")
    align  = load_csv(adir / "alignment_analysis.csv",   "alignment")

    missing = [n for df, n in [(linear, "linear"), (pert, "perturbation"), (align, "alignment")]
               if df is None]

    print("\nGenerating plots...")

    if linear is not None:
        plot_f1_vs_ratio(linear, odir / "f1_vs_ratio.png")
    else:
        print("  SKIP Figure 2: linear_results.csv missing")

    if pert is not None and linear is not None and align is not None:
        plot_nuisance_perturbation(pert, linear, align, odir / "nuisance_perturbation.png")
    else:
        print(f"  SKIP Figure 3: missing {', '.join(missing)}")

    if align is not None and linear is not None:
        plot_alignment_vs_f1(align, linear, odir / "alignment_vs_f1.png")
    else:
        print(f"  SKIP Figure 4: missing {', '.join(missing)}")

    plot_mel_views(args.audio_path, odir / "mel_views.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
