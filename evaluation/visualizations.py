#!/usr/bin/env python3
#
# visualizations.py  Andrew Belles  April 13th, 2026
#
# Notebook-facing plotting helpers for waveform Barlow Twins evaluation.
#

import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/spiky-matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/spiky-numba")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import torch
import torchaudio.transforms as T
import umap
from sklearn.preprocessing import StandardScaler

from representation.audio import (
    load_manifest,
    _load_waveform,
    _dct_cs_view,
    _srht_cs_view,
    apply_wave_policy,
)

_PURPLE = "#3B0F70"
_AMBER  = "#E05C00"
_CREAM  = "#F5DEB3"
_BG     = "#111111"
_GRID   = "#333333"
_SLATE  = "#AAAAAA"

mpl.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": True, "axes.spines.bottom": True,
    "axes.edgecolor": "#444444", "axes.facecolor": _BG,
    "figure.facecolor": _BG, "axes.grid": True,
    "grid.color": _GRID, "grid.linewidth": 0.5,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.labelcolor": _SLATE, "xtick.color": _SLATE, "ytick.color": _SLATE,
    "text.color": _SLATE, "legend.fontsize": 8, "legend.framealpha": 0.9,
    "legend.facecolor": "#1e1e1e", "legend.edgecolor": "#444444",
    "figure.dpi": 110, "axes.titlesize": 9, "axes.titlepad": 4,
    "axes.labelpad": 4, "xtick.major.pad": 3, "ytick.major.pad": 3,
})

_MEL_TRANSFORM = T.MelSpectrogram(
    sample_rate=22050, n_fft=1024, hop_length=256, n_mels=128, power=2.0,
    f_min=80.0, norm="slaney", mel_scale="htk",
)
_DART_CMAP = LinearSegmentedColormap.from_list(
    "dartmouth", ["#1a0030", _PURPLE, _AMBER, _CREAM], N=256
)

GENRE_ORDER = ["Electronic", "Experimental", "Folk", "Hip-Hop",
               "Instrumental", "International", "Pop", "Rock"]

_SAMPLE_RATE = 22050
_SEGMENT_SEC = 5.0
DEFAULT_UMAP_METHOD_A = "wave_barlow_abt_w3_d256_nopop"
DEFAULT_UMAP_METHOD_B = "wave_barlow_cs_uniform_r10_d256_nopop"

_FIG_BG  = "#191919"
_AX_BG   = "#202020"
_GRID_C  = "#4A4A4A"
_TEXT    = "#D8D8D8"
_RAW_COLOR = "#C8C8C8"
_TRAD_STYLES = {
    "W2": ("#8FE388", ":", "W2 traditional"),
    "W3": ("#FF6B8A", ":", "W3 traditional"),
    "W4": ("#FFD166", ":", "W4 traditional"),
}
_CS_STYLES = {
    "DCT biased":  ("#7E57C2", "o", "-"),
    "DCT uniform": ("#FF8C42", "s", "--"),
    "SRHT":        ("#F4D35E", "^", "--"),
}
_RATIO_METHOD_STYLES = {
    "biased":  (*_CS_STYLES["DCT biased"],  "DCT biased"),
    "uniform": (*_CS_STYLES["DCT uniform"], "DCT uniform"),
    "srht":    (*_CS_STYLES["SRHT"],        "SRHT"),
}


def _genre_order(labels: list[str]) -> list[str]:
    ordered = [g for g in GENRE_ORDER if g in labels]
    return ordered + [g for g in labels if g not in ordered]


def _probe_theme(ax: plt.Axes) -> None:
    ax.figure.patch.set_facecolor(_FIG_BG)
    ax.set_facecolor(_AX_BG)
    ax.grid(True, color=_GRID_C, linewidth=0.6, alpha=0.65)
    ax.tick_params(colors=_TEXT)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    for spine in ax.spines.values():
        spine.set_color("#666666")


def _probe_legend(legend) -> None:
    if legend is None:
        return
    legend.get_frame().set_facecolor("#242424")
    legend.get_frame().set_edgecolor("#666666")
    for text in legend.get_texts():
        text.set_color(_TEXT)


def _to_mel(y: np.ndarray) -> np.ndarray:
    mel = _MEL_TRANSFORM(torch.from_numpy(y).unsqueeze(0)).squeeze(0)
    mel = torch.log1p(mel)
    return ((mel - mel.mean()) / mel.std().clamp_min(1e-6)).numpy().astype(np.float32)


def _umap_project(embeddings: np.ndarray, seed: int, n_neighbors: int, min_dist: float) -> np.ndarray:
    z = StandardScaler().fit_transform(np.asarray(embeddings, dtype=np.float32))
    return umap.UMAP(n_neighbors=max(2, min(n_neighbors, len(z) - 1)),
                     min_dist=min_dist, random_state=seed, n_jobs=1, init="random").fit_transform(z)


def _embedding_cols(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("embedding_") and df[c].notna().all())


def _resolve_method(df: pd.DataFrame, method: str) -> str:
    methods = sorted(df["method"].dropna().astype(str).unique().tolist())
    if method in methods:
        return method
    matches = [m for m in methods if method in m]
    if not matches:
        raise ValueError(f"method not found: {method!r}")
    if len(matches) > 1:
        raise ValueError(f"method {method!r} is ambiguous: {matches[:8]}")
    return matches[0]


def _display_label(method: str, group: pd.DataFrame) -> str:
    m = re.search(r"_w(\d+)_", method)
    if m:
        return f"Trained W{m.group(1)}"
    ratio = None
    if "ratio_percent" in group.columns and group["ratio_percent"].notna().any():
        ratio = int(group["ratio_percent"].dropna().iloc[0])
    else:
        r = re.search(r"_r0?(\d+)_", method)
        if r:
            ratio = int(r.group(1))
    sfx = f" r{ratio}" if ratio is not None else ""
    if "uniform" in method:
        return f"Trained DCT uniform{sfx}"
    if "srht" in method:
        return f"Trained SRHT{sfx}"
    if "wave_barlow_cs" in method:
        return f"Trained DCT biased{sfx}"
    return method


def plot_representation_umap_grid(
    representations: dict[str, np.ndarray],
    labels: np.ndarray | list,
    label_order: list[str] | None = None,
    seed: int = 7,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    point_size: float = 8.0,
    alpha: float = 0.72,
    title: str | None = None,
) -> plt.Figure:
    labels_arr   = np.asarray(labels).astype(str)
    unique_labels = _genre_order(list(dict.fromkeys(labels_arr.tolist()))) if label_order is None else [str(l) for l in label_order]
    unique_labels = [l for l in unique_labels if np.any(labels_arr == l)]

    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
               "#E69F00", "#56B4E9", "#6A3D9A", "#666666"]
    color_map = {l: palette[i % len(palette)] for i, l in enumerate(unique_labels)}

    n_panels  = len(representations)
    fig, axes = plt.subplots(1, n_panels, figsize=(max(4.0, 3.9 * n_panels), 3.8), squeeze=False)
    fig.patch.set_facecolor(_BG)

    handles = []
    for i, (ax, (name, emb)) in enumerate(zip(axes.ravel(), representations.items())):
        coords = _umap_project(emb, seed, n_neighbors, min_dist)
        ax.set_facecolor(_BG)
        ax.grid(True, color=_GRID, linewidth=0.6, alpha=0.65)
        ax.set_axisbelow(True)
        for genre in unique_labels:
            mask = labels_arr == genre
            h = ax.scatter(coords[mask, 0], coords[mask, 1], c=[color_map[genre]],
                           s=point_size, alpha=alpha, label=genre, linewidths=0)
            if i == 0:
                handles.append(h)
        ax.set_title(name, color=_SLATE, fontsize=10, pad=8)
        ax.set_xlabel("UMAP 1", color=_SLATE)
        ax.set_ylabel("UMAP 2" if i == 0 else "", color=_SLATE)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    legend = fig.legend(handles, unique_labels, loc="center right",
                        bbox_to_anchor=(0.995, 0.5), framealpha=0.95, markerscale=2.0)
    legend.get_frame().set_facecolor("#1e1e1e")
    legend.get_frame().set_edgecolor("#444444")
    for t in legend.get_texts():
        t.set_color(_SLATE)

    if title:
        fig.suptitle(title, y=0.995, color=_SLATE, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 0.88, 0.94 if title else 1.0), pad=0.5)
    return fig


def load_mel_methods_umap_data(
    mel_data_dir: str | Path = "preprocess/data/fma_small_mel",
    parquet_path: str | Path = "representation/data/wave_barlow_fma_small.parquet",
    method_a: str = DEFAULT_UMAP_METHOD_A,
    method_b: str = DEFAULT_UMAP_METHOD_B,
    method_a_label: str | None = None,
    method_b_label: str | None = None,
    splits: tuple[str, ...] = ("validation", "test"),
    exclude_genres: list[str] | tuple[str, ...] | None = ("Pop",),
    sample_per_genre: int | None = None,
    seed: int = 7,
) -> tuple[dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    from evaluation.linear import load_mel_embeddings

    mel_data_dir = Path(mel_data_dir)
    parquet_path = Path(parquet_path)
    mel_df = load_mel_embeddings(mel_data_dir, splits=tuple(splits) if splits else None)
    wb_df  = pd.read_parquet(parquet_path)

    method_a = _resolve_method(wb_df, method_a)
    method_b = _resolve_method(wb_df, method_b)

    split_set = set(splits) if splits else None
    excluded  = set(exclude_genres or [])

    def _filt(frame: pd.DataFrame, method: str | None = None) -> pd.DataFrame:
        sub = frame.copy()
        if method is not None:
            sub = sub[sub["method"].astype(str) == method]
        if split_set:
            sub = sub[sub["split"].astype(str).isin(split_set)]
        if excluded:
            sub = sub[~sub["genre_top"].astype(str).isin(excluded)]
        sub = sub.dropna(subset=["track_id", "genre_top"])
        sub = sub.drop_duplicates("track_id", keep="first")
        return sub.set_index("track_id", drop=False).sort_index()

    mel_sub  = _filt(mel_df, "raw_mel")
    ma_sub   = _filt(wb_df, method_a)
    mb_sub   = _filt(wb_df, method_b)
    common   = mel_sub.index.intersection(ma_sub.index).intersection(mb_sub.index)
    if len(common) == 0:
        raise ValueError("no common tracks across mel, method_a, and method_b")

    meta = mel_sub.loc[common, ["track_id", "genre_top", "split"]].reset_index(drop=True)
    meta["track_id"]  = meta["track_id"].astype(int)
    meta["genre_top"] = meta["genre_top"].astype(str)

    if sample_per_genre is not None:
        rng = np.random.default_rng(seed)
        sel: list[int] = []
        for genre in _genre_order(meta["genre_top"].drop_duplicates().tolist()):
            ids = meta.loc[meta["genre_top"] == genre, "track_id"].to_numpy()
            if len(ids) > sample_per_genre:
                ids = rng.choice(ids, size=sample_per_genre, replace=False)
            sel.extend(int(i) for i in ids)
        meta = meta.loc[meta["track_id"].isin(sel)].sort_values(["genre_top", "track_id"])

    track_ids = meta["track_id"].to_numpy()
    ma_label  = method_a_label or _display_label(method_a, ma_sub)
    mb_label  = method_b_label or _display_label(method_b, mb_sub)
    representations = {
        "Mel baseline": mel_sub.loc[track_ids, _embedding_cols(mel_sub)].to_numpy(dtype=np.float32),
        ma_label:       ma_sub.loc[track_ids, _embedding_cols(ma_sub)].to_numpy(dtype=np.float32),
        mb_label:       mb_sub.loc[track_ids, _embedding_cols(mb_sub)].to_numpy(dtype=np.float32),
    }
    meta["method_a"] = method_a
    meta["method_b"] = method_b
    return representations, meta["genre_top"].to_numpy(dtype=str), meta.reset_index(drop=True)


def plot_mel_methods_umap(
    mel_data_dir: str | Path = "preprocess/data/fma_small_mel",
    parquet_path: str | Path = "representation/data/wave_barlow_fma_small.parquet",
    method_a: str = DEFAULT_UMAP_METHOD_A,
    method_b: str = DEFAULT_UMAP_METHOD_B,
    method_a_label: str | None = None,
    method_b_label: str | None = None,
    splits: tuple[str, ...] = ("validation", "test"),
    exclude_genres: list[str] | tuple[str, ...] | None = ("Pop",),
    sample_per_genre: int | None = None,
    seed: int = 7,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    title: str | None = "2D UMAP projection by representation",
) -> plt.Figure:
    representations, labels, _ = load_mel_methods_umap_data(
        mel_data_dir=mel_data_dir, parquet_path=parquet_path,
        method_a=method_a, method_b=method_b,
        method_a_label=method_a_label, method_b_label=method_b_label,
        splits=splits, exclude_genres=exclude_genres,
        sample_per_genre=sample_per_genre, seed=seed,
    )
    return plot_representation_umap_grid(representations, labels,
                                         seed=seed, n_neighbors=n_neighbors,
                                         min_dist=min_dist, title=title)


def compute_psnr_alignment_sweep(
    data_dir,
    audio_root,
    ratios_biased: list[int] | None = None,
    ratios_uniform: list[int] | None = None,
    n_samples: int = 256,
    seed: int = 7,
) -> pd.DataFrame:
    data_dir   = Path(data_dir)
    audio_root = Path(audio_root)
    if ratios_biased is None:
        ratios_biased = [10, 30, 40, 60]
    if ratios_uniform is None:
        ratios_uniform = [30, 50, 60, 80]

    manifest = load_manifest(data_dir, "training")
    rng      = np.random.default_rng(seed)
    idx      = rng.choice(len(manifest), min(n_samples, len(manifest)), replace=False)

    samples = []
    for _, row in manifest.iloc[idx].iterrows():
        tid_str    = f"{int(row['track_id']):06d}"
        audio_path = audio_root / "fma_small" / tid_str[:3] / f"{tid_str}.mp3"
        if not audio_path.exists():
            continue
        samples.append(_to_mel(_load_waveform(audio_path, _SAMPLE_RATE, 0.0, _SEGMENT_SEC)).ravel())
        if len(samples) >= n_samples:
            break

    all_configs = (
        [("biased", r) for r in ratios_biased]
        + [("uniform", r) for r in ratios_uniform]
        + [("srht", r) for r in ratios_biased]
    )
    records = []
    for method_tag, ratio in all_configs:
        psnrs, aligns = [], []
        rng1 = np.random.default_rng(seed + ratio)
        rng2 = np.random.default_rng(seed + ratio + 1)
        for flat in samples:
            if method_tag == "biased":
                v1 = _dct_cs_view(flat, float(ratio), rng1, uniform=False, energy_rescale=False).numpy()
                v2 = _dct_cs_view(flat, float(ratio), rng2, uniform=False, energy_rescale=False).numpy()
            elif method_tag == "uniform":
                v1 = _dct_cs_view(flat, float(ratio), rng1, uniform=True, energy_rescale=False).numpy()
                v2 = _dct_cs_view(flat, float(ratio), rng2, uniform=True, energy_rescale=False).numpy()
            else:
                v1 = _srht_cs_view(flat, float(ratio), rng1, energy_rescale=False).numpy()
                v2 = _srht_cs_view(flat, float(ratio), rng2, energy_rescale=False).numpy()
            peak = np.abs(flat).max()
            if peak < 1e-8:
                continue
            mse  = float(np.mean((flat - v1) ** 2))
            psnr = 10.0 * math.log10(float(peak ** 2) / (mse + 1e-12)) if mse > 0 else 100.0
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            cos_sim = float(np.dot(v1, v2) / (n1 * n2)) if n1 > 1e-8 and n2 > 1e-8 else 0.0
            psnrs.append(psnr)
            aligns.append(cos_sim)
        records.append({"method": method_tag, "ratio_percent": ratio,
                        "psnr_db": float(np.mean(psnrs)), "alignment": float(np.mean(aligns))})
    return pd.DataFrame.from_records(records)


def plot_f1_ci_vs_ratio(probe_cache: dict[str, dict], title: str | None = None) -> plt.Figure:
    _TRAD_LABELS = {"W2", "W3", "W4"}
    cs_groups: dict[str, list] = {k: [] for k in _CS_STYLES}
    trad: list = []
    raw_mel = None

    for label, result in probe_cache.items():
        if label == "Raw Mel":
            raw_mel = result
        elif label in _TRAD_LABELS:
            trad.append((label, result))
        else:
            for prefix in _CS_STYLES:
                if label.startswith(prefix + " r"):
                    cs_groups[prefix].append((int(label.split(" r")[-1]), result))

    best_trad = max(trad, key=lambda r: r[1]["test_f1"]) if trad else None

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    fig.patch.set_facecolor(_FIG_BG)

    for ax, (prefix, (color, marker, ls)) in zip(axes, _CS_STYLES.items()):
        _probe_theme(ax)
        if raw_mel is not None:
            f, ci = raw_mel["test_f1"], raw_mel.get("test_f1_ci", 0.0)
            ax.axhline(f, color=_RAW_COLOR, linestyle=":", linewidth=1.35, label="Raw Mel")
            ax.axhspan(f - ci, f + ci, alpha=0.10, color=_RAW_COLOR)
        if best_trad is not None:
            tl, tr = best_trad
            tc, tls, tn = _TRAD_STYLES.get(tl, ("#FF6B8A", ":", tl))
            f, ci = tr["test_f1"], tr.get("test_f1_ci", 0.0)
            ax.axhline(f, color=tc, linestyle=tls, linewidth=1.45, label=tn)
            ax.axhspan(f - ci, f + ci, alpha=0.10, color=tc)
        pts = sorted(cs_groups[prefix])
        if pts:
            ratios = [r for r, _ in pts]
            f1s    = [res["test_f1"] for _, res in pts]
            cis    = [res.get("test_f1_ci", 0.0) for _, res in pts]
            ax.errorbar(ratios, f1s, yerr=cis, color=color, marker=marker, linestyle=ls,
                        linewidth=2.0, markersize=5, capsize=3, elinewidth=1.2, label=prefix)
        ax.set_title(prefix)
        ax.set_xlabel("Compression ratio m/d (%)")
        _probe_legend(ax.legend(framealpha=0.92))

    axes[0].set_ylabel("Macro F1 (test)")
    if title:
        fig.suptitle(title, y=1.01, color=_TEXT)
    fig.tight_layout(pad=0.4)
    return fig


def plot_psnr_alignment_suite(results: pd.DataFrame, title: str | None = None,
                               w3_alignment: float | None = 0.9248) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    fig.patch.set_facecolor(_FIG_BG)
    _probe_theme(ax1)
    _probe_theme(ax2)

    for method in ["biased", "uniform", "srht"]:
        sub = results[results["method"] == method].sort_values("ratio_percent")
        if sub.empty:
            continue
        color, marker, ls, label = _RATIO_METHOD_STYLES[method]
        kw = dict(marker=marker, linestyle=ls, color=color, linewidth=2.0, markersize=5, label=label)
        ax1.plot(sub["ratio_percent"], sub["psnr_db"], **kw)
        ax2.plot(sub["ratio_percent"], sub["alignment"], **kw)

    ax1.set_xlabel("Compression ratio m/d (%)")
    ax1.set_ylabel("pSNR (dB)")
    ax1.set_xlim(left=0)
    _probe_legend(ax1.legend(framealpha=0.92))

    if w3_alignment is not None:
        tc, tls, _ = _TRAD_STYLES["W3"]
        ax2.axhline(w3_alignment, color=tc, linestyle=tls, linewidth=1.45,
                    label=f"W3 within-track ({w3_alignment:.3f})")
    ax2.set_xlabel("Compression ratio m/d (%)")
    ax2.set_ylabel("Cosine alignment")
    ax2.set_xlim(left=0)
    ax2.set_ylim(0, 1)
    _probe_legend(ax2.legend(framealpha=0.92))

    if title:
        fig.suptitle(title, y=1.01, color=_TEXT)
    fig.tight_layout(pad=0.4)
    return fig


def find_centroid_tracks(data_dir, audio_root, n_per_genre: int = 1,
                          split: str = "training", n_samples_per_genre: int = 200,
                          seed: int = 7) -> dict[str, list[int]]:
    data_dir   = Path(data_dir)
    audio_root = Path(audio_root)
    manifest   = load_manifest(data_dir, split)
    rng        = np.random.default_rng(seed)
    result: dict[str, list[int]] = {}

    for genre in manifest["genre_top"].dropna().unique():
        sub = manifest[manifest["genre_top"] == genre].sample(
            min(n_samples_per_genre, len(manifest[manifest["genre_top"] == genre])),
            random_state=int(rng.integers(1 << 31)),
        )
        mels, track_ids = [], []
        for _, row in sub.iterrows():
            tid_str    = f"{int(row['track_id']):06d}"
            audio_path = audio_root / "fma_small" / tid_str[:3] / f"{tid_str}.mp3"
            if not audio_path.exists():
                continue
            mels.append(_to_mel(_load_waveform(audio_path, _SAMPLE_RATE, 0.0, _SEGMENT_SEC)).ravel())
            track_ids.append(int(row["track_id"]))
        if not mels:
            continue
        mels_arr = np.stack(mels)
        dists    = np.linalg.norm(mels_arr - mels_arr.mean(axis=0, keepdims=True), axis=1)
        result[str(genre)] = [track_ids[i] for i in np.argsort(dists)[:n_per_genre]]

    return result


def build_showcase_data(centroid_tracks: dict[str, list[int]], audio_root,
                         ratio: int = 40, seed: int = 7) -> dict[str, dict]:
    audio_root = Path(audio_root)
    wave_cfg   = {"wave_stretch_scale": [0.8, 1.2], "wave_gain_strength": 0.25,
                  "wave_n_masks": 2, "wave_mask_width": 4410, "wave_noise_std": 0.005}
    showcase: dict[str, dict] = {}

    for genre, track_ids in centroid_tracks.items():
        tid_str    = f"{track_ids[0]:06d}"
        audio_path = audio_root / "fma_small" / tid_str[:3] / f"{tid_str}.mp3"
        if not audio_path.exists():
            continue
        y    = _load_waveform(audio_path, _SAMPLE_RATE, 0.0, _SEGMENT_SEC)
        mel  = _to_mel(y)
        flat = mel.ravel()
        y_w2 = apply_wave_policy(y, "w2", wave_cfg, np.random.default_rng(seed + 3))
        showcase[genre] = {
            "track_id": track_ids[0], "original": mel.copy(),
            "w2":      _to_mel(y_w2),
            "biased":  _dct_cs_view(flat, float(ratio), np.random.default_rng(seed),     uniform=False).numpy().reshape(mel.shape),
            "uniform": _dct_cs_view(flat, float(ratio), np.random.default_rng(seed + 1), uniform=True ).numpy().reshape(mel.shape),
            "srht":    _srht_cs_view(flat, float(ratio), np.random.default_rng(seed + 2)              ).numpy().reshape(mel.shape),
        }
    return showcase


def plot_augmentation_showcase(showcase: dict[str, dict], genres: list[str] | None = None,
                                figsize: tuple | None = None) -> plt.Figure:
    aug_keys   = ["original", "w2", "biased", "uniform", "srht"]
    aug_labels = ["Original", "W2", "DCT (biased)", "DCT (uniform)", "SRHT"]

    if genres is None:
        genres = _genre_order(list(showcase.keys()))
    genres = [g for g in genres if g in showcase]
    if not genres:
        raise ValueError("showcase is empty")

    n_genres = len(genres)
    n_augs = len(aug_keys)
    if figsize is None:
        figsize = (n_augs * 2.4, n_genres * 1.8)

    fig, axes = plt.subplots(n_genres, n_augs, figsize=figsize, squeeze=False)
    fig.subplots_adjust(wspace=0.03, hspace=0.06, left=0.12, right=0.99, top=0.95, bottom=0.02)

    for row_i, genre in enumerate(genres):
        entry = showcase[genre]
        for col_i, (key, label) in enumerate(zip(aug_keys, aug_labels)):
            ax  = axes[row_i][col_i]
            mel = entry.get(key)
            if mel is None:
                ax.axis("off")
                continue
            ax.imshow(mel, aspect="auto", origin="lower", interpolation="nearest", cmap="magma")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_i == 0:
                ax.set_title(label, fontsize=8, pad=3)
            if col_i == 0:
                ax.set_ylabel(genre, fontsize=8, rotation=0, labelpad=38, va="center")

    return fig
