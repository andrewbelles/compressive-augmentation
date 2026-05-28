#!/usr/bin/env python3
#
# visualizations.py  Andrew Belles  April 13th, 2026
#
# Notebook-facing plotting helpers for waveform Barlow Twins evaluation.
#

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/spiky-matplotlib")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from sklearn.preprocessing import LabelEncoder, StandardScaler


GENRE_ORDER = [
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Pop",
    "Rock",
]

METHOD_COLORS = {
    "cs_biased": "#1f77b4",
    "cs_uniform": "#ff7f0e",
    "abt": "#2ca02c",
    "raw_mel": "#9467bd",
}


def _genre_order(labels: list[str]) -> list[str]:
    ordered = [g for g in GENRE_ORDER if g in labels]
    ordered += [g for g in labels if g not in ordered]
    return ordered


def plot_ratio_vs_f1(
    summary: pd.DataFrame,
    ax: plt.Axes | None = None,
    metric: str = "test_f1_macro",
    title: str | None = None,
    baseline_f1: float | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    methods = summary["method"].unique()
    for method in sorted(methods):
        sub = summary[summary["method"] == method].sort_values("ratio_percent")
        color = METHOD_COLORS.get(method)
        ax.plot(sub["ratio_percent"], sub[metric], marker="o", linewidth=2,
                markersize=6, label=method, color=color)

    if baseline_f1 is not None:
        ax.axhline(baseline_f1, color="black", linestyle="--", linewidth=1.2,
                   alpha=0.8, label="ABT baseline (r=100)")

    ax.set_xlabel("Compression ratio m/d (%)")
    ax.set_ylabel("Macro F1 (test)")
    ax.set_xlim(left=0)
    ax.legend()
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title)
    return ax


def plot_per_genre_f1(
    per_genre: dict[str, float],
    ax: plt.Axes | None = None,
    title: str | None = None,
    baseline_per_genre: dict[str, float] | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    genres = _genre_order(list(per_genre.keys()))
    values = [per_genre.get(g, 0.0) for g in genres]
    x = np.arange(len(genres))
    width = 0.35 if baseline_per_genre else 0.6

    ax.bar(x if baseline_per_genre else x, values, width=width,
           color="#1f77b4", label="CS" if baseline_per_genre else None)

    if baseline_per_genre:
        base_values = [baseline_per_genre.get(g, 0.0) for g in genres]
        ax.bar(x + width, base_values, width=width, color="#9467bd", label="ABT baseline")
        ax.legend()

    ax.set_xticks(x + width / 2 if baseline_per_genre else x)
    ax.set_xticklabels(genres, rotation=30, ha="right")
    ax.set_ylabel("F1 (test)")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.25)
    if title:
        ax.set_title(title)
    return ax


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    ax: plt.Axes | None = None,
    title: str | None = None,
    normalize: bool = True,
) -> plt.Axes:
    from sklearn.metrics import confusion_matrix

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ordered = _genre_order(labels)
    le = LabelEncoder().fit(labels)
    cm = confusion_matrix(y_true, y_pred, labels=le.transform(ordered))

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.where(row_sums > 0, cm / row_sums, 0.0)
        fmt, vmax = ".2f", 1.0
    else:
        fmt, vmax = "d", None

    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", vmin=0, vmax=vmax,
                square=True, xticklabels=ordered, yticklabels=ordered,
                cbar_kws={"label": "Row-normalized accuracy" if normalize else "Count"},
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    return ax


def plot_umap(
    embeddings: np.ndarray,
    labels: np.ndarray | list,
    label_names: list[str] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    seed: int = 7,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> plt.Axes:
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required: pip install umap-learn")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    scaler = StandardScaler()
    z = scaler.fit_transform(embeddings)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=seed, n_jobs=1)
    coords = reducer.fit_transform(z)

    labels_arr = np.asarray(labels)
    unique_labels = _genre_order(list(dict.fromkeys(labels_arr.tolist())))
    palette = sns.color_palette("tab10", n_colors=len(unique_labels))
    color_map = {g: palette[i] for i, g in enumerate(unique_labels)}

    for genre in unique_labels:
        mask = labels_arr == genre
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[color_map[genre]],
                   s=8, alpha=0.6, label=genre)

    ax.legend(markerscale=2, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax


def plot_alignment_uniformity(
    align_uniform: pd.DataFrame,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """
    Expects a DataFrame with columns: ratio_percent, alignment, uniformity, method.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for method in sorted(align_uniform["method"].unique()):
        sub = align_uniform[align_uniform["method"] == method].sort_values("ratio_percent")
        color = METHOD_COLORS.get(method)
        ax.plot(sub["ratio_percent"], sub["alignment"], marker="o", linewidth=2,
                markersize=5, label=f"{method} (align)", color=color, linestyle="-")
        ax.plot(sub["ratio_percent"], sub["uniformity"], marker="s", linewidth=2,
                markersize=5, label=f"{method} (unif)", color=color, linestyle="--")

    ax.set_xlabel("Compression ratio m/d (%)")
    ax.set_ylabel("Score")
    ax.set_xlim(left=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title)
    return ax


def plot_training_curve(
    checkpoint_path,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """
    Plots train/val loss from a checkpoint's epoch_history list.
    checkpoint_path: str or Path to a .pt file.
    """
    import torch

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    history = ckpt.get("epoch_history", [])
    if not history:
        raise ValueError(f"no epoch_history in {checkpoint_path}")

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    ax.plot(epochs, train_loss, label="train", linewidth=2, color="#1f77b4")
    ax.plot(epochs, val_loss, label="val", linewidth=2, color="#ff7f0e")

    best_ep = ckpt.get("best_epoch")
    if best_ep is not None:
        best_vl = ckpt.get("best_val_loss")
        ax.axvline(best_ep, color="gray", linestyle=":", linewidth=1.2,
                   label=f"best epoch={best_ep} val={best_vl:.4f}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title)
    return ax
