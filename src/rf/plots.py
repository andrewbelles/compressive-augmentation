import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rf.preprocess.manifests import MOD_CLASSES

# One figure per function; callers own saving/closing. Styling follows the
# lightweight conventions of the repo's plot.py.

FIG_BG = "#FFFFFF"
AX_BG  = "#F7F7F7"
GRID_C = "#DDDDDD"

_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _style_axis(ax):
    ax.set_facecolor(AX_BG)
    ax.grid(True, color=GRID_C, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def plot_accuracy_vs_snr(curves: dict[str, pd.DataFrame], title: str) -> plt.Figure:
    """Overlay accuracy-vs-SNR curves; each value is an accuracy_vs_snr() frame."""
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=FIG_BG)
    for i, (label, df) in enumerate(curves.items()):
        ax.plot(df["snr"], df["accuracy"], marker="o", markersize=3,
                color=_CYCLE[i % len(_CYCLE)], label=label)
    ax.axhline(1.0 / len(MOD_CLASSES), color="gray", linestyle="--",
               linewidth=1, label="chance")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=8)
    _style_axis(ax)
    fig.tight_layout()
    return fig


def plot_accuracy_vs_rho(df: pd.DataFrame, title: str) -> plt.Figure:
    """Accuracy vs measurement ratio, one panel per SNR band, lines per (family, pipeline).

    Expects the accuracy_vs_rho() output (columns operator_family, pipeline,
    rho, accuracy, snr_band).
    """
    bands = list(df["snr_band"].unique())
    fig, axes = plt.subplots(1, len(bands), figsize=(4.5 * len(bands), 4),
                             facecolor=FIG_BG, squeeze=False)
    for ax, band in zip(axes[0], bands):
        sub = df[df["snr_band"] == band]
        for i, ((family, pipeline), grp) in enumerate(sub.groupby(["operator_family", "pipeline"])):
            grp = grp.sort_values("rho")
            ax.plot(grp["rho"], grp["accuracy"], marker="o", markersize=3,
                    color=_CYCLE[i % len(_CYCLE)], label=f"{family}/{pipeline}")
        ax.set_xlabel(r"measurement ratio $\rho$")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1)
        ax.set_title(f"SNR band: {band}")
        ax.legend(fontsize=7)
        _style_axis(ax)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_confusion(conf: pd.DataFrame, title: str) -> plt.Figure:
    """Row-normalized confusion matrix heatmap (confusion_matrix() output)."""
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=FIG_BG)
    im = ax.imshow(conf.to_numpy(), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(conf.columns)), conf.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(conf.index)), conf.index, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_phase_transition(surface: pd.DataFrame, title: str) -> plt.Figure:
    """Recovery phase-transition heatmaps (rho x SNR), one panel per family.

    Expects the recovery_surface() output (columns operator_family, rho, snr,
    recovered).
    """
    families = list(surface["operator_family"].unique())
    fig, axes = plt.subplots(1, len(families), figsize=(4.5 * len(families), 4),
                             facecolor=FIG_BG, squeeze=False)
    for ax, family in zip(axes[0], families):
        sub = surface[surface["operator_family"] == family]
        grid = sub.pivot_table(index="rho", columns="snr", values="recovered")
        im = ax.imshow(grid.to_numpy(), origin="lower", aspect="auto",
                       vmin=0, vmax=1, cmap="magma",
                       extent=[grid.columns.min(), grid.columns.max(),
                               0, len(grid.index)])
        ax.set_yticks(np.arange(len(grid.index)) + 0.5, [f"{r:g}" for r in grid.index])
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(r"$\rho$")
        ax.set_title(family)
        fig.colorbar(im, ax=ax, fraction=0.046, label="P(recovered)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
