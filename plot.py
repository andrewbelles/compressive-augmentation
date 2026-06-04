#!/usr/bin/env python3
#
# plot.py  Andrew Belles  June 2026
#
# Generates the three paper figures from analysis CSVs produced by analyze.py.
#
#   Figure 2: Test F1-macro vs measurement ratio (+ baselines)
#   Figure 3: Nuisance perturbation magnitude vs Test F1-macro and Between-view alignment
#   Figure 4: Alignment and Uniformity vs Test F1-macro
#
# Usage:
#   python plot.py --analysis-dir analysis/ --output-dir images/
#

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/spiky-matplotlib")

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    print("\nDone.")


if __name__ == "__main__":
    main()
