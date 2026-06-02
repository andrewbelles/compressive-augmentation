#!/usr/bin/env python3
#
# perturbation.py  Andrew Belles  May 2026
#
# Geometry analysis of CS augmentation representations relative to a SupCon W3
# reference manifold.  Loads embeddings directly from the parquet; no npy cache.
#
# Usage:
#   python -m evaluation.perturbation [--parquet PATH] [--ref METHOD] [--split SPLIT]
#

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder, StandardScaler

DEFAULT_PARQUET = Path("representation/data/wave_barlow_fma_small.parquet")
DEFAULT_REF     = "wave_barlow_abt_w3_d256_nopop"

_FIG_BG  = "#191919"
_AX_BG   = "#202020"
_GRID_C  = "#4A4A4A"
_TEXT    = "#D8D8D8"
_CS_STYLES = {
    "dct_biased":  ("#7E57C2", "o", "-",  "DCT biased"),
    "dct_uniform": ("#FF8C42", "s", "--", "DCT uniform"),
    "srht":        ("#F4D35E", "^", "--", "SRHT"),
}


def _parse_method(method: str) -> tuple[str, int | None]:
    base = re.sub(r"_s\d+$", "", method)
    m = re.search(r"_r0?(\d+)", base)
    ratio = int(m.group(1)) if m else None
    if "srht" in base:
        return "srht", ratio
    if "uniform" in base:
        return "dct_uniform", ratio
    if "wave_barlow_cs" in base:
        return "dct_biased", ratio
    return "abt", ratio


def _embedding_cols(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("embedding_") and df[c].notna().all())


def _load_split(df: pd.DataFrame, method: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    sub  = df[(df["method"].str.startswith(method)) & (df["split"] == split)].dropna(subset=["genre_top"])
    cols = _embedding_cols(sub)
    return sub[cols].to_numpy(dtype=np.float32), sub["genre_top"].to_numpy()


def build_lda(X_train: np.ndarray, y_train: np.ndarray, n_components: int | None = None) -> LinearDiscriminantAnalysis:
    le  = LabelEncoder().fit(y_train)
    yc  = le.transform(y_train)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X_train, yc)
    return lda


def linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    num = float(np.linalg.norm(A.T @ B, "fro") ** 2)
    den = float(np.linalg.norm(A.T @ A, "fro") * np.linalg.norm(B.T @ B, "fro"))
    return num / den if den > 1e-12 else 0.0


def analyze(
    parquet_path: Path = DEFAULT_PARQUET,
    ref_method: str = DEFAULT_REF,
    split: str = "test",
) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)

    X_ref_tr, y_ref_tr = _load_split(df, ref_method, "training")
    X_ref_te, y_ref_te = _load_split(df, ref_method, split)

    scaler  = StandardScaler().fit(X_ref_tr)
    X_ref_tr_s = scaler.transform(X_ref_tr)
    X_ref_te_s = scaler.transform(X_ref_te)

    n_classes  = len(np.unique(y_ref_tr))
    lda        = build_lda(X_ref_tr_s, y_ref_tr, n_components=n_classes - 1)
    sem_basis  = lda.scalings_[:, :n_classes - 1]
    sem_basis  = sem_basis / np.linalg.norm(sem_basis, axis=0, keepdims=True)

    def _project(X):
        Xs = scaler.transform(X)
        Xs_sem  = Xs @ sem_basis @ sem_basis.T
        Xs_nuis = Xs - Xs_sem
        return Xs, Xs_sem, Xs_nuis

    ref_s, ref_sem, ref_nuis = _project(X_ref_te)

    cs_methods = df[df["method"].apply(lambda m: _parse_method(m)[0] != "abt")]["method"].unique()
    rows = []
    for method in sorted(cs_methods):
        X, _ = _load_split(df, method, split)
        if len(X) == 0:
            continue
        group, ratio = _parse_method(method)
        Xs, Xs_sem, Xs_nuis = _project(X)

        delta_norm    = float(np.linalg.norm(Xs - ref_s, axis=1).mean())
        sem_norm      = float(np.linalg.norm(Xs_sem, axis=1).mean())
        nuis_norm     = float(np.linalg.norm(Xs_nuis, axis=1).mean())
        sem_ratio     = sem_norm / (sem_norm + nuis_norm + 1e-12)
        cka_full      = linear_cka(Xs,     ref_s)
        cka_sem       = linear_cka(Xs_sem,  ref_sem)
        cka_nuis      = linear_cka(Xs_nuis, ref_nuis)

        rows.append({
            "method": method, "group": group, "ratio": ratio,
            "delta_norm": delta_norm, "sem_norm": sem_norm,
            "nuis_norm": nuis_norm, "sem_ratio": sem_ratio,
            "cka_full": cka_full, "cka_sem": cka_sem, "cka_nuis": cka_nuis,
        })
        print(f"{method:55s}  delta={delta_norm:.3f}  sem_ratio={sem_ratio:.3f}  cka_full={cka_full:.3f}", flush=True)

    return pd.DataFrame(rows).sort_values(["group", "ratio"])


def plot_results(results: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    fig.patch.set_facecolor(_FIG_BG)

    metrics = [
        ("sem_ratio",  "Semantic ratio (sem / total)"),
        ("cka_full",   "CKA (full)"),
        ("cka_sem",    "CKA (semantic subspace)"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        ax.set_facecolor(_AX_BG)
        ax.grid(True, color=_GRID_C, linewidth=0.6, alpha=0.65)
        ax.tick_params(colors=_TEXT)
        ax.xaxis.label.set_color(_TEXT)
        ax.yaxis.label.set_color(_TEXT)
        ax.title.set_color(_TEXT)
        for spine in ax.spines.values():
            spine.set_color("#666666")

        for group, (color, marker, ls, label) in _CS_STYLES.items():
            sub = results[results["group"] == group].sort_values("ratio")
            if sub.empty:
                continue
            ax.plot(sub["ratio"], sub[metric], color=color, marker=marker,
                    linestyle=ls, linewidth=2.0, markersize=5, label=label)

        ax.set_xlabel("Compression ratio m/d (%)")
        ax.set_ylabel(ylabel)
        legend = ax.legend(framealpha=0.92)
        if legend:
            legend.get_frame().set_facecolor("#242424")
            legend.get_frame().set_edgecolor("#666666")
            for t in legend.get_texts():
                t.set_color(_TEXT)

    fig.tight_layout(pad=0.4)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description="CS representation geometry analysis vs SupCon W3 reference.")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--ref",     type=str,  default=DEFAULT_REF)
    parser.add_argument("--split",   type=str,  default="test", choices=("test", "validation"))
    parser.add_argument("--save",    type=Path, default=None,
                        help="path to save the figure (e.g. evaluation/images/perturbation.png)")
    args = parser.parse_args()

    results = analyze(args.parquet, args.ref, args.split)
    fig     = plot_results(results)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"saved figure to {args.save}", flush=True)
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
