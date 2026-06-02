#!/usr/bin/env python3
#
# linear.py  Andrew Belles  April 13th, 2026
#
# Linear probe evaluation over waveform Barlow Twins parquet embeddings.
#

import re

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from representation.audio import load_manifest

SPLITS = ("training", "validation", "test")

_C_GRID = list(np.logspace(-4, 1, 12).tolist())


def load_mel_embeddings(data_dir, splits=None) -> pd.DataFrame:
    from pathlib import Path
    data_dir = Path(data_dir)
    if splits is None:
        splits = SPLITS
    frames = []
    for split in SPLITS:
        if split not in splits:
            continue
        manifest = load_manifest(data_dir, split)
        rows = []
        for _, row in manifest.iterrows():
            mel_path = data_dir.parent / str(row["mel_path"])
            if not mel_path.exists():
                continue
            mel = torch.load(mel_path, map_location="cpu", weights_only=True)
            if mel.dim() == 3:
                mel = mel.squeeze(0)
            pooled = mel.mean(dim=1).numpy().astype(np.float32)
            rows.append({
                "track_id": int(row["track_id"]),
                "genre_top": str(row["genre_top"]),
                "split": split,
                "method": "raw_mel",
                **{f"embedding_{i:04d}": float(pooled[i]) for i in range(len(pooled))},
            })
        if rows:
            frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def _embedding_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("embedding_") and df[c].notna().all())


def _split_xy(df: pd.DataFrame, cols: list[str], exclude_genres: list[str] | None = None):
    if exclude_genres:
        df = df[~df["genre_top"].isin(exclude_genres)]
    df = df.dropna(subset=["genre_top"])
    le = LabelEncoder().fit(df["genre_top"])
    result = {}
    for split in SPLITS:
        sub = df[df["split"] == split]
        result[split] = (sub[cols].to_numpy(dtype=np.float32), le.transform(sub["genre_top"]))
    return result, le


def run_linear_probe(
    df: pd.DataFrame,
    exclude_genres: list[str] | None = None,
    seed: int = 7,
    c_grid: list[float] | None = None,
    max_iter: int = 2000,
    tol: float = 1e-4,
) -> dict:
    if c_grid is None:
        c_grid = _C_GRID
    cols = _embedding_columns(df)
    splits, le = _split_xy(df, cols, exclude_genres)
    x_tr, y_tr = splits["training"]
    x_va, y_va = splits["validation"]
    x_te, y_te = splits["test"]

    scaler = StandardScaler().fit(x_tr)
    x_tr_s, x_va_s, x_te_s = scaler.transform(x_tr), scaler.transform(x_va), scaler.transform(x_te)

    best_C, best_val = None, -1.0
    for C in c_grid:
        clf = LogisticRegression(C=C, max_iter=max_iter, tol=tol, solver="liblinear", random_state=seed)
        clf.fit(x_tr_s, y_tr)
        vf1 = float(f1_score(y_va, clf.predict(x_va_s), average="macro"))
        if vf1 > best_val:
            best_val, best_C = vf1, C

    clf = LogisticRegression(C=best_C, max_iter=max_iter, tol=tol, solver="liblinear", random_state=seed)
    clf.fit(x_tr_s, y_tr)

    val_pred  = clf.predict(x_va_s)
    test_pred = clf.predict(x_te_s)
    val_f1    = float(f1_score(y_va, val_pred, average="macro"))
    test_f1   = float(f1_score(y_te, test_pred, average="macro"))
    per_genre = dict(zip(le.classes_, f1_score(y_te, test_pred, average=None,
                                               labels=list(range(len(le.classes_)))).tolist()))

    rng   = np.random.default_rng(seed)
    n_te  = len(y_te)
    boots = [float(f1_score(y_te[idx], test_pred[idx], average="macro"))
             for idx in (rng.integers(0, n_te, size=n_te) for _ in range(1000))]
    boot_arr     = np.array(boots)
    test_f1_ci   = float(np.percentile(boot_arr, 97.5) - np.percentile(boot_arr, 2.5)) / 2.0

    return {"val_f1": val_f1, "test_f1": test_f1, "test_f1_ci": test_f1_ci,
            "per_genre": per_genre, "best_C": best_C}


def _method_label(method: str, group: pd.DataFrame) -> str:
    labels = {
        "raw_mel":                           "Raw Mel",
        "wave_barlow_abt_w2_d256_nopop":     "W2",
        "wave_barlow_abt_w3_d256_nopop":     "W3",
        "wave_barlow_abt_w4_d256_nopop":     "W4",
        "wave_barlow_abt_w3_d256_sup_nopop": "W3-Sup",
        "supcon_w3_d256_nopop":              "SupCon-W3",
    }
    base = re.sub(r"_s\d+$", "", method)
    if base in labels:
        return labels[base]
    ratio = int(group["ratio_percent"].iloc[0]) if "ratio_percent" in group.columns else "?"
    if "srht" in method:
        return f"SRHT r{ratio}"
    if "uniform" in method:
        return f"DCT uniform r{ratio}"
    return f"DCT biased r{ratio}"


def run_probe_suite(
    df: pd.DataFrame,
    methods: list[str] | None = None,
    exclude_genres: list[str] | None = None,
) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    for method, group in df.groupby("method"):
        if methods is not None and method not in methods:
            continue
        result = run_linear_probe(group, exclude_genres=exclude_genres)
        label  = _method_label(method, group)
        cache[label] = result
        f1, ci = result["test_f1"], result["test_f1_ci"]
        print(f"{label:35s} val={result['val_f1']:.4f}  test=[{f1 - ci:.4f}, {f1 + ci:.4f}]")
    return cache


def _base_method(method: str) -> str:
    return re.sub(r"_s\d+$", "", str(method))


def run_probe_suite_seeded(
    df: pd.DataFrame,
    methods: list[str] | None = None,
    exclude_genres: list[str] | None = None,
) -> dict[str, dict]:
    """Group encoder-seeded runs (_sN suffix), pool F1 across seeds, report t-CI (df=n_seeds-1)."""
    from scipy import stats

    cache: dict[str, dict] = {}
    groups: dict[str, dict[int, pd.DataFrame]] = {}

    for method, group in df.groupby("method"):
        if methods is not None and method not in methods:
            continue
        base = _base_method(method)
        seed = int(group["encoder_seed"].iloc[0]) if "encoder_seed" in group.columns else 0
        groups.setdefault(base, {})[seed] = group

    for base, seed_groups in groups.items():
        if not seed_groups:
            continue
        any_group  = next(iter(seed_groups.values()))
        label      = _method_label(base, any_group)
        seed_f1s   = []
        best_C_per = []

        for seed, group in seed_groups.items():
            result = run_linear_probe(group, exclude_genres=exclude_genres)
            seed_f1s.append(result["test_f1"])
            best_C_per.append(result["best_C"])

        arr  = np.array(seed_f1s)
        mean = float(arr.mean())
        if len(arr) > 1:
            t_val  = stats.t.ppf(0.975, df=len(arr) - 1)
            ci     = float(t_val * arr.std(ddof=1) / np.sqrt(len(arr)))
        else:
            ci = 0.0

        combined = {"val_f1": mean, "test_f1": mean, "test_f1_ci": ci,
                    "per_genre": {}, "best_C": float(np.mean(best_C_per)),
                    "n_seeds": len(arr), "seed_f1s": seed_f1s}
        cache[label] = combined
        print(f"{label:35s} seeds={len(arr)} test=[{mean - ci:.4f}, {mean + ci:.4f}]")

    return cache
