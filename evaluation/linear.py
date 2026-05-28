#!/usr/bin/env python3
#
# linear.py  Andrew Belles  April 13th, 2026
#
# Linear and kNN probe evaluation over waveform Barlow Twins parquet embeddings.
#

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from compression.train_utils import load_config

SPLITS = ("training", "validation", "test")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "linear.yaml"
DEFAULT_CONFIG = {
    "seed": 7,
    "c_grid": list(np.logspace(-4, 1, 12).tolist()),
    "max_iter": 2000,
    "tol": 1e-4,
    "knn_neighbors": [3, 5, 9, 15, 25],
}


def _embedding_columns(df: pd.DataFrame) -> list[str]:
    cols = sorted(c for c in df.columns if c.startswith("embedding_"))
    if not cols:
        raise ValueError("no embedding columns found")
    return cols


def _split_xy(df: pd.DataFrame, cols: list[str], exclude_genres: list[str] | None = None):
    if exclude_genres:
        df = df[~df["genre_top"].isin(exclude_genres)]
    df = df.dropna(subset=["genre_top"])
    le = LabelEncoder()
    le.fit(df["genre_top"])
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
        c_grid = list(np.logspace(-4, 1, 12))
    cols = _embedding_columns(df)
    splits, le = _split_xy(df, cols, exclude_genres)
    x_tr, y_tr = splits["training"]
    x_va, y_va = splits["validation"]
    x_te, y_te = splits["test"]

    scaler = StandardScaler().fit(x_tr)
    x_tr_s, x_va_s, x_te_s = scaler.transform(x_tr), scaler.transform(x_va), scaler.transform(x_te)

    best_C, best_val = None, -1.0
    for C in c_grid:
        clf = LogisticRegression(C=C, max_iter=max_iter, tol=tol, solver="saga",
                                 multi_class="multinomial", random_state=seed, n_jobs=4)
        clf.fit(x_tr_s, y_tr)
        val_f1 = float(f1_score(y_va, clf.predict(x_va_s), average="macro"))
        if val_f1 > best_val:
            best_val, best_C = val_f1, C

    clf = LogisticRegression(C=best_C, max_iter=max_iter, tol=tol, solver="saga",
                             multi_class="multinomial", random_state=seed, n_jobs=4)
    clf.fit(x_tr_s, y_tr)

    val_pred = clf.predict(x_va_s)
    test_pred = clf.predict(x_te_s)
    val_f1 = float(f1_score(y_va, val_pred, average="macro"))
    test_f1 = float(f1_score(y_te, test_pred, average="macro"))
    per_genre = dict(zip(le.classes_, f1_score(y_te, test_pred, average=None,
                                                labels=list(range(len(le.classes_)))).tolist()))
    return {"val_f1": val_f1, "test_f1": test_f1, "per_genre": per_genre, "best_C": best_C}


def run_knn_probe(
    df: pd.DataFrame,
    exclude_genres: list[str] | None = None,
    seed: int = 7,
    k_choices: list[int] | None = None,
) -> dict:
    if k_choices is None:
        k_choices = [3, 5, 9, 15, 25]
    cols = _embedding_columns(df)
    splits, le = _split_xy(df, cols, exclude_genres)
    x_tr, y_tr = splits["training"]
    x_va, y_va = splits["validation"]
    x_te, y_te = splits["test"]

    best_k, best_val = None, -1.0
    for k in k_choices:
        clf = Pipeline([("scaler", StandardScaler()),
                        ("knn", KNeighborsClassifier(n_neighbors=k, n_jobs=4))])
        clf.fit(x_tr, y_tr)
        val_f1 = float(f1_score(y_va, clf.predict(x_va), average="macro"))
        if val_f1 > best_val:
            best_val, best_k = val_f1, k

    clf = Pipeline([("scaler", StandardScaler()),
                    ("knn", KNeighborsClassifier(n_neighbors=best_k, n_jobs=4))])
    clf.fit(x_tr, y_tr)

    val_f1 = float(f1_score(y_va, clf.predict(x_va), average="macro"))
    test_f1 = float(f1_score(y_te, clf.predict(x_te), average="macro"))
    return {"val_f1": val_f1, "test_f1": test_f1, "best_k": best_k}


def run_ratio_curve(
    parquet_path: Path,
    exclude_genres: list[str] | None = None,
    seed: int = 7,
    c_grid: list[float] | None = None,
) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    records = []
    for (method, ratio), group in df.groupby(["method", "ratio_percent"], dropna=False):
        result = run_linear_probe(group, exclude_genres=exclude_genres,
                                  seed=seed, c_grid=c_grid)
        records.append({"method": method, "ratio_percent": ratio,
                        "val_f1": result["val_f1"], "test_f1": result["test_f1"]})
    return pd.DataFrame.from_records(records).sort_values(["method", "ratio_percent"])


def _probe_group(group: pd.DataFrame, config: dict) -> dict:
    seed = int(config["seed"])
    c_grid = [float(c) for c in config.get("c_grid", list(np.logspace(-4, 1, 12)))]
    linear = run_linear_probe(group, seed=seed, c_grid=c_grid,
                              max_iter=int(config["max_iter"]), tol=float(config["tol"]))
    knn = run_knn_probe(group, seed=seed,
                        k_choices=[int(k) for k in config["knn_neighbors"]])
    first = group.iloc[0]
    ratio = first.get("ratio_percent", None)
    return {
        "method": str(first.get("method", "")),
        "ratio_percent": None if pd.isna(ratio) else int(ratio),
        "best_C": linear["best_C"],
        "best_k": knn["best_k"],
        "validation_f1_macro": linear["val_f1"],
        "test_f1_macro": linear["test_f1"],
        "knn_val_f1_macro": knn["val_f1"],
        "knn_test_f1_macro": knn["test_f1"],
        "n_train": int(len(group[group["split"] == "training"])),
        "n_val": int(len(group[group["split"] == "validation"])),
        "n_test": int(len(group[group["split"] == "test"])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear and kNN probe evaluation.")
    parser.add_argument("-p", "--parquet", type=Path, required=True)
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_CONFIG)
    parquet_path = args.parquet.expanduser().resolve()
    if not parquet_path.exists():
        raise FileNotFoundError(f"parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"START module=evaluation.linear parquet={parquet_path} rows={len(df)}", flush=True)

    records = []
    for keys, group in df.groupby(["method", "ratio_percent"], dropna=False):
        label = "_".join(str(k) for k in (keys if isinstance(keys, tuple) else (keys,)))
        print(f"probing group={label} n={len(group)}", flush=True)
        records.append(_probe_group(group, config))

    summary = pd.DataFrame.from_records(records)
    data_root = Path(__file__).resolve().parent / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    out_path = data_root / f"{parquet_path.stem}_probe_summary.csv"
    summary.to_csv(out_path, index=False)
    print(summary[["method", "ratio_percent", "validation_f1_macro", "test_f1_macro",
                    "knn_test_f1_macro"]].to_string(index=False), flush=True)
    print(f"DONE saved={out_path}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
