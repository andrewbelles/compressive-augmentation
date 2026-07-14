import json
from pathlib import Path

import numpy as np
import pandas as pd

from rf.preprocess.manifests import MOD_CLASSES

# Shard-level aggregation for the SRC ladder. Everything is a pure function of
# the parquet shards (+ meta sidecars), so the driver's in-job best-config
# selection and the offline analysis CLI share one implementation.

SPEC_COLS = ["rung", "operator_family", "rho", "pipeline", "dict_variant",
             "error_mode", "corruption", "seed"]


def load_ladder_results(results_dir: Path, rungs: list[int] | None = None) -> pd.DataFrame:
    """Concatenate per-spec parquet shards (optionally restricted to some rungs)."""
    files = sorted(Path(results_dir).glob("*.parquet"))
    frames = []
    for path in files:
        df = pd.read_parquet(path)
        if rungs is not None and int(df["rung"].iloc[0]) not in rungs:
            continue
        df["spec_name"] = path.stem
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no matching shards under {results_dir} (rungs={rungs})")
    return pd.concat(frames, ignore_index=True)


def load_meta(results_dir: Path) -> pd.DataFrame:
    """Load the per-spec meta sidecars (one row per shard) as a DataFrame."""
    rows = []
    for path in sorted(Path(results_dir).glob("*.meta.json")):
        row = json.loads(path.read_text())
        row["spec_name"] = path.name.removesuffix(".meta.json")
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no meta sidecars under {results_dir}")
    return pd.DataFrame(rows)


def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    for col, val in filters.items():
        df = df[df[col].isin(val)] if isinstance(val, (list, tuple, set)) else df[df[col] == val]
    return df


def accuracy_by(df: pd.DataFrame, by: list[str], **filters) -> pd.DataFrame:
    """Group-wise classification accuracy with sample counts."""
    df = _apply_filters(df, filters)
    correct = (df["mod_true"] == df["mod_pred"]).astype(float)
    out = df.assign(correct=correct).groupby(by, as_index=False).agg(
        accuracy=("correct", "mean"), n=("correct", "size"))
    return out


def accuracy_vs_snr(df: pd.DataFrame, **filters) -> pd.DataFrame:
    return accuracy_by(df, ["snr"], **filters).sort_values("snr", ignore_index=True)


def accuracy_vs_rho(df: pd.DataFrame, snr_bands: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """Accuracy per (family, pipeline, rho, SNR band); expects rung-2/3 style shards."""
    parts = []
    for band, (lo, hi) in snr_bands.items():
        sub = df[(df["snr"] >= lo) & (df["snr"] <= hi)]
        if len(sub) == 0:
            continue
        agg = accuracy_by(sub, ["operator_family", "pipeline", "rho"])
        agg["snr_band"] = band
        parts.append(agg)
    return pd.concat(parts, ignore_index=True)


def confusion_matrix(df: pd.DataFrame, **filters) -> pd.DataFrame:
    """Row-normalized 24 x 24 confusion matrix over MOD_CLASSES."""
    df = _apply_filters(df, filters)
    mat = pd.crosstab(df["mod_true"], df["mod_pred"]).reindex(
        index=MOD_CLASSES, columns=MOD_CLASSES, fill_value=0).astype(float)
    row_sums = mat.sum(axis=1).replace(0.0, np.nan)
    return mat.div(row_sums, axis=0).fillna(0.0)


def recovery_surface(df: pd.DataFrame) -> pd.DataFrame:
    """Mean recovery success per (family, rho, snr) for reconstruct-pipeline shards."""
    sub = df[df["pipeline"] == "reconstruct"]
    return sub.assign(rec=sub["recovered"].astype(float)).groupby(
        ["operator_family", "rho", "snr"], as_index=False).agg(recovered=("rec", "mean"))


def select_best_config(df: pd.DataFrame, snr_min: int = 10) -> tuple[str, float, str]:
    """Pick the (family, rho, pipeline) with best mean accuracy at snr >= snr_min.

    Deterministic tie-break (accuracy desc, then family/rho/pipeline asc) so the
    two --half processes always compute the same answer from the same shards.
    """
    sub = df[df["snr"] >= snr_min]
    agg = accuracy_by(sub, ["operator_family", "rho", "pipeline"])
    agg = agg[agg["operator_family"] != "identity"]
    if len(agg) == 0:
        raise ValueError("select_best_config: no non-identity configs to select from")
    agg = agg.sort_values(["accuracy", "operator_family", "rho", "pipeline"],
                          ascending=[False, True, True, True], ignore_index=True)
    best = agg.iloc[0]
    return str(best["operator_family"]), float(best["rho"]), str(best["pipeline"])


def rung_delta_table(df: pd.DataFrame, snr_min: int = 10) -> pd.DataFrame:
    """Per-config accuracy at snr >= snr_min with the delta vs the previous rung's best."""
    sub = df[df["snr"] >= snr_min]
    agg = accuracy_by(sub, SPEC_COLS).sort_values(SPEC_COLS, ignore_index=True)
    best_prev, deltas = {}, []
    for rung in sorted(agg["rung"].unique()):
        prev = [r for r in best_prev if r < rung]
        base = best_prev[max(prev)] if prev else np.nan
        rows = agg[agg["rung"] == rung]
        deltas.extend((rows["accuracy"] - base).tolist())
        best_prev[rung] = rows["accuracy"].max()
    agg["delta_vs_prev_rung"] = deltas
    return agg
