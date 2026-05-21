#!/usr/bin/env python3
#
# compare.py  Andrew Belles  May 19th, 2026
#
# Unified results table comparing linear probe results across methods.
# Reads evaluation/data/linear_logistic_summary.csv and filters by method prefix.
#
# Prints a compact markdown table and saves CSV.
#

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def log(msg: str) -> None:
    print(msg, flush=True)


def report(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified comparison table across SSL methods.")
    parser.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing evaluation CSVs. Defaults to {DEFAULT_DATA_DIR}.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_DATA_DIR / "compare_summary.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--prefixes",
        nargs="*",
        default=["barlow_", "cs_vicreg_"],
        help="Method prefixes to include. Defaults to barlow_ and cs_vicreg_.",
    )
    return parser.parse_args()


def load_linear_summary(data_dir: Path, prefixes: list[str]) -> pd.DataFrame:
    path = data_dir / "linear_logistic_summary.csv"
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    mask = frame["method"].apply(lambda m: any(str(m).startswith(p) for p in prefixes))
    return frame[mask].copy()


def format_markdown(frame: pd.DataFrame) -> str:
    keep = {"method", "classifier_type", "m_dim", "ratio_percent",
            "test_accuracy", "test_f1_macro", "test_pr_auc_macro"}
    cols = [c for c in frame.columns if c in keep]
    sub = frame[cols].copy()
    for col in ["test_accuracy", "test_f1_macro", "test_pr_auc_macro"]:
        if col in sub.columns:
            sub[col] = sub[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
    header = " | ".join(str(c) for c in sub.columns)
    sep = " | ".join(["---"] * len(sub.columns))
    rows = [" | ".join(str(v) for v in row) for row in sub.itertuples(index=False)]
    return "\n".join(["| " + header + " |", "| " + sep + " |"] + ["| " + r + " |" for r in rows])


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    report(f"START module=evaluation.compare data_dir={data_dir}")

    frame = load_linear_summary(data_dir, args.prefixes)
    if frame.empty:
        log("No results found. Run evaluation/linear.py first.")
        return 1

    sort_cols = [c for c in ["method", "m_dim", "ratio_percent"] if c in frame.columns]
    frame = frame.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    log(f"saved compare_summary={args.output} rows={len(frame)}")
    log("\n" + format_markdown(frame))
    report(f"DONE module=evaluation.compare rows={len(frame)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
