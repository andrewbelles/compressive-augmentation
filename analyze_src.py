#!/usr/bin/env python3
#
# analyze_src.py  Andrew Belles  July 2026
#
# Thin CLI over rf.analysis / rf.plots: aggregates the SRC-ladder parquet
# shards into per-rung CSV tables and figures.
#
# Usage:
#   python analyze_src.py \
#       --results-dir results/src_ladder \
#       --output-dir  analysis/src_ladder
#

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from rf.analysis import (
    accuracy_vs_rho,
    accuracy_vs_snr,
    confusion_matrix,
    load_ladder_results,
    load_meta,
    recovery_surface,
    rung_delta_table,
    select_best_config,
)
from rf.ladder import BEST_SNR_MIN, SNR_BANDS
from rf.plots import (
    plot_accuracy_vs_rho,
    plot_accuracy_vs_snr,
    plot_confusion,
    plot_phase_transition,
)


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote {name}.png", flush=True)


def _snr_curves(df, key_cols: list[str]) -> dict:
    return {
        " ".join(str(v) for v in (key if isinstance(key, tuple) else (key,))):
            accuracy_vs_snr(grp)
        for key, grp in df.groupby(key_cols)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate SRC-ladder shards into tables/figures.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir",  type=Path, required=True)
    parser.add_argument("--snr-min",     type=int, default=BEST_SNR_MIN)
    args = parser.parse_args()

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    df = load_ladder_results(args.results_dir)
    rungs = sorted(df["rung"].unique())
    print(f"loaded {len(df)} rows across rungs {rungs}", flush=True)

    # R1: raw-SRC ceiling
    if 1 in rungs:
        r1 = df[df["rung"] == 1]
        accuracy_vs_snr(r1).to_csv(out / "r1_accuracy_vs_snr.csv", index=False)
        _save(plot_accuracy_vs_snr({"identity / raw SRC": accuracy_vs_snr(r1)},
                                   "R1: SRC ceiling (Phi = I)"), out, "r1_accuracy_vs_snr")
        conf = confusion_matrix(r1[r1["snr"] >= args.snr_min])
        conf.to_csv(out / "r1_confusion_high_snr.csv")
        _save(plot_confusion(conf, f"R1 confusion (snr >= {args.snr_min})"),
              out, "r1_confusion_high_snr")

    # R2/R3: compression sweep + reconstruct-vs-smashed
    r23 = df[df["rung"].isin([2, 3])]
    if len(r23):
        rho_table = accuracy_vs_rho(r23, SNR_BANDS)
        rho_table.to_csv(out / "r23_accuracy_vs_rho.csv", index=False)
        _save(plot_accuracy_vs_rho(rho_table, "R2/R3: accuracy vs measurement ratio"),
              out, "r23_accuracy_vs_rho")
        surface = recovery_surface(r23)
        if len(surface):
            surface.to_csv(out / "r2_recovery_surface.csv", index=False)
            _save(plot_phase_transition(surface, "R2: recovery phase transition"),
                  out, "r2_phase_transition")
        best = select_best_config(r23, snr_min=args.snr_min)
        print(f"BEST family={best[0]} rho={best[1]:g} pipeline={best[2]}", flush=True)
        at_best = r23[(r23["operator_family"] == best[0]) & (r23["rho"] == best[1])]
        _save(plot_accuracy_vs_snr(_snr_curves(at_best, ["pipeline"]),
                                   f"R2 vs R3 at best config ({best[0]}, rho={best[1]:g})"),
              out, "r23_pipeline_overlay_at_best")

    # R4: V0 vs V1 at best config
    if 4 in rungs:
        r4 = df[df["rung"] == 4]
        base = r23[(r23["operator_family"] == r4["operator_family"].iloc[0]) &
                   (r23["rho"] == r4["rho"].iloc[0]) &
                   (r23["pipeline"] == r4["pipeline"].iloc[0])]
        curves = {"v0": accuracy_vs_snr(base), "v1": accuracy_vs_snr(r4)}
        _save(plot_accuracy_vs_snr(curves, "R4: exemplar (v0) vs orbit (v1) dictionary"),
              out, "r4_v0_vs_v1")

    # R5: corruption x error-handling + coherence table
    if 5 in rungs:
        r5 = df[df["rung"] == 5]
        table = (r5[r5["snr"] >= args.snr_min]
                 .assign(correct=lambda d: (d["mod_true"] == d["mod_pred"]).astype(float))
                 .groupby(["corruption", "error_mode"], as_index=False)
                 .agg(accuracy=("correct", "mean")))
        table.to_csv(out / "r5_corruption_table.csv", index=False)
        print(table.to_string(index=False), flush=True)
        _save(plot_accuracy_vs_snr(_snr_curves(r5, ["corruption", "error_mode"]),
                                   "R5: corruption x error handling"), out, "r5_error_handling")
        try:
            meta = load_meta(args.results_dir)
            cols = [c for c in ("rung", "corruption", "error_mode", "mu_phid", "mu_phid_phie")
                    if c in meta.columns]
            meta.loc[meta["rung"] == 5, cols].to_csv(out / "r5_coherence.csv", index=False)
        except FileNotFoundError:
            pass

    # R6: dictionary-variant ladder + final delta table
    if 6 in rungs:
        variants = df[df["rung"].isin([3, 4, 6]) | (df["rung"] == 2)]
        best = select_best_config(r23, snr_min=args.snr_min)
        at_best = variants[(variants["operator_family"] == best[0]) &
                           (variants["rho"] == best[1]) &
                           (variants["pipeline"] == best[2])]
        if len(at_best):
            _save(plot_accuracy_vs_snr(_snr_curves(at_best, ["dict_variant"]),
                                       "R6: dictionary variants v0-v3"), out, "r6_variants")

    delta = rung_delta_table(df, snr_min=args.snr_min)
    delta.to_csv(out / "rung_delta_table.csv", index=False)
    try:
        md = delta.to_markdown(index=False)          # needs optional tabulate
    except ImportError:
        md = "```\n" + delta.to_string(index=False) + "\n```"
    (out / "rung_delta_table.md").write_text(md + "\n")
    print(delta.to_string(index=False), flush=True)
    print("DONE analyze_src", flush=True)


if __name__ == "__main__":
    main()
