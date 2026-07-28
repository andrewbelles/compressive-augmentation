#!/usr/bin/env python3
"""Aggregate hypothesis artifacts into accept/reject verdicts and figures."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.utils import enable_fast_matmul
from rf.analysis import ARTIFACT_SOURCE, HIGH_SNR_DB, build_verdicts, is_go
from rf.hypotheses._artifacts import load_records


def _load(out_dir: Path, mechanism: str):
    """Read a mechanism's rows, following the redirect when another driver measured them."""
    return load_records(out_dir, ARTIFACT_SOURCE.get(mechanism, mechanism))


def _save(fig_dir: Path, name: str):
    plt.savefig(fig_dir / f"{name}.png", dpi=120, bbox_inches="tight")
    plt.close()


def _high(df):
    hi = df[df["snr"] >= HIGH_SNR_DB] if "snr" in df else df
    return hi if not hi.empty else df


def _plot_tradeoff(out_dir: Path, fig_dir: Path):
    df = _load(out_dir, "label_nuisance_tradeoff")
    if df.empty:
        return
    df = df[df["arm"] == "cs"] if "arm" in df else df
    g = _high(df).groupby("rho")[["label_retention", "view_diversity", "nuisance_collapse"]].mean()
    plt.figure()
    plt.plot(g.index, g["label_retention"], "o-", label="label retention")
    plt.plot(g.index, g["view_diversity"], "s-", label="view diversity")
    plt.plot(g.index, g["nuisance_collapse"], "^-", label="nuisance collapse")
    plt.axhline(1.0, color="gray", lw=0.8, ls=":")
    plt.xlabel("rho = m/N")
    plt.ylabel("ratio")
    plt.title(f"Label retention vs view diversity (SNR >= {HIGH_SNR_DB} dB)")
    plt.legend()
    _save(fig_dir, "label_nuisance_tradeoff")


def _plot_se(out_dir: Path, fig_dir: Path):
    df = _load(out_dir, "se_calibration")
    if df.empty:
        return
    g = _high(df).groupby("rho")[["realized_snr", "se_snr"]].mean()
    plt.figure()
    plt.plot(g.index, g["se_snr"], "o-", label="SE prediction")
    plt.plot(g.index, g["realized_snr"], "s-", label="realized OAMP")
    plt.xlabel("rho = m/N")
    plt.ylabel("recovery SNR (dB)")
    plt.title(f"Layer IV calibration (SNR >= {HIGH_SNR_DB} dB)")
    plt.legend()
    _save(fig_dir, "se_calibration")


def _plot_compressibility(out_dir: Path, fig_dir: Path):
    df = load_records(out_dir, "dictionary_compressibility")
    if df.empty:
        return
    plt.figure()
    for name, sub in df.groupby("dictionary"):
        g = sub.groupby("snr")["k90_over_d"].mean()
        plt.plot(g.index, g.values, "o-", label=name)
    plt.xlabel("SNR (dB)")
    plt.ylabel("k90 / n_atoms")
    plt.title("Compressibility by dictionary")
    plt.legend()
    _save(fig_dir, "dictionary_compressibility")


def _plot_backprojection(out_dir: Path, fig_dir: Path):
    df = load_records(out_dir, "backprojection_law")
    if df.empty:
        return
    g = df.groupby("rho")[["bp_frac_mean", "bp_frac_pred"]].mean()
    plt.figure()
    plt.plot(g.index, g["bp_frac_pred"], "-", label="Beta law 1 - rho")
    plt.plot(g.index, g["bp_frac_mean"], "s", label="measured")
    plt.xlabel("rho = m/N")
    plt.ylabel("residual energy fraction")
    plt.title("Result 3 back-projection control")
    plt.legend()
    _save(fig_dir, "backprojection_law")


def _plot_cumulants(out_dir: Path, fig_dir: Path):
    df = _load(out_dir, "cumulant_margin")
    if df.empty:
        return
    g = _high(df).groupby("rho")[["mean_cumulant_dist", "delta_q10", "delta_min"]].mean()
    plt.figure()
    plt.plot(g.index, g["mean_cumulant_dist"], "o-", label="||T(x~) - T(x)||")
    plt.plot(g.index, g["delta_q10"], "--", label="class margin (q10)")
    plt.plot(g.index, g["delta_min"], ":", label="class margin (min)")
    plt.xlabel("rho = m/N")
    plt.ylabel("cumulant distance")
    plt.title(f"Layer V margin (SNR >= {HIGH_SNR_DB} dB)")
    plt.legend()
    _save(fig_dir, "cumulant_margin")


def _plot_kernel_geometry(out_dir: Path, fig_dir: Path):
    df = load_records(out_dir, "kernel_geometry")
    if df.empty:
        return
    g = _high(df).groupby(["arm", "rho"])["zeta_perp"].mean().unstack(0)
    plt.figure()
    for arm in g.columns:
        plt.plot(g.index, g[arm], "o-", label=arm)
    plt.axhline(0.0, color="gray", lw=0.8, ls=":")
    plt.xlabel("rho = m/N")
    plt.ylabel("zeta_perp (shared structure, gain removed)")
    plt.title(f"Kernel geometry at matched distortion (SNR >= {HIGH_SNR_DB} dB)")
    plt.legend()
    _save(fig_dir, "kernel_geometry")


def _plot_retention_vs_diversity(out_dir: Path, fig_dir: Path):
    df = _load(out_dir, "label_nuisance_tradeoff")
    if df.empty or "arm" not in df:
        return
    plt.figure()
    for arm, sub in _high(df).groupby("arm"):
        g = sub.groupby("rho")[["view_diversity", "label_retention"]].mean()
        g = g.sort_values("view_diversity")
        plt.plot(g["view_diversity"], g["label_retention"], "o-", label=arm)
    plt.xlabel("view diversity V")
    plt.ylabel("label retention R")
    plt.title("Retention at matched diversity")
    plt.legend()
    _save(fig_dir, "retention_vs_diversity")


FIGURES = (_plot_kernel_geometry, _plot_retention_vs_diversity, _plot_tradeoff, _plot_se,
           _plot_compressibility, _plot_backprojection, _plot_cumulants)


def main() -> int:
    enable_fast_matmul()
    p = argparse.ArgumentParser(description="Report CoAug first-stage hypothesis verdicts.")
    p.add_argument("--out", type=Path, required=True, help="artifact directory")
    args = p.parse_args()
    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    verdicts = build_verdicts(args.out)
    verdicts.to_csv(args.out / "verdicts.csv", index=False)
    for fn in FIGURES:
        fn(args.out, fig_dir)

    print(verdicts.to_string(index=False), flush=True)
    print(f"\nSTAGE-1 GO/NO-GO: {'GO' if is_go(verdicts) else 'NO-GO'}", flush=True)
    print("(informational rows are controls and do not vote)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
