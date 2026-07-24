#!/usr/bin/env python3
"""Aggregate hypothesis artifacts into accept/reject verdicts and figures."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rf.analysis import build_verdicts
from rf.hypotheses._artifacts import load_records


def _plot_se(out_dir: Path, fig_dir: Path):
    df = load_records(out_dir, "se_calibration")
    if df.empty:
        return
    g = df.groupby("rho")[["realized_snr", "se_snr"]].mean().reset_index()
    plt.figure()
    plt.plot(g["rho"], g["se_snr"], "o-", label="SE prediction")
    plt.plot(g["rho"], g["realized_snr"], "s-", label="realized OAMP")
    plt.xlabel("rho = m/N")
    plt.ylabel("recovery SNR (dB)")
    plt.legend()
    plt.title("Layer IV calibration")
    plt.savefig(fig_dir / "se_calibration.png", dpi=120, bbox_inches="tight")
    plt.close()


def main() -> int:
    p = argparse.ArgumentParser(description="Report CoAug first-stage hypothesis verdicts.")
    p.add_argument("--out", type=Path, required=True, help="artifact directory")
    args = p.parse_args()
    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    verdicts = build_verdicts(args.out)
    verdicts.to_csv(args.out / "verdicts.csv", index=False)
    _plot_se(args.out, fig_dir)

    print(verdicts.to_string(index=False), flush=True)
    go = (verdicts["verdict"] == "accept").all()
    print(f"\nSTAGE-1 GO/NO-GO: {'GO' if go else 'NO-GO'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
