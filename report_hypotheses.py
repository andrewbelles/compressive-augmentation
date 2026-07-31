#!/usr/bin/env python3
"""Aggregate hypothesis artifacts into accept/reject verdicts and figures."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from common.utils import enable_fast_matmul
from rf.analysis import ARTIFACT_SOURCE, HIGH_SNR_DB, MIN_BASE_SEP, build_verdicts, is_go
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


def _levels(df):
    """One series per measurement-noise level; sigma sets the ceiling, so curves must not be pooled."""
    if "measurement_snr" not in df:
        return [("", df)]
    out = []
    for lv, sub in df.groupby("measurement_snr"):
        out.append(("noiseless" if lv == float("inf") else f"{lv:g} dB", sub))
    return out


def _plot_tradeoff(out_dir: Path, fig_dir: Path):
    df = _load(out_dir, "label_nuisance_tradeoff")
    if df.empty:
        return
    df = df[df["arm"] == "cs"] if "arm" in df else df
    plt.figure()
    for label, sub in _levels(_high(df)):
        g = sub.groupby("rho")[["label_retention", "view_diversity", "nuisance_collapse"]].mean()
        line, = plt.plot(g.index, g["label_retention"], "o-", label=f"retention {label}".strip())
        c = line.get_color()
        plt.plot(g.index, g["view_diversity"], "s--", color=c, label=f"diversity {label}".strip())
        plt.plot(g.index, g["nuisance_collapse"], "^:", color=c, label=f"collapse {label}".strip())
    plt.axhline(1.0, color="gray", lw=0.8, ls=":")
    plt.xlabel("rho = m/N")
    plt.ylabel("ratio")
    plt.title(f"Label retention vs view diversity (SNR >= {HIGH_SNR_DB} dB)")
    plt.legend(fontsize=6, ncol=2)
    _save(fig_dir, "label_nuisance_tradeoff")


def _plot_se(out_dir: Path, fig_dir: Path):
    df = _load(out_dir, "se_calibration")
    if df.empty:
        return
    plt.figure()
    for label, sub in _levels(_high(df)):
        g = sub.groupby("rho")[["realized_snr", "se_snr"]].mean()
        line, = plt.plot(g.index, g["se_snr"], "o--", label=f"SE {label}".strip())
        plt.plot(g.index, g["realized_snr"], "s-", color=line.get_color(),
                 label=f"realized {label}".strip())
    plt.xlabel("rho = m/N")
    plt.ylabel("recovery SNR (dB)")
    plt.title(f"Layer IV calibration (SNR >= {HIGH_SNR_DB} dB)")
    plt.legend(fontsize=7)
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
    hi = _high(df)
    plt.figure()
    for label, sub in _levels(hi):
        g = sub.groupby("rho")["mean_cumulant_dist"].mean()
        plt.plot(g.index, g.values, "o-", label=f"||T(x~) - T(x)|| {label}".strip())
    g = hi.groupby("rho")[["delta_q10", "delta_min"]].mean()
    plt.plot(g.index, g["delta_q10"], "k--", label="class margin (q10)")
    plt.plot(g.index, g["delta_min"], "k:", label="class margin (min)")
    plt.xlabel("rho = m/N")
    plt.ylabel("cumulant distance")
    plt.title(f"Layer V margin (SNR >= {HIGH_SNR_DB} dB)")
    plt.legend(fontsize=7)
    _save(fig_dir, "cumulant_margin")


def _plot_kernel_geometry(out_dir: Path, fig_dir: Path):
    df = load_records(out_dir, "kernel_geometry")
    if df.empty:
        return
    plt.figure()
    for label, lev in _levels(_high(df)):
        g = lev.groupby(["arm", "rho"])["zeta_perp"].mean().unstack(0)
        for arm in g.columns:
            plt.plot(g.index, g[arm], "o-" if arm == "cs" else "s--",
                     label=f"{arm} {label}".strip())
    plt.axhline(0.0, color="gray", lw=0.8, ls=":")
    plt.xlabel("rho = m/N")
    plt.ylabel("zeta_perp (shared structure, gain removed)")
    plt.title(f"Kernel geometry at matched distortion (SNR >= {HIGH_SNR_DB} dB)")
    plt.legend(fontsize=6, ncol=2)
    _save(fig_dir, "kernel_geometry")


def _plot_retention_vs_diversity(out_dir: Path, fig_dir: Path):
    df = _load(out_dir, "label_nuisance_tradeoff")
    if df.empty or "arm" not in df:
        return
    plt.figure()
    for label, lev in _levels(_high(df)):
        for arm, style in (("cs", "o-"), ("awgn", "s--")):
            sub = lev[lev["arm"] == arm]
            if sub.empty:
                continue
            g = sub.groupby("rho")[["view_diversity", "label_retention"]].mean()
            g = g.sort_values("view_diversity")
            plt.plot(g["view_diversity"], g["label_retention"], style,
                     label=f"{arm} {label}".strip())
    plt.xlabel("view diversity V")
    plt.ylabel("label retention R")
    plt.title("Retention at matched diversity")
    plt.legend(fontsize=7)
    _save(fig_dir, "retention_vs_diversity")


# the operator and its three matched-distortion nulls, in fixed slot order so a series keeps its
# hue when a level drops out; validated for CVD separation on the light chart surface
CONTRACTION_ARMS = ("cs", "awgn", "backprojection", "dropout")
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SURFACE, MUTED, GRID, RULE = "#fcfcfb", "#898781", "#e1e0d9", "#52514e"


def _chrome(ax):
    """Recede the frame so the marks carry the chart."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _chrome_log(ax):
    """Log ticks on a three-decade range crowd the spine, so only the decades are drawn."""
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def _plot_class_contraction(out_dir: Path, fig_dir: Path):
    df = load_records(out_dir, "class_diameter")
    if df.empty or "class_alignment_between" not in df:
        return
    hi = _high(df).copy()
    # the contraction is the gap between the two terms: each alone carries the projection's
    # indifference to direction, and only their difference is zero for a direction-agnostic arm
    hi["delta"] = hi["class_alignment"] - hi["class_alignment_between"]
    levels = _levels(hi)
    # two panels rather than two y-scales: the quantities share only the rho axis, and a second
    # scale would put the curves' crossing wherever its limits were chosen
    fig, axes = plt.subplots(2, len(levels), figsize=(3.3 * len(levels) + 0.6, 4.4),
                             squeeze=False, sharex="col", sharey="row")
    fig.set_facecolor(SURFACE)
    for col, (label, sub) in enumerate(levels):
        top, bot = axes[0][col], axes[1][col]
        drawn_d, drawn_r = {}, {}
        for arm in CONTRACTION_ARMS:
            rows = sub[sub["arm"] == arm]
            if rows.empty:
                continue
            color = SERIES[CONTRACTION_ARMS.index(arm)]
            wide = 2.0 if arm == "cs" else 1.2
            g = rows.groupby("rho")["delta"].mean()
            top.plot(g.index, g.values, "-", color=color, lw=wide, marker="o", ms=3.5)
            drawn_d[arm] = (list(g.index), list(g.values))
            # retention against a near-zero base is two noise floors, so it is dropped not drawn
            keep = rows[rows["base_sep"] > MIN_BASE_SEP]
            r = keep.groupby("rho")["label_retention"].mean()
            r = r[r > 0]
            if r.empty:
                continue
            bot.plot(r.index, r.values, "-", color=color, lw=wide, marker="o", ms=3.5)
            drawn_r[arm] = (list(r.index), list(r.values))
        for ax, ref in ((top, 0.0), (bot, 1.0)):
            ax.axhline(ref, color=RULE, lw=0.9, ls=(0, (4, 3)))
            _chrome(ax)
        if drawn_r:
            _chrome_log(bot)
        # the window the two panels agree on, marked on both so it reads down the shared axis
        d, r = drawn_d.get("cs"), drawn_r.get("cs")
        if d and r:
            both = sorted({x for x, y in zip(*d) if y > 0.0} & {x for x, y in zip(*r) if y > 1.0})
            if both:
                for ax in (top, bot):
                    ax.axvspan(min(both), max(both), color="#2a78d6", alpha=0.07, lw=0)
        top.set_title(label or "pooled", fontsize=9, color=RULE)
        bot.set_xlabel("rho = m/N", fontsize=8, color=RULE)
    axes[0][0].set_ylabel("contraction\n(within - between)", fontsize=8, color=RULE)
    axes[1][0].set_ylabel("label retention", fontsize=8, color=RULE)
    # the series converge at both ends, so one legend carries identity where end labels would collide
    handles = [plt.Line2D([], [], color=SERIES[i], lw=2.0 if a == "cs" else 1.2, marker="o", ms=3.5)
               for i, a in enumerate(CONTRACTION_ARMS)]
    fig.legend(handles, CONTRACTION_ARMS, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=len(CONTRACTION_ARMS), frameon=False, fontsize=8, labelcolor=RULE,
               handlelength=1.6, columnspacing=1.8)
    fig.suptitle(f"Class contraction and label retention (channel SNR >= {HIGH_SNR_DB} dB)",
                 fontsize=10, color="#0b0b0b")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig_dir, "class_contraction")


FIGURES = (_plot_kernel_geometry, _plot_class_contraction, _plot_retention_vs_diversity,
           _plot_tradeoff, _plot_se, _plot_compressibility, _plot_backprojection, _plot_cumulants)


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
