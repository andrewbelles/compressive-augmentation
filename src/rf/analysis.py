import pandas as pd

# per-hypothesis verdicts over aggregated seed artifacts, restricted to the signal regime

from rf.hypotheses import INFORMATIONAL
from rf.hypotheses._artifacts import load_records

HIGH_SNR_DB = 10
BAND_RHO = 0.7
RETENTION_FLOOR = 0.7
DIVERSITY_FLOOR = 0.1


def _verdict(name, ok, detail):
    kind = "informational" if name in INFORMATIONAL else ("accept" if ok else "reject")
    return {"mechanism": name, "verdict": kind, "detail": detail}


def _high(df):
    hi = df[df["snr"] >= HIGH_SNR_DB] if "snr" in df else df
    return hi if not hi.empty else df


def operator_isometry(df):
    rank_ok = (df["n_nonzero_sv"] == df["n_expected_sv"]).all()
    ok = df["gram_max_err"].max() < 1e-3 and df["sv_cv"].max() < 1e-2 and rank_ok
    return _verdict("operator_isometry", ok,
                    f"gram_err={df['gram_max_err'].max():.2e} sv_cv={df['sv_cv'].max():.2e} "
                    f"rank_ok={rank_ok}")


def backprojection_law(df):
    err = (df["bp_frac_mean"] - df["bp_frac_pred"]).abs().max()
    return _verdict("backprojection_law", err < 0.02, f"max_mean_err={err:.4f} (Result 3 control)")


def nuisance_equivariance(df):
    hi = _high(df)
    gaps = [c for c in hi.columns if c.endswith("_gap")]
    worst = hi[gaps].mean().max()
    name = hi[gaps].mean().idxmax()
    # a recovery floor would also produce small gaps, so require the base to be off the floor
    live = hi["snr_base"].mean() > 3.0
    return _verdict("nuisance_equivariance", worst < 1.0 and live,
                    f"worst={name}={worst:.3f} dB snr_base={hi['snr_base'].mean():.1f} dB")


def dictionary_compressibility(df):
    hi = _high(df)
    g = hi.groupby("dictionary")[["class_scatter_ratio", "k90_over_d", "circular_span_ratio"]].mean()
    best = g["class_scatter_ratio"].idxmax()
    dft_sep = g.loc["dft", "class_scatter_ratio"]
    ok = best != "dft" and g.loc[best, "class_scatter_ratio"] > dft_sep
    return _verdict("dictionary_compressibility", bool(ok),
                    f"best={best} sep={g.loc[best, 'class_scatter_ratio']:.3f} vs dft={dft_sep:.3f} "
                    f"dft_span={g.loc['dft', 'circular_span_ratio']:.2f}")


def se_calibration(df):
    hi = _high(df)
    band = hi[hi["rho"] >= BAND_RHO]
    gap = band["gap"].abs().mean()
    return _verdict("se_calibration", gap < 8.0,
                    f"mean|gap|(rho>={BAND_RHO},snr>={HIGH_SNR_DB})={gap:.2f} dB "
                    f"eps_meas={hi['eps_measured'].mean():.4f} eps_pred={hi['eps_pred'].mean():.4f}")


def operator_draw_variance(df):
    hi = _high(df)
    excludes = (hi["var_ratio_hi"] < 1.0).mean()
    return _verdict("operator_draw_variance", bool(excludes > 0.5),
                    f"median_ratio={hi['var_ratio'].median():.3f} "
                    f"CI_excludes_1 in {excludes * 100:.0f}% of cells")


def cumulant_margin(df):
    hi = _high(df)
    band = hi[hi["rho"] >= BAND_RHO]
    dist, margin = band["mean_cumulant_dist"].mean(), band["delta_q10"].mean()
    return _verdict("cumulant_margin", dist < margin,
                    f"dist={dist:.3f} vs delta_q10={margin:.3f} (min={band['delta_min'].mean():.3f})")


def label_nuisance_tradeoff(df):
    hi = _high(df)
    g = hi.groupby("rho")[["label_retention", "view_diversity", "nuisance_collapse"]].mean()
    usable = g[(g["label_retention"] >= RETENTION_FLOOR) & (g["view_diversity"] >= DIVERSITY_FLOOR)]
    detail = "no rho keeps the label while staying diverse"
    if not usable.empty:
        detail = (f"band=[{usable.index.min():.2f},{usable.index.max():.2f}] "
                  f"retention={usable['label_retention'].max():.2f} "
                  f"diversity={usable['view_diversity'].max():.3f}")
    return _verdict("label_nuisance_tradeoff", not usable.empty, detail)


def admissible_band(df):
    hi = _high(df)
    if hi["rho_dt"].mean() > 1.0:
        # rho is m/N and cannot exceed 1: the measured sparsity is past the DT threshold
        return _verdict("admissible_band", False,
                        f"infeasible: rho_dt={hi['rho_dt'].mean():.2f} > 1 at "
                        f"eps={hi['eps_measured'].mean():.4f}")
    return _verdict("admissible_band", bool(hi.iloc[0]["nonempty"]),
                    f"band=[{hi['lo'].mean():.3f},{hi['hi'].mean():.3f}] "
                    f"width={hi['width'].mean():.3f} eps={hi['eps_measured'].mean():.4f}")


VERDICTS = {
    "operator_isometry": operator_isometry,
    "backprojection_law": backprojection_law,
    "nuisance_equivariance": nuisance_equivariance,
    "dictionary_compressibility": dictionary_compressibility,
    "se_calibration": se_calibration,
    "operator_draw_variance": operator_draw_variance,
    "cumulant_margin": cumulant_margin,
    "label_nuisance_tradeoff": label_nuisance_tradeoff,
    "admissible_band": admissible_band,
}


def build_verdicts(out_dir) -> pd.DataFrame:
    """Apply every hypothesis verdict over its aggregated artifacts."""
    rows = []
    for name, fn in VERDICTS.items():
        df = load_records(out_dir, name)
        if df.empty:
            rows.append({"mechanism": name, "verdict": "missing", "detail": "no artifacts"})
        else:
            rows.append(fn(df))
    return pd.DataFrame(rows)


def is_go(verdicts: pd.DataFrame) -> bool:
    """GO requires every measuring mechanism to accept; controls do not vote."""
    scored = verdicts[~verdicts["mechanism"].isin(INFORMATIONAL)]
    return bool((scored["verdict"] == "accept").all()) and not scored.empty
