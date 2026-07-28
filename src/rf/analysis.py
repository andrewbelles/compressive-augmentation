import numpy as np
import pandas as pd

# per-hypothesis verdicts over aggregated seed artifacts, restricted to the signal regime

from rf.hypotheses import INFORMATIONAL
from rf.hypotheses._artifacts import load_records

HIGH_SNR_DB = 10
BAND_RHO = 0.7


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


def compressibility_budget(df):
    """Prop 1: k99 matches N(1+beta)/kappa in the matched frame and misses in unmatched ones."""
    hi = _high(df).copy()
    if "k99_over_k_eff" not in hi:
        hi["k99_over_k_eff"] = hi["k99"] / hi["k_eff_pred"]
    g = hi.groupby("dictionary")["k99_over_k_eff"].mean()
    matched = g.get("gabor_symbol", float("nan"))
    others = g.drop(labels=["gabor_symbol"], errors="ignore")
    # pre-registered: the matched frame lands, and at least one unmatched frame misses
    hit = 0.9 <= matched <= 1.1
    discriminates = bool(((others < 0.7) | (others > 1.3)).any())
    detail = " ".join(f"{k}={v:.2f}" for k, v in g.items())
    return _verdict("compressibility_budget", hit and discriminates,
                    f"k99/k_eff: {detail} (matched_ok={hit} discriminates={discriminates})")


def operator_equivariance(df):
    """Prop 3 exact identities, with SRHT as the null that must fail."""
    conv = bool(df["conv_exact"].all())
    srht_fails = not bool(df["srht_exact"].any())
    return _verdict("operator_equivariance", conv and srht_fails,
                    f"conv_timing={df['conv_timing_err'].max():.2e} "
                    f"conv_cfo={df['conv_cfo_ongrid_err'].max():.2e} "
                    f"srht_timing={df['srht_timing_err'].min():.2e} (null fails={srht_fails})")


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
                    f"eps={hi['eps'].mean():.4f} ({hi['eps_source'].iloc[0]}) "
                    f"kappa={hi['kappa'].mean():.2f}")


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


def kernel_geometry(df):
    """H0: at matched distortion the CS kernel is isotropic. Both contrasts must exclude 0."""
    cs = _high(df)
    cs = cs[cs["arm"] == "cs"]
    vs_awgn = bool((cs["zeta_perp_vs_awgn_lo"] > 0).all())
    vs_bp = bool((cs["zeta_perp_vs_bp_lo"] > 0).all())
    arms = _high(df).groupby("arm")["zeta_perp"].mean()
    detail = " ".join(f"{k}={v:.3f}" for k, v in arms.items())
    return _verdict("kernel_geometry", vs_awgn and vs_bp,
                    f"zeta_perp: {detail} (excludes 0 vs awgn={vs_awgn} vs bp={vs_bp})")


def se_kappa_prediction(df):
    hi = _high(df)
    err, regret = hi["kappa_err"].mean(), hi["regret"].mean()
    return _verdict("se_kappa_prediction", err <= 0.2 and regret <= 0.5,
                    f"|kappa_pred-kappa_meas|={err:.2f} regret={regret:.2f} dB "
                    f"(pred={hi['kappa_pred'].mean():.2f} meas={hi['kappa_meas'].mean():.2f})")


def se_knee(df):
    hi = _high(df)
    finite = hi[hi["measurement_snr"] < float("inf")]
    ref = finite if not finite.empty else hi
    interior = bool(ref["knee_interior"].any())
    err = ref["knee_err"].mean()
    return _verdict("se_knee", interior and err <= 0.1,
                    f"measured={ref['knee_measured'].mean():.3f} "
                    f"predicted={ref['knee_predicted'].mean():.3f} err={err:.3f} "
                    f"interior={interior}")


def label_nuisance_tradeoff(df):
    """Compare retention at matched view diversity: rho is a parameter, not a strength."""
    hi = _high(df)
    if "arm" not in hi:
        return _verdict("label_nuisance_tradeoff", False, "artifacts predate the awgn arm")
    grid = np.linspace(0.05, 0.6, 12)
    curves = {}
    for arm, sub in hi.groupby("arm"):
        g = sub.groupby("rho")[["view_diversity", "label_retention"]].mean().sort_values("view_diversity")
        curves[arm] = np.interp(grid, g["view_diversity"], g["label_retention"],
                                left=np.nan, right=np.nan)
    if "cs" not in curves or "awgn" not in curves:
        return _verdict("label_nuisance_tradeoff", False, "missing an arm")
    both = ~(np.isnan(curves["cs"]) | np.isnan(curves["awgn"]))
    if not both.any():
        return _verdict("label_nuisance_tradeoff", False, "no overlapping diversity range")
    margin = float(np.mean(curves["cs"][both] - curves["awgn"][both]))
    return _verdict("label_nuisance_tradeoff", margin > 0.0,
                    f"mean R(V) margin over awgn = {margin:+.3f} across {int(both.sum())} "
                    f"matched-diversity points")


def admissible_band(df):
    hi = _high(df)
    eps = hi["eps"] if "eps" in hi else hi["eps_measured"]
    if hi["rho_dt"].mean() > 1.0:
        # rho is m/N and cannot exceed 1: the sparsity is past the DT threshold
        return _verdict("admissible_band", False,
                        f"infeasible: rho_dt={hi['rho_dt'].mean():.2f} > 1 at eps={eps.mean():.4f}")
    return _verdict("admissible_band", bool(hi.iloc[0]["nonempty"]),
                    f"band=[{hi['lo'].mean():.3f},{hi['hi'].mean():.3f}] "
                    f"width={hi['width'].mean():.3f} eps={eps.mean():.4f}")


VERDICTS = {
    "operator_isometry": operator_isometry,
    "operator_equivariance": operator_equivariance,
    "kernel_geometry": kernel_geometry,
    "backprojection_law": backprojection_law,
    "compressibility_budget": compressibility_budget,
    "dictionary_compressibility": dictionary_compressibility,
    "se_calibration": se_calibration,
    "se_kappa_prediction": se_kappa_prediction,
    "se_knee": se_knee,
    "operator_draw_variance": operator_draw_variance,
    "cumulant_margin": cumulant_margin,
    "label_nuisance_tradeoff": label_nuisance_tradeoff,
    "admissible_band": admissible_band,
}


# verdicts scored from another mechanism's artifacts rather than their own
ARTIFACT_SOURCE = {"compressibility_budget": "dictionary_compressibility"}


def build_verdicts(out_dir) -> pd.DataFrame:
    """Apply every hypothesis verdict over its aggregated artifacts."""
    rows = []
    for name, fn in VERDICTS.items():
        df = load_records(out_dir, ARTIFACT_SOURCE.get(name, name))
        if df.empty:
            rows.append({"mechanism": name, "verdict": "missing", "detail": "no artifacts"})
        else:
            rows.append(fn(df))
    return pd.DataFrame(rows)


GATES = ("kernel_geometry", "label_nuisance_tradeoff")


def is_go(verdicts: pd.DataFrame) -> bool:
    """GO needs both gates: the kernel must be a distinct object and must keep the label."""
    gates = verdicts[verdicts["mechanism"].isin(GATES)]
    return len(gates) == len(GATES) and bool((gates["verdict"] == "accept").all())
