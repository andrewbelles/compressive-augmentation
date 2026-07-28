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


def _by_noise(df):
    """Split on the measurement-noise axis; sigma sets the ceiling, so pooling averages
    physically different quantities exactly the way SNR pooling did."""
    if "measurement_snr" not in df:
        return [(float("inf"), df)]
    return [(lv, sub) for lv, sub in df.groupby("measurement_snr") if not sub.empty]


def _level(lv):
    return "noiseless" if lv == float("inf") else f"{lv:g}dB"


def _worst(df, fn):
    """Apply a per-level statistic and return the worst value plus a per-level summary."""
    vals = {lv: fn(sub) for lv, sub in _by_noise(df)}
    detail = " ".join(f"{_level(lv)}={v:.2f}" for lv, v in sorted(vals.items()))
    return max(vals.values()), detail


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
    worst, detail = _worst(band, lambda s: s["gap"].abs().mean())
    return _verdict("se_calibration", worst < 8.0,
                    f"mean|gap| by measurement SNR: {detail} dB (worst {worst:.2f}) "
                    f"eps={hi['eps'].mean():.4f} ({hi['eps_source'].iloc[0]})")


def operator_draw_variance(df):
    hi = _high(df)
    excludes = (hi["var_ratio_hi"] < 1.0).mean()
    return _verdict("operator_draw_variance", bool(excludes > 0.5),
                    f"median_ratio={hi['var_ratio'].median():.3f} "
                    f"CI_excludes_1 in {excludes * 100:.0f}% of cells")


def cumulant_margin(df):
    band = _high(df)
    band = band[band["rho"] >= BAND_RHO]
    # ratio below 1 means the distortion sits inside the class margin at that noise level
    worst, detail = _worst(band, lambda s: s["mean_cumulant_dist"].mean() / max(s["delta_q10"].mean(), 1e-12))
    return _verdict("cumulant_margin", worst < 1.0,
                    f"dist/delta_q10 by measurement SNR: {detail} (worst {worst:.2f}) "
                    f"delta_min={band['delta_min'].mean():.3f}")


def kernel_geometry(df):
    """H0: at matched distortion the CS kernel is isotropic. Both contrasts must exclude 0."""
    hi = _high(df)
    cs = hi[hi["arm"] == "cs"]
    # already per-cell, so requiring every row keeps each measurement-noise level on its own terms
    vs_awgn = bool((cs["zeta_perp_vs_awgn_lo"] > 0).all())
    vs_bp = bool((cs["zeta_perp_vs_bp_lo"] > 0).all())
    arms = hi.groupby("arm")["zeta_perp"].mean()
    by_level = " ".join(f"{_level(lv)}={sub[sub['arm'] == 'cs']['zeta_perp'].mean():.3f}"
                        for lv, sub in sorted(_by_noise(hi)))
    detail = " ".join(f"{k}={v:.3f}" for k, v in arms.items())
    return _verdict("kernel_geometry", vs_awgn and vs_bp,
                    f"zeta_perp: {detail}; cs by measurement SNR: {by_level} "
                    f"(excludes 0 vs awgn={vs_awgn} vs bp={vs_bp})")


def se_kappa_prediction(df):
    hi = _high(df)
    worst_err, err_detail = _worst(hi, lambda s: s["kappa_err"].mean())
    worst_regret, regret_detail = _worst(hi, lambda s: s["regret"].mean())
    return _verdict("se_kappa_prediction", worst_err <= 0.2 and worst_regret <= 0.5,
                    f"|kappa_pred-kappa_meas| by measurement SNR: {err_detail} (worst {worst_err:.2f}); "
                    f"regret: {regret_detail} dB (worst {worst_regret:.2f})")


def se_knee(df):
    hi = _high(df)
    # the interior maximum is what sigma buys, so it need only appear at some finite level, but
    # wherever it appears the location must be the predicted one
    interior = [(lv, sub) for lv, sub in _by_noise(hi)
                if lv < float("inf") and bool(sub["knee_interior"].any())]
    if not interior:
        _, detail = _worst(hi, lambda s: s["knee_measured"].mean())
        return _verdict("se_knee", False,
                        f"no interior maximum at any finite measurement SNR; "
                        f"argmax by level: {detail}")
    worst = max(sub["knee_err"].mean() for _, sub in interior)
    detail = " ".join(f"{_level(lv)}: meas={sub['knee_measured'].mean():.2f} "
                      f"pred={sub['knee_predicted'].mean():.2f}" for lv, sub in sorted(interior))
    return _verdict("se_knee", worst <= 0.1,
                    f"interior at {len(interior)} level(s) -- {detail} (worst err {worst:.3f})")


DIVERSITY_GRID = np.linspace(0.05, 0.6, 12)


def _retention_margin(sub):
    """Mean R(cs) - R(awgn) read at matched view diversity, or nan if the arms do not overlap."""
    curves = {}
    for arm, rows in sub.groupby("arm"):
        g = rows.groupby("rho")[["view_diversity", "label_retention"]].mean()
        g = g.sort_values("view_diversity")
        curves[arm] = np.interp(DIVERSITY_GRID, g["view_diversity"], g["label_retention"],
                                left=np.nan, right=np.nan)
    if "cs" not in curves or "awgn" not in curves:
        return float("nan"), 0
    both = ~(np.isnan(curves["cs"]) | np.isnan(curves["awgn"]))
    if not both.any():
        return float("nan"), 0
    return float(np.mean(curves["cs"][both] - curves["awgn"][both])), int(both.sum())


def label_nuisance_tradeoff(df):
    """Compare retention at matched view diversity: rho is a parameter, not a strength."""
    hi = _high(df)
    if "arm" not in hi:
        return _verdict("label_nuisance_tradeoff", False, "artifacts predate the awgn arm")
    # each measurement-noise level is its own R(V) curve; averaging points across levels
    # interpolates through a curve that describes no single operating condition
    per = {lv: _retention_margin(sub) for lv, sub in _by_noise(hi)}
    usable = {lv: m for lv, (m, npts) in per.items() if npts > 0 and m == m}
    if not usable:
        return _verdict("label_nuisance_tradeoff", False,
                        "no overlapping diversity range between the cs and awgn arms")
    detail = " ".join(f"{_level(lv)}={m:+.3f}" for lv, m in sorted(usable.items()))
    return _verdict("label_nuisance_tradeoff", min(usable.values()) > 0.0,
                    f"R(V) margin over awgn by measurement SNR: {detail} "
                    f"(worst {min(usable.values()):+.3f})")


def admissible_band(df):
    hi = _high(df)
    eps = hi["eps"] if "eps" in hi else hi["eps_measured"]
    if hi["rho_dt"].mean() > 1.0:
        # rho is m/N and cannot exceed 1: the sparsity is past the DT threshold
        return _verdict("admissible_band", False,
                        f"infeasible: rho_dt={hi['rho_dt'].mean():.2f} > 1 at eps={eps.mean():.4f}")
    # sigma moves both endpoints, so a single pooled band would describe no operating condition
    bands = " ".join(f"{_level(lv)}=[{sub['lo'].mean():.2f},{sub['hi'].mean():.2f}]"
                     for lv, sub in sorted(_by_noise(hi)))
    nonempty = bool(all(sub["nonempty"].all() for _, sub in _by_noise(hi)))
    return _verdict("admissible_band", nonempty,
                    f"band by measurement SNR: {bands} eps={eps.mean():.4f}")


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
ARTIFACT_SOURCE = {
    "compressibility_budget": "dictionary_compressibility",
    "label_nuisance_tradeoff": "kernel_geometry",
    "se_calibration": "se_knee",
    "cumulant_margin": "se_knee",
}


def _coverage(df) -> str:
    """Artifacts behind a verdict; a preempted task shrinks the pool with no other trace."""
    seeds = df["seed"].nunique()
    if "measurement_snr" not in df:
        return f"{seeds} seeds"
    return f"{seeds} seeds x {df['measurement_snr'].nunique()} levels"


def build_verdicts(out_dir) -> pd.DataFrame:
    """Apply every hypothesis verdict over its aggregated artifacts."""
    rows = []
    for name, fn in VERDICTS.items():
        df = load_records(out_dir, ARTIFACT_SOURCE.get(name, name))
        if df.empty:
            rows.append({"mechanism": name, "verdict": "missing", "detail": "no artifacts",
                         "coverage": ""})
            continue
        try:
            row = fn(df)
        except (ValueError, KeyError, IndexError) as err:
            # a gap in one mechanism's rows must not take down the other twelve verdicts
            row = _verdict(name, False, f"{type(err).__name__}: {err}")
            row["verdict"] = "incomplete"
        rows.append({**row, "coverage": _coverage(df)})
    return pd.DataFrame(rows)


GATES = ("kernel_geometry", "label_nuisance_tradeoff")


def is_go(verdicts: pd.DataFrame) -> bool:
    """GO needs both gates: the kernel must be a distinct object and must keep the label."""
    gates = verdicts[verdicts["mechanism"].isin(GATES)]
    return len(gates) == len(GATES) and bool((gates["verdict"] == "accept").all())
