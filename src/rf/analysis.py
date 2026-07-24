import pandas as pd

# per-hypothesis accept/reject verdicts over aggregated seed artifacts

from rf.hypotheses._artifacts import load_records


def _verdict(name, ok, detail):
    return {"mechanism": name, "verdict": "accept" if ok else "reject", "detail": detail}


def operator_isometry(df):
    ok = df["gram_max_err"].max() < 1e-3 and df["sv_cv"].max() < 1e-2
    return _verdict("operator_isometry", ok, f"gram_err={df['gram_max_err'].max():.2e} sv_cv={df['sv_cv'].max():.2e}")


def nuisance_equivariance(df):
    gap = max(df["shift_gap"].mean(), df["cfo_gap"].mean())
    return _verdict("nuisance_equivariance", gap < 1.0, f"max_mean_gap={gap:.3f} dB")


def gabor_compressibility(df):
    g = df.groupby("mod")[["k90_gabor", "k90_dft"]].mean()
    ok = (g["k90_gabor"] < g["k90_dft"]).mean() > 0.5
    return _verdict("gabor_compressibility", ok, f"gabor<dft in {(g['k90_gabor']<g['k90_dft']).mean()*100:.0f}% of mods")


def se_calibration(df):
    hi = df[df["rho"] >= 0.7]
    ok = (hi["gap"] >= -1.0).all() and hi["gap"].mean() < 8.0
    return _verdict("se_calibration", ok, f"mean_gap(rho>=0.7)={hi['gap'].mean():.2f} dB")


def structure_vs_unstructured(df):
    ok = (df["var_ratio"] < 1.0).mean() > 0.5
    beta_ok = (df["bp_frac_mean"] - df["bp_frac_pred"]).abs().max() < 0.05
    return _verdict("structure_vs_unstructured", ok and beta_ok,
                    f"var_ratio_median={df['var_ratio'].median():.2f} beta_ok={beta_ok}")


def cumulant_margin(df):
    dmin = df["delta_min"].dropna()
    ref = dmin.mean() if len(dmin) else float("nan")
    hi = df[df["rho"] >= 0.7]["mean_cumulant_dist"].mean()
    return _verdict("cumulant_margin", hi < ref, f"dist(rho>=0.7)={hi:.3f} vs delta_min={ref:.3f}")


def admissible_band(df):
    row = df.iloc[0]
    return _verdict("admissible_band", bool(row["nonempty"]),
                    f"band=[{row['lo']:.3f},{row['hi']:.3f}] width={row['width']:.3f}")


VERDICTS = {
    "operator_isometry": operator_isometry,
    "nuisance_equivariance": nuisance_equivariance,
    "gabor_compressibility": gabor_compressibility,
    "se_calibration": se_calibration,
    "structure_vs_unstructured": structure_vs_unstructured,
    "cumulant_margin": cumulant_margin,
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
