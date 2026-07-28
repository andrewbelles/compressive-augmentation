import torch

from dsp.operators import measure, measurement_noise, random_convolution
from dsp.recovery import oamp, reconstruct
from dsp.state_evolution import calibration_snr, optimal_kappa
from rf.hypotheses._artifacts import normalize_power, snr_strata
from rf.signal_model import DEFAULT_DICTIONARY, build_dictionary, k_eff

# Layer II says a compressible prior turns the Donoho-Tanner cliff into a knee, the rho where
# dSNR/drho peaks. With sigma = 0 recovery runs away as rho -> 1 and the maximum sits at the grid
# edge, so an interior argmax exists only once measurement noise caps the curve.
# Accept: the measured knee is interior and lands where SE predicts.


def _snr(x, xhat):
    num = x.abs().pow(2).sum(-1)
    den = (x - xhat).abs().pow(2).sum(-1).clamp_min(1e-12)
    return (10 * torch.log10(num / den)).mean().item()


def _knee(rhos, values):
    """Midpoint of the steepest segment, and whether it is interior to the grid."""
    slopes = [(values[i] - values[i - 1]) / (rhos[i] - rhos[i - 1]) for i in range(1, len(values))]
    mids = [0.5 * (rhos[i] + rhos[i - 1]) for i in range(1, len(rhos))]
    j = max(range(len(slopes)), key=lambda i: slopes[i])
    return mids[j], slopes[j], 0 < j < len(slopes) - 1


def run(frames, meta, ratios, seed, device, dictionary=DEFAULT_DICTIONARY,
        measurement_snr=None) -> list[dict]:
    n = frames.shape[-1]
    frame = build_dictionary(dictionary, n, device)
    x = normalize_power(frames)
    eps = k_eff(n) / frame.n_atoms
    sigma = 0.0 if measurement_snr is None else 10.0 ** (-measurement_snr / 20.0)
    rhos = sorted(ratios)
    records = []
    for snr_db, rows in snr_strata(meta, x.shape[0]).items():
        sel = x[torch.tensor(rows, device=device)]
        realized, predicted = [], []
        for rho in rhos:
            m = max(1, round(rho * n))
            gen = torch.Generator(device=device).manual_seed(seed)
            op = random_convolution(n, m, gen, device=device)
            clean = measure(sel, op)
            y = clean + measurement_noise(clean, measurement_snr, op, gen)
            kappa = optimal_kappa(rho, sigma, eps, frame.gamma, n)
            realized.append(_snr(sel, reconstruct(oamp(y, op, frame, kappa=kappa), frame)))
            predicted.append(calibration_snr(rho, sigma, eps, frame.gamma, n, kappa=kappa))
        knee_m, slope_m, interior_m = _knee(rhos, realized)
        knee_p, slope_p, interior_p = _knee(rhos, predicted)
        for rho, r, p in zip(rhos, realized, predicted):
            records.append({
                "dictionary": dictionary,
                "snr": snr_db,
                "measurement_snr": measurement_snr if measurement_snr is not None else float("inf"),
                "rho": rho,
                "realized_snr": r,
                "se_snr": p,
                "knee_measured": knee_m,
                "knee_predicted": knee_p,
                "knee_err": abs(knee_m - knee_p),
                "knee_interior": interior_m,
                "max_slope": slope_m,
            })
    return records
