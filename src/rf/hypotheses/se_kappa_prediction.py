import torch

from dsp.operators import measure, measurement_noise, random_convolution
from dsp.recovery import oamp, reconstruct
from dsp.state_evolution import optimal_kappa
from rf.hypotheses._artifacts import normalize_power, snr_strata
from rf.signal_model import DEFAULT_DICTIONARY, build_dictionary, dither_sigma, frame_sparsity

# Layer IV Result 2 falsified through the threshold rather than through an SNR value: SE must
# predict which threshold is best, not merely a number to compare against. Accept: the predicted
# kappa is the measured argmax and costs little regret.

SWEEP = (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6)


def _snr(x, xhat):
    num = x.abs().pow(2).sum(-1)
    den = (x - xhat).abs().pow(2).sum(-1).clamp_min(1e-12)
    return (10 * torch.log10(num / den)).mean().item()


def run(frames, meta, ratios, seed, device, dictionary=DEFAULT_DICTIONARY,
        measurement_snr=None) -> list[dict]:
    n = frames.shape[-1]
    frame = build_dictionary(dictionary, n, device)
    x = normalize_power(frames)
    eps = frame_sparsity(frame, n)
    sigma = dither_sigma(measurement_snr)
    records = []
    for snr_db, rows in snr_strata(meta, x.shape[0]).items():
        sel = x[torch.tensor(rows, device=device)]
        for rho in ratios:
            m = max(1, round(rho * n))
            gen = torch.Generator(device=device).manual_seed(seed)
            op = random_convolution(n, m, gen, device=device)
            clean = measure(sel, op)
            y = clean + measurement_noise(clean, measurement_snr, op, gen)
            pred = optimal_kappa(rho, sigma, eps, frame.gamma, n)
            scored = [(_snr(sel, reconstruct(oamp(y, op, frame, kappa=k), frame)), k) for k in SWEEP]
            best_snr, best_k = max(scored)
            at_pred = _snr(sel, reconstruct(oamp(y, op, frame, kappa=pred), frame))
            records.append({
                "dictionary": dictionary,
                "snr": snr_db,
                "measurement_snr": measurement_snr if measurement_snr is not None else float("inf"),
                "rho": rho,
                "m": m,
                "eps": eps,
                "kappa_pred": pred,
                "kappa_meas": best_k,
                "kappa_err": abs(pred - best_k),
                "snr_at_pred": at_pred,
                "snr_at_best": best_snr,
                "regret": best_snr - at_pred,
            })
    return records
