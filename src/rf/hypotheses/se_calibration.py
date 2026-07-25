import torch

from dsp.operators import random_convolution, measure
from dsp.recovery import oamp, reconstruct, sparse_code
from dsp.state_evolution import calibration_snr, se_fixed_point
from rf.hypotheses._artifacts import noise_sigma, normalize_power, snr_strata
from rf.signal_model import DEFAULT_DICTIONARY, build_dictionary, k_eff

# Layer IV Result 2: state evolution predicts realized OAMP recovery SNR of the CS pipeline.
# Accept: SE tracks realized SNR per SNR stratum at the measured sparsity, not the predicted one.

LAM = 0.05


def _snr(x, xhat):
    num = x.abs().pow(2).sum(-1)
    den = (x - xhat).abs().pow(2).sum(-1).clamp_min(1e-12)
    return (10 * torch.log10(num / den)).mean().item()


def run(frames, meta, ratios, seed, device, dictionary=DEFAULT_DICTIONARY) -> list[dict]:
    n = frames.shape[-1]
    frame = build_dictionary(dictionary, n, device)
    x = normalize_power(frames)
    eps_pred = k_eff(n) / frame.n_atoms
    records = []
    for snr_db, rows in snr_strata(meta, x.shape[0]).items():
        sel = x[torch.tensor(rows, device=device)]
        sigma = noise_sigma(snr_db)
        coeffs, _ = sparse_code(sel, frame, LAM)
        active = (coeffs.abs() > 1e-6).float().sum(-1).mean().item()
        eps = max(active / frame.n_atoms, 1e-6)
        for rho in ratios:
            m = max(1, round(rho * n))
            gen = torch.Generator(device=device).manual_seed(seed)
            op = random_convolution(n, m, gen, device=device)
            realized = _snr(sel, reconstruct(oamp(measure(sel, op), op, frame), frame))
            fp = se_fixed_point(rho, sigma, eps, frame.gamma, n)
            se = calibration_snr(rho, sigma, eps, frame.gamma, n)
            records.append({
                "dictionary": dictionary,
                "snr": snr_db,
                "rho": rho,
                "m": m,
                "sigma": sigma,
                "eps_measured": eps,
                "eps_pred": eps_pred,
                "realized_snr": realized,
                "se_snr": se,
                "se_mse": fp["mse"],  # exposes the 1e-12 clamp sitting behind se_snr
                "gap": se - realized,
            })
    return records
