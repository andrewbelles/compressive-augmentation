import torch

from dsp.recovery import sparse_code
from dsp.state_evolution import calibration_snr
from rf.hypotheses._artifacts import noise_sigma, normalize_power, snr_strata
from rf.signal_model import DEFAULT_DICTIONARY, build_dictionary, admissible_band as assemble_band
from rf.signal_model import k_eff, rho_dt, statistical_dimension_l1

# Layer V Result 4: synthesize the band per SNR stratum from the measured sparsity and the stratum
# noise level. rho_dt from the statistical dimension, rho_label from the SE label floor, rho_max
# from the non-degeneracy ceiling. Accept: the band is non-empty at usable SNR.

LAM = 0.05
LABEL_FLOOR_DB = 12.0
DEGENERACY_CEIL_DB = 30.0
GRID = [i / 100 for i in range(5, 100, 5)]


def _invert(n, eps, sigma, gamma, floor, ascending) -> float:
    for r in GRID:
        snr = calibration_snr(r, sigma, eps, gamma, n)
        if ascending and snr >= floor:
            return r
        if not ascending and snr > floor:
            return r
    return GRID[-1] if ascending else GRID[0]


def run(frames, meta, ratios, seed, device, dictionary=DEFAULT_DICTIONARY) -> list[dict]:
    n = frames.shape[-1]
    frame = build_dictionary(dictionary, n, device)
    x = normalize_power(frames)
    records = []
    for snr_db, rows in snr_strata(meta, x.shape[0]).items():
        sel = x[torch.tensor(rows, device=device)]
        sigma = noise_sigma(snr_db)
        coeffs, _ = sparse_code(sel, frame, LAM)
        active = (coeffs.abs() > 1e-6).float().sum(-1).mean().item()
        eps = max(active / frame.n_atoms, 1e-6)
        dt = frame.gamma * statistical_dimension_l1(eps)
        label = _invert(n, eps, sigma, frame.gamma, LABEL_FLOOR_DB, ascending=True)
        ceil_rho = _invert(n, eps, sigma, frame.gamma, DEGENERACY_CEIL_DB, ascending=False)
        band = assemble_band(dt, label, ceil_rho)
        records.append({
            "dictionary": dictionary,
            "snr": snr_db,
            "sigma": sigma,
            "eps_measured": eps,
            "eps_pred": k_eff(n) / frame.n_atoms,
            "rho_dt": dt,
            "rho_dt_pred": rho_dt(n, int(frame.gamma)),
            "rho_label": label,
            "rho_max": ceil_rho,
            "lo": band["lo"],
            "hi": band["hi"],
            "width": band["width"],
            "nonempty": band["nonempty"],
        })
    return records
