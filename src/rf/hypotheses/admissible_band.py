import torch

from dsp.state_evolution import calibration_snr
from rf.hypotheses._artifacts import normalize_power, snr_strata
from rf.signal_model import DEFAULT_DICTIONARY, build_dictionary, admissible_band as assemble_band
from rf.signal_model import k_eff, rho_dt, statistical_dimension_l1

# Layer V Result 4: synthesize the band per SNR stratum from the measured sparsity and the stratum
# noise level. rho_dt from the statistical dimension, rho_label from the SE label floor, rho_max
# from the non-degeneracy ceiling. Accept: the band is non-empty at usable SNR.

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
    # never reaching the floor means no rho is degenerate, so the ceiling is the top of the grid;
    # returning the bottom would report an inverted band
    return GRID[-1]


def run(frames, meta, ratios, seed, device, dictionary=DEFAULT_DICTIONARY,
        measurement_snr=None) -> list[dict]:
    n = frames.shape[-1]
    frame = build_dictionary(dictionary, n, device)
    x = normalize_power(frames)
    eps = k_eff(n) / frame.n_atoms
    sigma = 0.0 if measurement_snr is None else 10.0 ** (-measurement_snr / 20.0)
    records = []
    for snr_db, rows in snr_strata(meta, x.shape[0]).items():
        dt = statistical_dimension_l1(k_eff(n) / n)
        label = _invert(n, eps, sigma, frame.gamma, LABEL_FLOOR_DB, ascending=True)
        ceil_rho = _invert(n, eps, sigma, frame.gamma, DEGENERACY_CEIL_DB, ascending=False)
        band = assemble_band(dt, label, ceil_rho)
        records.append({
            "dictionary": dictionary,
            "snr": snr_db,
            "measurement_snr": measurement_snr if measurement_snr is not None else float("inf"),
            "sigma": sigma,
            "eps": eps,
            "eps_source": "prop1",
            "rho_dt": dt,
            "rho_dt_pred": rho_dt(n),
            "rho_label": label,
            "rho_max": ceil_rho,
            "lo": band["lo"],
            "hi": band["hi"],
            "width": band["width"],
            "nonempty": band["nonempty"],
        })
    return records
