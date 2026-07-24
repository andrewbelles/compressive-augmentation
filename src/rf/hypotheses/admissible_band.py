from dsp.state_evolution import calibration_snr
from rf.signal_model import k_eff, rho_dt, admissible_band as assemble_band

# Layer V Result 4: synthesize the training-free admissible band for rml2018 and emit go/no-go.
# rho_dt from the statistical dimension; rho_label from the SE label floor; rho_max from the
# non-degeneracy ceiling. Accept: the band is non-empty.

GAMMA = 2
LABEL_FLOOR_DB = 12.0
DEGENERACY_CEIL_DB = 30.0


def _invert(n, eps, floor, ascending) -> float:
    grid = [i / 100 for i in range(5, 100, 5)]
    for r in grid:
        snr = calibration_snr(r, 0.0, eps, GAMMA, n)
        if ascending and snr >= floor:
            return r
        if not ascending and snr > floor:
            return r
    return grid[-1] if ascending else grid[0]


def run(frames, meta, ratios, seed, device) -> list[dict]:
    n = frames.shape[-1] if frames is not None else 1024
    eps = k_eff(n) / (GAMMA * n)
    dt = rho_dt(n, GAMMA)
    label = _invert(n, eps, LABEL_FLOOR_DB, ascending=True)
    ceil_rho = _invert(n, eps, DEGENERACY_CEIL_DB, ascending=False)
    band = assemble_band(dt, label, ceil_rho)
    return [{
        "rho_dt": dt,
        "rho_label": label,
        "rho_max": ceil_rho,
        "lo": band["lo"],
        "hi": band["hi"],
        "width": band["width"],
        "nonempty": band["nonempty"],
    }]
