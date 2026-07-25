import math

from dsp.frames import Frame, dft_frame, gabor_frame

# rml2018 generation constants and the training-free admissible-band endpoints (Layer II and V)
# Constants follow O'Shea, Roy, Clancy 2018 (JSTSP 12(1)) section II.

FRAME_LEN = 1024
N_CLASSES = 24
SAMPLES_PER_SYMBOL = 8
RRC_ROLLOFF = 0.35


def k_eff(n: int = FRAME_LEN, beta: float = RRC_ROLLOFF, kappa: int = SAMPLES_PER_SYMBOL) -> float:
    """Compressibility budget k_eff = n (1 + beta) / kappa (Eq. 2)."""
    return n * (1.0 + beta) / kappa


def _phi(u: float) -> float:
    return math.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)


def _Phi(u: float) -> float:
    return 0.5 * (1.0 + math.erf(u / math.sqrt(2.0)))


def statistical_dimension_l1(eps: float) -> float:
    """Normalized statistical dimension of the l1 descent cone at an eps-sparse point."""
    eps = min(max(eps, 1e-6), 1.0)
    best = 1.0
    for i in range(1, 2000):
        tau = i * 0.01
        off = 2.0 * ((1.0 + tau * tau) * (1.0 - _Phi(tau)) - tau * _phi(tau))
        val = eps * (1.0 + tau * tau) + (1.0 - eps) * off
        best = min(best, val)
    return min(best, 1.0)


def rho_dt(n: int = FRAME_LEN, gamma: int = 2, beta: float = RRC_ROLLOFF,
           kappa: int = SAMPLES_PER_SYMBOL) -> float:
    """Recovery-stability lower bound on rho from the statistical dimension (Eq. 10)."""
    d = gamma * n
    eps = k_eff(n, beta, kappa) / d
    return gamma * statistical_dimension_l1(eps)


def admissible_band(rho_dt_val: float, rho_label: float, rho_max: float) -> dict:
    """Assemble the admissible rho band [max(rho_dt, rho_label), rho_max] (Eq. 12)."""
    lo = max(rho_dt_val, rho_label)
    hi = rho_max
    return {"lo": lo, "hi": hi, "nonempty": lo <= hi, "width": max(0.0, hi - lo)}


# dictionary arms for rml2018: window and hop are set by the symbol rate, so they live here
SYMBOL_WINDOW = 4 * SAMPLES_PER_SYMBOL

DICTIONARIES = {
    "dft": lambda n, device: dft_frame(n, device=device),
    "windowed_dft": lambda n, device: gabor_frame(n, n, n // 2, device=device),
    "gabor_symbol": lambda n, device: gabor_frame(n, SYMBOL_WINDOW, SAMPLES_PER_SYMBOL, device=device),
}

DEFAULT_DICTIONARY = "gabor_symbol"


def build_dictionary(name: str, n: int = FRAME_LEN, device=None) -> Frame:
    """Build one named dictionary arm at frame length n."""
    if name not in DICTIONARIES:
        raise ValueError(f"unknown dictionary {name!r}, expected one of {sorted(DICTIONARIES)}")
    return DICTIONARIES[name](n, device)
