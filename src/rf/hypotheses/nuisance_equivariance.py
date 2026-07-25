import math

import torch

from dsp.operators import random_convolution, measure
from dsp.recovery import oamp, reconstruct
from rf.hypotheses._artifacts import normalize_power, snr_strata
from rf.signal_model import DEFAULT_DICTIONARY, build_dictionary

# Layer III Prop 3: recovery is equivariant in law under the channel nuisances. On-grid shift and
# CFO are near-structural for a circulant Phi, so the off-grid and fractional cases carry the test.
# Accept: mean recovery SNR is invariant under every nuisance, with snr_base above the floor.


def _snr(x, xhat):
    num = x.abs().pow(2).sum(-1)
    den = (x - xhat).abs().pow(2).sum(-1).clamp_min(1e-12)
    return 10 * torch.log10(num / den)


def _recover_snr(x, op, frame):
    return _snr(x, reconstruct(oamp(measure(x, op), op, frame), frame)).mean().item()


def _fractional_delay(x, delay):
    """Shift by a non-integer number of samples via a spectral phase ramp."""
    n = x.shape[-1]
    f = torch.fft.fftfreq(n, device=x.device)
    return torch.fft.ifft(torch.fft.fft(x) * torch.exp(-2j * math.pi * f * delay))


def _nuisances(x):
    n = x.shape[-1]
    k = torch.arange(n, device=x.device)
    return {
        "shift": torch.roll(x, n // 4, dims=-1),
        "cfo_ongrid": x * torch.exp(2j * math.pi * 3 * k / n),
        "cfo_offgrid": x * torch.exp(2j * math.pi * 3.5 * k / n),
        "timing": _fractional_delay(x, 2.5),
        "phase": x * torch.exp(torch.tensor(0.7j, device=x.device)),
    }


def run(frames, meta, ratios, seed, device, dictionary=DEFAULT_DICTIONARY) -> list[dict]:
    n = frames.shape[-1]
    frame = build_dictionary(dictionary, n, device)
    x = normalize_power(frames)
    records = []
    for snr_db, rows in snr_strata(meta, x.shape[0]).items():
        sel = x[torch.tensor(rows, device=device)]
        perturbed = _nuisances(sel)
        for rho in ratios:
            m = max(1, round(rho * n))
            gen = torch.Generator(device=device).manual_seed(seed)
            op = random_convolution(n, m, gen, device=device)
            base = _recover_snr(sel, op, frame)
            row = {"snr": snr_db, "rho": rho, "m": m, "snr_base": base}
            for name, xp in perturbed.items():
                row[f"{name}_gap"] = abs(_recover_snr(xp, op, frame) - base)
            records.append(row)
    return records
