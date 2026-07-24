import math

import torch

from dsp.operators import random_convolution, measure
from dsp.frames import gabor_frame
from dsp.recovery import oamp, reconstruct

# Layer III Prop 3: recovery is equivariant in law under cyclic time shift and on-grid CFO.
# Accept: mean recovery SNR is invariant (small gap) under both nuisance transforms.

GAMMA = 2


def _snr(x, xhat):
    num = x.abs().pow(2).sum(-1)
    den = (x - xhat).abs().pow(2).sum(-1).clamp_min(1e-12)
    return 10 * torch.log10(num / den)


def _recover_snr(x, op, frame):
    return _snr(x, reconstruct(oamp(measure(x, op), op, frame), frame)).mean().item()


def run(frames, meta, ratios, seed, device) -> list[dict]:
    n = frames.shape[-1]
    frame = gabor_frame(n, GAMMA, device=device)
    k = torch.arange(n, device=device)
    shift = torch.roll(frames, n // 4, dims=-1)
    cfo = frames * torch.exp(2j * math.pi * 3 * k / n)
    records = []
    for rho in ratios:
        m = max(1, round(rho * n))
        op = random_convolution(n, m, torch.Generator(device=device).manual_seed(seed), device=device)
        base = _recover_snr(frames, op, frame)
        records.append({
            "rho": rho,
            "m": m,
            "snr_base": base,
            "shift_gap": abs(_recover_snr(shift, op, frame) - base),
            "cfo_gap": abs(_recover_snr(cfo, op, frame) - base),
        })
    return records
