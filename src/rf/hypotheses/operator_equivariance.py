import math

import torch

from dsp.operators import (
    ConvOperator,
    SRHTOperator,
    measure,
    random_convolution,
    srht,
    srht_measure,
)
from rf.hypotheses._artifacts import normalize_power

# Layer III Prop 3 as exact operator algebra, with no solver in the loop. Timing and on-grid CFO
# admit per-draw identities; SRHT is the negative control the derivation's Layer III table predicts
# must fail. Accept: the convolution identities hold to machine precision and the SRHT one does not.

SHIFT_DIV = 4
CFO_BIN = 3
CFO_OFFGRID = 3.5
FRAC_DELAY = 2.5
EXACT_TOL = 1e-4


def _rel(a, b):
    """Max deviation relative to the measurement scale."""
    return ((a - b).abs().max() / a.abs().mean().clamp_min(1e-12)).item()


def _fractional_delay(x, delay):
    n = x.shape[-1]
    f = torch.fft.fftfreq(n, device=x.device)
    return torch.fft.ifft(torch.fft.fft(x) * torch.exp(-2j * math.pi * f * delay))


def run(frames, meta, ratios, seed, device) -> list[dict]:
    n = frames.shape[-1]
    x = normalize_power(frames)
    a = n // SHIFT_DIV
    k = torch.arange(n, device=device)
    shifted = torch.roll(x, a, dims=-1)
    cfo_on = x * torch.exp(2j * math.pi * CFO_BIN * k / n)
    cfo_off = x * torch.exp(2j * math.pi * CFO_OFFGRID * k / n)
    timing_frac = _fractional_delay(x, FRAC_DELAY)
    records = []
    for rho in ratios:
        m = max(1, round(rho * n))
        gen = torch.Generator(device=device).manual_seed(seed)
        op = random_convolution(n, m, gen, device=device)
        base = measure(x, op)

        # timing: C circulant commutes with T_a, and selecting Omega from a shifted signal is
        # selecting Omega - a from the original, so the identity is exact per draw
        op_shift = ConvOperator(n, m, (op.omega - a) % n, op.theta)
        conv_timing = _rel(measure(shifted, op), measure(x, op_shift))

        # on-grid CFO: Phi M_b = U Phi' with U unimodular and Phi' carrying cyclically shifted phases
        op_cfo = ConvOperator(n, m, op.omega, torch.roll(op.theta, -CFO_BIN))
        conv_cfo = _rel(measure(cfo_on, op).abs(), measure(x, op_cfo).abs())

        sop = srht(n, m, torch.Generator(device=device).manual_seed(seed), device=device)
        sop_shift = SRHTOperator(n, m, (sop.omega - a) % n, sop.signs)
        srht_timing = _rel(srht_measure(shifted, sop), srht_measure(x, sop_shift))

        records.append({
            "rho": rho,
            "m": m,
            "conv_timing_err": conv_timing,
            "conv_cfo_ongrid_err": conv_cfo,
            "srht_timing_err": srht_timing,
            "conv_exact": conv_timing < EXACT_TOL and conv_cfo < EXACT_TOL,
            "srht_exact": srht_timing < EXACT_TOL,
            # no exact claim in the theory: recorded for the record, not scored
            "conv_cfo_offgrid_err": _rel(measure(cfo_off, op).abs(), base.abs()),
            "conv_timing_frac_err": _rel(measure(timing_frac, op).abs(), base.abs()),
        })
    return records
