import torch

from dsp.frames import synthesis
from dsp.operators import backproject, measure, measurement_noise, random_convolution
from dsp.recovery import debias, oamp, reconstruct

# augmentation arms produced at a matched signal-domain distortion, so comparisons are made at equal
# corruption rather than at equal rho; load-bearing for both GO gates, hence tested on its own

EPS = 1e-12
BISECT_ITERS = 12


def distortion(x: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
    """Per-frame relative squared error ||x_tilde - x||^2 / ||x||^2."""
    return (xt - x).abs().pow(2).sum(-1) / x.abs().pow(2).sum(-1).clamp_min(EPS)


def draw_noise(x, op, measurement_snr, gen):
    """Measurement noise for one cell; share it across views whose difference must isolate x."""
    return measurement_noise(measure(x, op), measurement_snr, op, gen)


def cs_view(x, op, frame, kappa, noise=None, debiased=True):
    """Full CS round trip, the reference arm that defines the target distortion."""
    y = measure(x, op)
    y = y if noise is None else y + noise
    alpha = oamp(y, op, frame, kappa=kappa)
    # soft thresholding shrinks amplitude non-uniformly, and the confusable classes are separated
    # by amplitude structure, so the shrunk estimate discards the label the view is meant to keep
    if debiased:
        alpha = debias(alpha, y, op, frame)
    return reconstruct(alpha, frame)


def awgn_view(x, target, gen):
    """Isotropic noise rescaled to hit the target distortion exactly, per frame."""
    z = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=gen)
    scale = (target * x.abs().pow(2).sum(-1) / z.abs().pow(2).sum(-1).clamp_min(EPS)).sqrt()
    return x + z * scale.unsqueeze(-1)


def backprojection_view(x, target_mean, seed, device):
    """Beta law gives residual 1 - rho, so rho_bp = 1 - target sets the distortion."""
    n = x.shape[-1]
    rho_bp = min(max(1.0 - target_mean, 1.0 / n), 1.0)
    m = max(1, round(rho_bp * n))
    gen = torch.Generator(device=device).manual_seed(seed)
    op = random_convolution(n, m, gen, device=device)
    return backproject(measure(x, op), op), rho_bp


def dropout_view(x, alpha, frame, target_mean, gen_seed, device):
    """Bernoulli atom dropout with the rate bisected onto the target distortion."""
    lo, hi = 0.0, 1.0
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        got = distortion(x, _drop(alpha, mid, frame, gen_seed, device)).mean().item()
        if got < target_mean:
            lo = mid
        else:
            hi = mid
    p = 0.5 * (lo + hi)
    return _drop(alpha, p, frame, gen_seed, device), p


def _drop(alpha, p, frame, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    keep = (torch.rand(alpha.shape, generator=gen, device=device) >= p).to(alpha.dtype)
    return synthesis(alpha * keep, frame)
