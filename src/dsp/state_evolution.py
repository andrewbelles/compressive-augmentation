from functools import lru_cache

import torch

from .recovery import complex_soft_threshold, soft_threshold_divergence

# state evolution for the OAMP soft-threshold denoiser: scalar tau, calibration SNR, Beta-law

EPS = 1e-12


def _bernoulli_gaussian(shape, eps: float, gen: torch.Generator, device) -> torch.Tensor:
    """Draw unit-power complex Bernoulli-Gaussian coefficients (spike rate eps)."""
    mask = (torch.rand(shape, generator=gen, device=device) < eps).to(torch.complex64)
    scale = (1.0 / max(eps, EPS)) ** 0.5
    g = torch.randn(shape, generator=gen, device=device, dtype=torch.complex64)
    return mask * g * scale


def _denoise_mse(tau: float, eps: float, kappa: float, gen, device, n: int = 100000) -> float:
    """Monte-Carlo E|eta(alpha + tau Z) - alpha|^2 for the divergence-free denoiser."""
    alpha = _bernoulli_gaussian((n,), eps, gen, device)
    z = torch.randn(n, generator=gen, device=device, dtype=torch.complex64)
    r = alpha + tau * z
    theta = kappa * tau
    eta = complex_soft_threshold(r, theta)
    div = soft_threshold_divergence(r.unsqueeze(0), theta).squeeze().clamp(0.0, 0.95)
    s = (eta - div * r) / (1.0 - div)
    return (s - alpha).abs().pow(2).mean().item()


def se_fixed_point(
    rho: float,
    sigma: float,
    eps: float,
    gamma: int,
    n: int,
    kappa: float = 1.5,
    seed: int = 0,
    iters: int = 40,
    orthogonal: bool = True,
    mc: int = 100000,
) -> dict:
    """Solve the OAMP state-evolution fixed point, returning tau and denoiser MSE."""
    device = torch.device("cpu")
    gen = torch.Generator(device=device).manual_seed(seed)
    d = gamma * n
    m = round(rho * n)
    delta = m / d
    coef = (1.0 - delta) / delta if orthogonal else 1.0 / delta
    tau2 = sigma * sigma + coef
    for _ in range(iters):
        v2 = _denoise_mse(tau2 ** 0.5, eps, kappa, gen, device, mc)
        tau2 = sigma * sigma + coef * v2
    return {"tau": tau2 ** 0.5, "mse": v2, "delta": delta, "rho": rho}


KAPPA_GRID = tuple(round(0.3 + 0.1 * i, 1) for i in range(16))
KAPPA_MC = 10000


@lru_cache(maxsize=4096)
def optimal_kappa(rho: float, sigma: float, eps: float, gamma: float, n: int) -> float:
    """Threshold minimizing the SE fixed-point MSE; a prediction, never fitted to realized data."""
    # rho comes from a discrete grid, so caching on the argument tuple collapses the repeats across
    # strata; the reduced Monte-Carlo budget is enough for an argmin over a 0.1-spaced grid
    scored = [
        (se_fixed_point(rho, sigma, eps, gamma, n, kappa=k, mc=KAPPA_MC)["mse"], k)
        for k in KAPPA_GRID
    ]
    return min(scored)[1]


def calibration_snr(rho: float, sigma: float, eps: float, gamma: int, n: int, **kw) -> float:
    """Result 2 recovery SNR in dB at the SE fixed point (unit-power signal)."""
    fp = se_fixed_point(rho, sigma, eps, gamma, n, **kw)
    return -10.0 * torch.log10(torch.tensor(fp["mse"]).clamp_min(EPS)).item()


def beta_law_moments(n: int, m: int) -> dict:
    """Result 3 mean and variance of the back-projection residual energy fraction."""
    rho = m / n
    return {"mean": 1.0 - rho, "var": rho * (1.0 - rho) / (n + 1)}
