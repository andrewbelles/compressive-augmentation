import math

import torch

# Complex sensing operators (domain-agnostic, single C-linear maps on complex64
# signals): all matrices are materialized explicitly so downstream code can
# precompute Phi @ D products, Gram matrices, and coherence checks uniformly.

FAMILIES = ("identity", "gaussian", "demod", "fourier")


def complex_gaussian_matrix(
    m: int, n: int, gen: torch.Generator, device: torch.device
) -> torch.Tensor:
    """Return an m x n iid CN(0, 1/m) sensing matrix (unit expected column energy)."""
    real = torch.randn(m, n, generator=gen, device=device)
    imag = torch.randn(m, n, generator=gen, device=device)
    return ((real + 1j * imag) / math.sqrt(2.0 * m)).to(torch.complex64)


def random_demodulator_matrix(
    m: int, n: int, gen: torch.Generator, device: torch.device, unit_phase: bool = False
) -> torch.Tensor:
    """Return an m x n random-demodulator matrix Phi = B @ diag(c).

    c is a length-n chip sequence (random +-1, or unit-phase complex when
    ``unit_phase``); B is the m x n boxcar integrate-and-dump matrix that sums
    sample i into row floor(i * m / n), so any m <= n works.
    """
    if unit_phase:
        theta = torch.rand(n, generator=gen, device=device) * (2.0 * math.pi)
        chips = torch.exp(1j * theta).to(torch.complex64)
    else:
        signs = torch.randint(0, 2, (n,), generator=gen, device=device, dtype=torch.float32)
        chips = (signs * 2.0 - 1.0).to(torch.complex64)
    rows = (torch.arange(n, device=device) * m) // n
    boxcar = torch.zeros(m, n, dtype=torch.complex64, device=device)
    boxcar[rows, torch.arange(n, device=device)] = 1.0
    return boxcar * chips.unsqueeze(0)


def partial_fourier_matrix(
    m: int, n: int, gen: torch.Generator, device: torch.device
) -> torch.Tensor:
    """Return an m x n partial-Fourier matrix sqrt(n/m) * R_Omega @ F (F unitary DFT)."""
    k = torch.arange(n, device=device, dtype=torch.float32)
    F = torch.exp((-2j * math.pi / n) * torch.outer(k, k).to(torch.complex64)) / math.sqrt(n)
    rows = torch.randperm(n, generator=gen, device=device)[:m]
    return (math.sqrt(n / m) * F[rows]).to(torch.complex64)


def identity_operator(n: int, device: torch.device) -> torch.Tensor:
    """Return the n x n identity as a complex64 sensing matrix (rho = 1 control)."""
    return torch.eye(n, dtype=torch.complex64, device=device)


def build_sensing_matrix(
    family: str, rho: float, n: int, seed: int, device: torch.device
) -> torch.Tensor:
    """Build the deterministic sensing matrix for (family, rho, seed) with m = round(rho * n)."""
    if family == "identity":
        return identity_operator(n, device)
    m   = max(1, int(round(rho * n)))
    gen = torch.Generator(device=device).manual_seed(seed)
    if family == "gaussian":
        return complex_gaussian_matrix(m, n, gen, device)
    if family == "demod":
        return random_demodulator_matrix(m, n, gen, device)
    if family == "fourier":
        return partial_fourier_matrix(m, n, gen, device)
    raise ValueError(f"unknown sensing family: {family!r} (expected one of {FAMILIES})")


def mutual_coherence(A: torch.Tensor, B: torch.Tensor | None = None) -> float:
    """Return max normalized |<a_i, b_j>| between columns (self-coherence excludes the diagonal)."""
    eps = torch.finfo(torch.float32).tiny
    An  = A / A.norm(dim=0, keepdim=True).clamp_min(eps)
    if B is None:
        G = (An.conj().T @ An).abs()
        G.fill_diagonal_(0.0)
        return G.max().item()
    Bn = B / B.norm(dim=0, keepdim=True).clamp_min(eps)
    return (An.conj().T @ Bn).abs().max().item()
