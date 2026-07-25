import math
from dataclasses import dataclass

import torch

# random-convolution measurement operator Phi = sqrt(N/m) R_Omega F* Theta F


@dataclass(frozen=True)
class ConvOperator:
    """Random-convolution sensing operator: index set Omega and unit-modulus spectrum Theta."""
    n: int
    m: int
    omega: torch.Tensor
    theta: torch.Tensor

    @property
    def rho(self) -> float:
        return self.m / self.n

    def to(self, device) -> "ConvOperator":
        return ConvOperator(self.n, self.m, self.omega.to(device), self.theta.to(device))


def random_convolution(n: int, m: int, gen: torch.Generator, device=None) -> ConvOperator:
    """Draw a random-convolution operator with m of n retained rows and random phases."""
    device = device or gen.device
    theta = torch.exp(2j * math.pi * torch.rand(n, generator=gen, device=device))
    omega = torch.randperm(n, generator=gen, device=device)[:m].sort().values
    return ConvOperator(n, m, omega, theta.to(torch.complex64))


def measure(x: torch.Tensor, op: ConvOperator) -> torch.Tensor:
    """Apply Phi to a batch of complex frames, returning m-dimensional measurements."""
    scale = math.sqrt(op.n / op.m)
    cx = torch.fft.ifft(op.theta * torch.fft.fft(x, norm="ortho"), norm="ortho")
    return scale * cx[..., op.omega]


def adjoint(y: torch.Tensor, op: ConvOperator) -> torch.Tensor:
    """Apply Phi* to measurements, returning n-dimensional frames."""
    scale = math.sqrt(op.n / op.m)
    z = torch.zeros(*y.shape[:-1], op.n, dtype=y.dtype, device=y.device)
    z[..., op.omega] = y
    cz = torch.fft.ifft(op.theta.conj() * torch.fft.fft(z, norm="ortho"), norm="ortho")
    return scale * cz


def backproject(y: torch.Tensor, op: ConvOperator) -> torch.Tensor:
    """Apply the pseudo-inverse Phi^dagger = (m/n) Phi* (Result 3 linear baseline)."""
    return (op.m / op.n) * adjoint(y, op)


def to_dense(op: ConvOperator) -> torch.Tensor:
    """Materialize Phi as an (m, n) complex matrix by measuring the identity."""
    eye = torch.eye(op.n, dtype=torch.complex64, device=op.theta.device)
    return measure(eye, op).transpose(-1, -2).contiguous()


def mutual_coherence(a: torch.Tensor) -> float:
    """Largest normalized inner product between distinct columns of a."""
    cols = a / a.norm(dim=0, keepdim=True).clamp_min(1e-12)
    gram = (cols.conj().transpose(-1, -2) @ cols).abs()
    gram.fill_diagonal_(0.0)
    return gram.max().item()
