import math
from dataclasses import dataclass

import torch

# tight Gabor synthesis frame D (DD* = gamma I) and orthonormal DFT frame for the degeneracy check


@dataclass(frozen=True)
class Frame:
    """Dense synthesis frame D of shape (n, d) with tight-frame constant gamma."""
    d: torch.Tensor
    n: int
    n_atoms: int
    gamma: float

    @property
    def redundancy(self) -> float:
        return self.n_atoms / self.n

    def to(self, device) -> "Frame":
        return Frame(self.d.to(device), self.n, self.n_atoms, self.gamma)


def _hann(length: int, device) -> torch.Tensor:
    k = torch.arange(length, device=device, dtype=torch.float32)
    return torch.sin(math.pi * (k + 0.5) / length) ** 2


def gabor_frame(n: int, gamma: int = 2, device=None) -> Frame:
    """Build an exactly tight Gabor frame with gamma time shifts and n frequency channels."""
    if n % gamma != 0:
        raise ValueError(f"n={n} must be divisible by gamma={gamma}")
    device = device or torch.device("cpu")
    a = n // gamma
    idx = torch.arange(n, device=device)
    base = _hann(min(n, 2 * a), device)
    shifts = torch.zeros(gamma, n, device=device)
    for k in range(gamma):
        placed = torch.zeros(n, device=device)
        pos = (idx[: base.numel()] + k * a) % n
        placed[pos] = base
        shifts[k] = placed
    norm = shifts.pow(2).sum(dim=0).clamp_min(1e-12).sqrt()
    win = shifts / norm
    freqs = torch.exp(2j * math.pi * torch.outer(idx.float(), idx.float()) / n)
    scale = math.sqrt(gamma / n)
    cols = []
    for k in range(gamma):
        cols.append(scale * win[k].to(torch.complex64).unsqueeze(1) * freqs)
    d = torch.cat(cols, dim=1)
    return Frame(d, n, gamma * n, float(gamma))


def dft_frame(n: int, device=None) -> Frame:
    """Build the orthonormal DFT frame (gamma=1), whose supports are contiguous bands."""
    device = device or torch.device("cpu")
    idx = torch.arange(n, device=device).float()
    d = torch.exp(2j * math.pi * torch.outer(idx, idx) / n) / math.sqrt(n)
    return Frame(d.to(torch.complex64), n, n, 1.0)


def synthesis(alpha: torch.Tensor, frame: Frame) -> torch.Tensor:
    """Map coefficients to a signal, x = D alpha."""
    return alpha.to(frame.d.dtype) @ frame.d.transpose(-1, -2)


def analysis(x: torch.Tensor, frame: Frame) -> torch.Tensor:
    """Map a signal to coefficients, alpha = D* x."""
    return x.to(frame.d.dtype) @ frame.d.conj()


def is_tight(frame: Frame, atol: float = 1e-4) -> bool:
    """Check DD* = gamma I within tolerance."""
    gram = frame.d @ frame.d.conj().transpose(-1, -2)
    target = frame.gamma * torch.eye(frame.n, dtype=gram.dtype, device=gram.device)
    return torch.allclose(gram, target, atol=atol)
