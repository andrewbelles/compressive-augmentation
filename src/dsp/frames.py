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


def gabor_frame(n: int, window: int, hop: int, n_freq: int = None, device=None) -> Frame:
    """Build an exactly tight Gabor frame from window length, hop, and frequency channels."""
    n_freq = n_freq or window
    if n % hop != 0:
        raise ValueError(f"n={n} must be divisible by hop={hop}")
    if hop > window:
        raise ValueError(f"hop={hop} must not exceed window={window} or the lattice leaves gaps")
    if n_freq < window:
        raise ValueError(f"n_freq={n_freq} must be at least window={window} for tightness")
    if window > n:
        raise ValueError(f"window={window} must not exceed n={n}")
    device = device or torch.device("cpu")
    shifts = n // hop
    idx = torch.arange(n, device=device)
    base = _hann(window, device)
    placed = torch.zeros(shifts, n, device=device)
    for j in range(shifts):
        placed[j, (idx[:window] + j * hop) % n] = base
    win = placed / placed.pow(2).sum(dim=0).clamp_min(1e-12).sqrt()
    ramp = torch.arange(n_freq, device=device)
    phase = torch.outer(idx, ramp) % n_freq  # reduce before exp: float32 loses precision past ~1e3 rad
    freqs = torch.exp(2j * math.pi * phase.float() / n_freq)
    scale = 1.0 / math.sqrt(hop)  # makes DD* = (n_freq/hop) I
    cols = [scale * win[j].to(torch.complex64).unsqueeze(1) * freqs for j in range(shifts)]
    return Frame(torch.cat(cols, dim=1), n, shifts * n_freq, n_freq / hop)


def dft_frame(n: int, device=None) -> Frame:
    """Build the orthonormal DFT frame (gamma=1), whose supports are contiguous bands."""
    device = device or torch.device("cpu")
    idx = torch.arange(n, device=device)
    phase = (torch.outer(idx, idx) % n).float()
    d = torch.exp(2j * math.pi * phase / n) / math.sqrt(n)
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
