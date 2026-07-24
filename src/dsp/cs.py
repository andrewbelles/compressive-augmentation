import math

import numpy as np
import torch

_DCT_PROBS_CACHE: dict[int, np.ndarray] = {}

EPS = 1e-12

# domain-agnostic CS operators (real 1-D signals): gpu_dct_cs_view_batch, gpu_srht_batch


def _get_dct_probs(n: int) -> np.ndarray:
    """Return cached frequency-biased DCT sampling probabilities for signal length n."""
    if n not in _DCT_PROBS_CACHE:
        probs = 1.0 / np.sqrt(np.arange(1, n + 1, dtype=np.float32))
        probs /= probs.sum()
        _DCT_PROBS_CACHE[n] = probs
    return _DCT_PROBS_CACHE[n]


def _gpu_dct_batch(x: torch.Tensor) -> torch.Tensor:
    """Compute orthonormal DCT-II for a batch using FFT primitives."""
    B, T  = x.shape
    v     = torch.cat([x, x.flip(-1)], dim=-1)
    V     = torch.fft.rfft(v.float(), n=2 * T)[:, :T]
    k     = torch.arange(T, device=x.device, dtype=torch.float32)
    phase = torch.exp((-1j * math.pi / (2 * T)) * k.to(torch.complex64))
    C     = (V.to(torch.complex64) * phase).real
    C[:, 0]  /= math.sqrt(4 * T)
    C[:, 1:] /= math.sqrt(2 * T)
    return C


def _gpu_idct_batch(C: torch.Tensor) -> torch.Tensor:
    """Invert the orthonormal DCT produced by _gpu_dct_batch."""
    B, T   = C.shape
    k      = torch.arange(T, device=C.device, dtype=torch.float32)
    C2     = C.float().clone()
    C2[:, 0]  *= math.sqrt(4 * T)
    C2[:, 1:] *= math.sqrt(2 * T)
    phase  = torch.exp((1j * math.pi / (2 * T)) * k.to(torch.complex64))
    V_half = C2.to(torch.complex64) * phase
    V      = torch.zeros(B, 2 * T, dtype=torch.complex64, device=C.device)
    V[:, :T]     = V_half
    V[:, T + 1:] = V_half[:, 1:].flip(-1).conj()
    return torch.fft.ifft(V).real[:, :T].float()


def _gpu_wht_batch(x: torch.Tensor) -> torch.Tensor:
    """Apply Walsh-Hadamard transform over batched power-of-two vectors."""
    B, p2 = x.shape
    h = 1
    while h < p2:
        x = x.view(B, -1, 2, h)
        u = x[:, :, 0, :].clone()
        x[:, :, 0, :] = u + x[:, :, 1, :]
        x[:, :, 1, :] = u - x[:, :, 1, :]
        x = x.view(B, p2)
        h *= 2
    return x


def gpu_dct_cs_view_batch(
    x: torch.Tensor,
    ratio: float,
    gen: torch.Generator,
    uniform: bool = False,
    energy_rescale: bool = True,
) -> torch.Tensor:
    """Generate batched DCT compressive-sensing reconstruction views on GPU."""
    B, T = x.shape
    m    = max(1, int(round(T * ratio / 100.0)))
    C    = _gpu_dct_batch(x)
    if uniform:
        scores = torch.rand(B, T, device=x.device, generator=gen)
    else:
        log_p  = -0.5 * torch.arange(1, T + 1, device=x.device, dtype=torch.float32).log()
        gumbel = -torch.log(
            -torch.log(torch.rand(B, T, device=x.device, generator=gen).clamp_min(EPS))
        )
        scores = log_p.unsqueeze(0) + gumbel
    _, idx = scores.topk(m, dim=-1)
    mask   = torch.zeros(B, T, device=x.device)
    mask.scatter_(1, idx, math.sqrt(T / m) if energy_rescale else 1.0)
    return _gpu_idct_batch(C * mask)


def gpu_srht_batch(
    x: torch.Tensor,
    ratio: float,
    gen: torch.Generator,
    energy_rescale: bool = True,
) -> torch.Tensor:
    """Generate batched SRHT compressive-sensing reconstruction views on GPU."""
    B, T  = x.shape
    p2    = 1 << math.ceil(math.log2(max(T, 2)))
    m     = max(1, int(round(T * ratio / 100.0)))
    signs = torch.randint(0, 2, (B, p2), device=x.device, generator=gen, dtype=torch.float32) * 2 - 1
    xp    = torch.zeros(B, p2, device=x.device, dtype=torch.float32)
    xp[:, :T] = x.float() * signs[:, :T]
    xp    = _gpu_wht_batch(xp) / math.sqrt(p2)
    _, idx = torch.rand(B, p2, device=x.device, generator=gen).topk(m, dim=-1)
    mask  = torch.zeros(B, p2, device=x.device)
    mask.scatter_(1, idx, math.sqrt(p2 / m) if energy_rescale else 1.0)
    z     = _gpu_wht_batch(xp * mask) / math.sqrt(p2)
    return z[:, :T] * signs[:, :T]
