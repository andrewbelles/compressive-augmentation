import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample as scipy_resample


_DCT_PROBS_CACHE: dict[int, np.ndarray] = {}

EPS = 1e-12


def _get_dct_probs(n: int) -> np.ndarray:
    """
    Return cached biased DCT sampling probabilities for a signal length.

    - Lower DCT frequencies should be sampled with higher probability.
    """
    if n not in _DCT_PROBS_CACHE:
        probs = 1.0 / np.sqrt(np.arange(1, n + 1, dtype=np.float32))
        probs /= probs.sum()
        _DCT_PROBS_CACHE[n] = probs
    return _DCT_PROBS_CACHE[n]


def _gpu_dct_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Compute an orthonormal DCT-II for a batch using FFT primitives.

    Assumptions:
    - Input is shaped as batch by time and may be safely promoted to float32.
    """
    B, T = x.shape
    v    = torch.cat([x, x.flip(-1)], dim=-1)
    V    = torch.fft.rfft(v.float(), n=2 * T)[:, :T]
    k    = torch.arange(T, device=x.device, dtype=torch.float32)
    phase = torch.exp((-1j * math.pi / (2 * T)) * k.to(torch.complex64))
    C     = (V.to(torch.complex64) * phase).real
    C[:, 0]  /= math.sqrt(4 * T)
    C[:, 1:] /= math.sqrt(2 * T)
    return C


def _gpu_idct_batch(C: torch.Tensor) -> torch.Tensor:
    """
    Invert the batched orthonormal DCT representation used by _gpu_dct_batch.

    Assumptions:
    - Coefficients follow the same normalization as _gpu_dct_batch.
    """
    B, T = C.shape
    k    = torch.arange(T, device=C.device, dtype=torch.float32)
    C2   = C.float().clone()
    C2[:, 0]  *= math.sqrt(4 * T)
    C2[:, 1:] *= math.sqrt(2 * T)
    phase  = torch.exp((1j * math.pi / (2 * T)) * k.to(torch.complex64))
    V_half = C2.to(torch.complex64) * phase
    V      = torch.zeros(B, 2 * T, dtype=torch.complex64, device=C.device)
    V[:, :T]     = V_half
    V[:, T + 1:] = V_half[:, 1:].flip(-1).conj()
    return torch.fft.ifft(V).real[:, :T].float()


def _gpu_wht_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Apply Walsh-Hadamard transform over batched vectors.

    Assumptions:
    - The time dimension is already padded to a power of two.
    """
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
    """
    Generate batched DCT compressive-sensing reconstruction views on GPU.

    Assumptions:
    - ratio is a percent and gen is seeded by the caller for reproducible views.
    """
    B, T = x.shape
    m    = max(1, int(round(T * ratio / 100.0)))
    C    = _gpu_dct_batch(x)
    if uniform:
        scores = torch.rand(B, T, device=x.device, generator=gen)
    else:
        log_p  = -0.5 * torch.arange(1, T + 1, device=x.device, dtype=torch.float32).log()
        gumbel = -torch.log(
            -torch.log(torch.rand(B, T, device=x.device, generator=gen
        ).clamp_min(EPS)))
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
    """
    Generate batched SRHT compressive-sensing reconstruction views on GPU.

    Assumptions:
    - Input waveforms are fixed length, padding is internal and removed on return.
    """
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


def gpu_wave_policy_batch(
    x: torch.Tensor,
    policy: str,
    config: dict,
    gen: torch.Generator,
) -> torch.Tensor:
    """
    Apply a waveform augmentation policy to a batch on GPU.

    Assumptions:
    - policy uses the w2/w3/w4 semantics and config provides all required keys.
    """
    B, T   = x.shape
    lo, hi = float(config["wave_stretch_scale"][0]), float(config["wave_stretch_scale"][1])
    scale  = float(torch.empty(1, device=x.device).uniform_(lo, hi, generator=gen))
    n_res  = max(1, int(round(T * scale)))
    x_s    = F.interpolate(x.unsqueeze(1).float(), size=n_res, mode="linear", align_corners=False).squeeze(1)
    if n_res >= T:
        start = int(torch.randint(0, n_res - T + 1, (1,), device=x.device, generator=gen).item())
        x     = x_s[:, start : start + T]
    else:
        pad      = T - n_res
        pad_left = int(torch.randint(0, pad + 1, (1,), device=x.device, generator=gen).item())
        x_new    = torch.zeros(B, T, device=x.device, dtype=torch.float32)
        x_new[:, pad_left : pad_left + n_res] = x_s
        x = x_new
    if policy in {"w2", "w3", "w4"}:
        strength = float(config["wave_gain_strength"])
        gains    = torch.empty(B, device=x.device).uniform_(1.0 - strength, 1.0 + strength, generator=gen)
        x        = x * gains.unsqueeze(1)
    if policy in {"w3", "w4"}:
        n_masks = int(config["wave_n_masks"])
        max_w   = int(config["wave_mask_width"])
        for _ in range(n_masks):
            w = int(torch.randint(1, max_w + 1, (1,), device=x.device, generator=gen).item())
            s = int(torch.randint(0, max(1, T - w), (1,), device=x.device, generator=gen).item())
            x[:, s : s + w] = 0.0
    if policy == "w3":
        std = float(config["wave_noise_std"])
        x   = x + torch.randn(B, T, device=x.device, generator=gen) * std
    return x


def apply_wave_policy(y: np.ndarray, policy: str, config: dict, rng: np.random.Generator) -> np.ndarray:
    """
    Apply the numpy/scipy version of the waveform augmentation policy.

    Assumptions:
    - Used for CPU dataset paths; GPU training should use gpu_wave_policy_batch.
    """
    scale   = float(rng.uniform(float(config["wave_stretch_scale"][0]), float(config["wave_stretch_scale"][1])))
    n       = len(y)
    n_res   = max(1, int(round(n * scale)))
    y       = scipy_resample(y.astype(np.float64), n_res).astype(np.float32)
    if n_res >= n:
        start = int(rng.integers(0, n_res - n + 1))
        y     = y[start : start + n]
    else:
        pad      = n - n_res
        pad_left = int(rng.integers(0, pad + 1))
        y        = np.pad(y, (pad_left, pad - pad_left))
    if policy in {"w2", "w3", "w4"}:
        y = (y * float(rng.uniform(1.0 - config["wave_gain_strength"], 1.0 + config["wave_gain_strength"]))).astype(np.float32)
    if policy == "w3":
        y = _mask(y, int(config["wave_n_masks"]), int(config["wave_mask_width"]), rng)
        y = (y + rng.standard_normal(len(y)).astype(np.float32) * float(config["wave_noise_std"])).astype(np.float32)
    if policy == "w4":
        y = _mask(y, int(config["wave_n_masks"]), int(config["wave_mask_width"]), rng)
    return y


def _mask(y: np.ndarray, n_masks: int, max_width: int, rng: np.random.Generator) -> np.ndarray:
    """
    Zero random contiguous spans in a waveform copy.

    Assumptions:
    - max_width is positive and no larger than the intended augmentation scale.
    """
    y = y.copy()
    n = len(y)
    for _ in range(n_masks):
        w = int(rng.integers(1, max_width + 1))
        s = int(rng.integers(0, max(1, n - w)))
        y[s : s + w] = 0.0
    return y
