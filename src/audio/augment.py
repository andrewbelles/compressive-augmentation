import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample as scipy_resample

# audio-specific augmentation policies: gpu_wave_policy_batch, apply_wave_policy, _mask


def gpu_wave_policy_batch(
    x: torch.Tensor,
    policy: str,
    config: dict,
    gen: torch.Generator,
) -> torch.Tensor:
    """Apply a waveform augmentation policy to a batch on GPU."""
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
    """Apply the numpy/scipy waveform augmentation policy; use gpu_wave_policy_batch for GPU training."""
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
    """Zero random contiguous spans in a waveform copy."""
    y = y.copy()
    n = len(y)
    for _ in range(n_masks):
        w = int(rng.integers(1, max_width + 1))
        s = int(rng.integers(0, max(1, n - w)))
        y[s : s + w] = 0.0
    return y
