import math

import torch

# Rung-5 impairment injection. Corruption is added to x BEFORE sensing so the
# error-dictionary model y = Phi D alpha + Phi e holds exactly: impulsive
# bursts are sparse in time (E = I), CCI tones are sparse in frequency
# (E = inverse-DFT columns).


def inject_impulsive(
    x: torch.Tensor,
    n_bursts: int,
    burst_len: int,
    amp_rel: float,
    gen: torch.Generator,
) -> torch.Tensor:
    """Add time-sparse complex bursts with per-sample amplitude amp_rel * frame RMS."""
    B, N = x.shape
    rms  = x.abs().pow(2).mean(dim=1, keepdim=True).sqrt()
    starts  = torch.randint(0, N, (B, n_bursts), generator=gen, device=x.device)
    offsets = torch.arange(burst_len, device=x.device)
    idx     = (starts.unsqueeze(-1) + offsets).reshape(B, -1) % N
    theta   = torch.rand(B, n_bursts * burst_len, generator=gen, device=x.device) * (2.0 * math.pi)
    values  = (amp_rel * rms).to(x.dtype) * torch.exp(1j * theta).to(x.dtype)
    e = torch.zeros_like(x)
    e.scatter_(1, idx, values)
    return x + e


def inject_cci(
    x: torch.Tensor,
    n_tones: int,
    sir_db: float,
    gen: torch.Generator,
) -> torch.Tensor:
    """Add frequency-sparse co-channel interference tones at random DFT bins.

    Total interference power is set relative to the frame's signal power by
    sir_db (signal-to-interference ratio).
    """
    B, N = x.shape
    p_sig  = x.abs().pow(2).mean(dim=1, keepdim=True)
    p_int  = p_sig * (10.0 ** (-sir_db / 10.0))
    amp    = (p_int / n_tones).sqrt()
    bins   = torch.randint(0, N, (B, n_tones), generator=gen, device=x.device)
    theta  = torch.rand(B, n_tones, generator=gen, device=x.device) * (2.0 * math.pi)
    n      = torch.arange(N, device=x.device, dtype=torch.float32)
    phase  = (2.0 * math.pi / N) * bins.unsqueeze(-1).float() * n + theta.unsqueeze(-1)
    tones  = torch.exp(1j * phase).to(x.dtype).sum(dim=1)
    return x + amp.to(x.dtype) * tones
