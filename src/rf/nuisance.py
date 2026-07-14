import math

import torch

# V1 nuisance-orbit transforms: global phase, carrier frequency offset (CFO),
# and cyclic time shift. Each is a C-linear (unitary, diagonal or permutation)
# map applied jointly to I/Q via the complex tensor.


def random_global_phase(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Multiply each frame by a random global phase e^{j theta}, theta ~ U[0, 2pi)."""
    theta = torch.rand(x.shape[0], 1, generator=gen, device=x.device) * (2.0 * math.pi)
    return x * torch.exp(1j * theta).to(x.dtype)


def random_cfo(x: torch.Tensor, max_eps: float, gen: torch.Generator) -> torch.Tensor:
    """Apply a random carrier frequency offset x[n] * e^{j 2 pi eps n / N}, eps ~ U[-max, max]."""
    B, N = x.shape
    eps  = (torch.rand(B, 1, generator=gen, device=x.device) * 2.0 - 1.0) * max_eps
    n    = torch.arange(N, device=x.device, dtype=torch.float32).unsqueeze(0)
    return x * torch.exp((2j * math.pi / N) * (eps * n)).to(x.dtype)


def random_cyclic_shift(x: torch.Tensor, max_shift: int, gen: torch.Generator) -> torch.Tensor:
    """Cyclically shift each frame by a random integer in [-max_shift, max_shift]."""
    B, N  = x.shape
    shift = torch.randint(-max_shift, max_shift + 1, (B, 1), generator=gen, device=x.device)
    idx   = (torch.arange(N, device=x.device).unsqueeze(0) - shift) % N
    return x.gather(1, idx.expand(B, N))


def orbit_augment(
    x: torch.Tensor,
    n_copies: int,
    max_eps: float,
    max_shift: int,
    gen: torch.Generator,
) -> torch.Tensor:
    """Expand [B, N] to [B * n_copies, N] over the nuisance orbit (copy-major blocks).

    Copy 0 is the identity; the remaining copies apply phase, CFO, and cyclic
    shift jointly. Output rows [0:B] are the originals, [B:2B] copy 1, etc.
    """
    copies = [x]
    for _ in range(n_copies - 1):
        v = random_global_phase(x, gen)
        v = random_cfo(v, max_eps, gen)
        v = random_cyclic_shift(v, max_shift, gen)
        copies.append(v)
    return torch.cat(copies, dim=0)
