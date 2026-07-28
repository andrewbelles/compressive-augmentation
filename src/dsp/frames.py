import math
from dataclasses import dataclass

import torch

# tight Gabor synthesis frame D (DD* = gamma I) and orthonormal DFT frame for the degeneracy check


@dataclass(frozen=True)
class GaborLattice:
    """Time-frequency lattice letting synthesis and analysis run as an inverse STFT and an STFT."""
    window: int
    hop: int
    n_freq: int
    shifts: int
    taps: int                # window // hop; when exact, overlap-add needs no atomics
    pos: torch.Tensor        # (shifts, window) output indices, already reduced mod n
    win: torch.Tensor        # (shifts, window) scaled window values at those indices
    phase_syn: torch.Tensor  # (shifts, n_freq) cyclic shift as a frequency ramp, carrying n_freq
    phase_ana: torch.Tensor  # (shifts, n_freq) its conjugate, without the n_freq factor

    def to(self, device) -> "GaborLattice":
        return GaborLattice(self.window, self.hop, self.n_freq, self.shifts, self.taps,
                            self.pos.to(device), self.win.to(device),
                            self.phase_syn.to(device), self.phase_ana.to(device))


@dataclass(frozen=True)
class Frame:
    """Dense synthesis frame D of shape (n, d) with tight-frame constant gamma."""
    d: torch.Tensor
    n: int
    n_atoms: int
    gamma: float
    lattice: GaborLattice = None  # None means no fast path, so synthesis falls back to D itself

    @property
    def redundancy(self) -> float:
        return self.n_atoms / self.n

    def to(self, device) -> "Frame":
        lat = self.lattice.to(device) if self.lattice is not None else None
        return Frame(self.d.to(device), self.n, self.n_atoms, self.gamma, lat)


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

    lattice = None
    # the fast path indexes the length-n_freq segment at (j*hop + u) mod n_freq, which only agrees
    # with the atom's absolute phase when wrapping at n cannot change the residue
    if n % n_freq == 0:
        shift = torch.arange(shifts, device=device).unsqueeze(1)
        pos = (idx[:window].unsqueeze(0) + shift * hop) % n
        ramp = torch.arange(n_freq, device=device).unsqueeze(0)
        # the gather is a cyclic shift of the segment, so carry it as a frequency ramp instead
        lat_phase = torch.exp(2j * math.pi * ((ramp * (shift * hop % n_freq)) % n_freq).float() / n_freq)
        lattice = GaborLattice(
            window, hop, n_freq, shifts, window // hop,
            pos,
            (scale * win.gather(1, pos)).to(torch.complex64),
            (n_freq * lat_phase).to(torch.complex64),
            lat_phase.conj().to(torch.complex64),
        )
    return Frame(torch.cat(cols, dim=1), n, shifts * n_freq, n_freq / hop, lattice)


def dft_frame(n: int, device=None) -> Frame:
    """Build the orthonormal DFT frame (gamma=1), whose supports are contiguous bands."""
    device = device or torch.device("cpu")
    idx = torch.arange(n, device=device)
    phase = (torch.outer(idx, idx) % n).float()
    d = torch.exp(2j * math.pi * phase / n) / math.sqrt(n)
    return Frame(d.to(torch.complex64), n, n, 1.0)


def synthesis(alpha: torch.Tensor, frame: Frame) -> torch.Tensor:
    """Map coefficients to a signal, x = D alpha."""
    alpha = alpha.to(frame.d.dtype)
    lat = frame.lattice
    if lat is None:
        return alpha @ frame.d.transpose(-1, -2)
    lead = alpha.shape[:-1]
    flat = alpha.reshape(-1, lat.shifts, lat.n_freq)
    seg = lat.win * torch.fft.ifft(flat * lat.phase_syn, n=lat.n_freq, dim=-1)
    b = flat.shape[0]
    if lat.taps * lat.hop == lat.window:
        # sample q*hop + c takes one tap from each of taps consecutive shifts, so the overlap-add is
        # a sum of shifted slices: no atomics, hence bit-reproducible across runs
        w = seg.reshape(b, lat.shifts, lat.taps, lat.hop)
        out = w[:, :, 0, :]
        for a in range(1, lat.taps):
            out = out + torch.roll(w[:, :, a, :], a, dims=1)
        return out.reshape(*lead, frame.n)
    out = torch.zeros(b, frame.n, dtype=alpha.dtype, device=alpha.device)
    out.scatter_add_(1, lat.pos.reshape(1, -1).expand(b, -1), seg.reshape(b, -1))
    return out.reshape(*lead, frame.n)


def analysis(x: torch.Tensor, frame: Frame) -> torch.Tensor:
    """Map a signal to coefficients, alpha = D* x."""
    x = x.to(frame.d.dtype)
    lat = frame.lattice
    if lat is None:
        return x @ frame.d.conj()
    lead = x.shape[:-1]
    flat = x.reshape(-1, frame.n)
    seg = flat[:, lat.pos.reshape(-1)].reshape(-1, lat.shifts, lat.window) * lat.win.conj()
    coeff = torch.fft.fft(seg, n=lat.n_freq, dim=-1) * lat.phase_ana
    return coeff.reshape(*lead, frame.n_atoms)


def is_tight(frame: Frame, atol: float = 1e-4) -> bool:
    """Check DD* = gamma I within tolerance."""
    gram = frame.d @ frame.d.conj().transpose(-1, -2)
    target = frame.gamma * torch.eye(frame.n, dtype=gram.dtype, device=gram.device)
    return torch.allclose(gram, target, atol=atol)
