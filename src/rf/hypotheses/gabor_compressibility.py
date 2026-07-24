import torch

from dsp.frames import gabor_frame, dft_frame, synthesis, analysis
from dsp.recovery import complex_soft_threshold
from rf.signal_model import k_eff

# Layer II Eq. 2: real frames admit a sparse synthesis in the Gabor frame near k_eff, while the DFT
# representation is a contiguous band (degenerate). Accept: Gabor sparse-code k90 well below the
# DFT support and near k_eff; DFT energy concentrated in a contiguous band.

GAMMA = 2


def _sparse_code(x, frame, lam=0.05, iters=150):
    step = 1.0 / frame.gamma  # 1/L, L = ||D||^2 = gamma
    alpha = torch.zeros(*x.shape[:-1], frame.n_atoms, dtype=frame.d.dtype, device=x.device)
    z, t = alpha.clone(), 1.0
    for _ in range(iters):
        grad = analysis(synthesis(z, frame) - x, frame)
        nxt = complex_soft_threshold(z - step * grad, lam * step)
        t2 = 0.5 * (1 + (1 + 4 * t * t) ** 0.5)
        z = nxt + ((t - 1) / t2) * (nxt - alpha)
        alpha, t = nxt, t2
    return alpha


def _k_energy(coeffs, frac=0.9):
    p = coeffs.abs().pow(2)
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
    cum = p.sort(dim=-1, descending=True).values.cumsum(-1)
    return (cum < frac).sum(-1).float() + 1.0


def _dft_span(coeffs, frac=0.9):
    p = coeffs.abs().pow(2)
    order = p.argsort(dim=-1, descending=True)
    cum = p.gather(-1, order).cumsum(-1) / p.sum(-1, keepdim=True).clamp_min(1e-12)
    kept = cum < frac
    spans = []
    for i in range(coeffs.shape[0]):
        idx = order[i][kept[i]]
        spans.append(1.0 if idx.numel() < 2 else ((idx.max() - idx.min()).item() + 1) / idx.numel())
    return torch.tensor(spans, device=coeffs.device)


def run(frames, meta, ratios, seed, device) -> list[dict]:
    n = frames.shape[-1]
    gabor = gabor_frame(n, GAMMA, device=device)
    dft = dft_frame(n, device=device)
    pred = k_eff(n)
    mods = meta.get("mod", [""] * frames.shape[0])
    snrs = meta.get("snr", [0] * frames.shape[0])
    groups: dict[tuple, list[int]] = {}
    for i, key in enumerate(zip(mods, snrs)):
        groups.setdefault(key, []).append(i)
    records = []
    for (mod, snr), rows in sorted(groups.items()):
        sel = frames[torch.tensor(rows, device=device)]
        gcode = _sparse_code(sel, gabor)
        dcoef = analysis(sel, dft)
        records.append({
            "mod": mod,
            "snr": int(snr),
            "k90_gabor": _k_energy(gcode).mean().item(),
            "k90_dft": _k_energy(dcoef).mean().item(),
            "k_eff_pred": pred,
            "dft_span_ratio": _dft_span(dcoef).mean().item(),
        })
    return records
