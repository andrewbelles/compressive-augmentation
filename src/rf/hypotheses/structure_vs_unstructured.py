import torch

from dsp.operators import random_convolution, measure, backproject, to_dense
from dsp.frames import gabor_frame, synthesis, analysis
from dsp.recovery import complex_soft_threshold
from dsp.state_evolution import beta_law_moments

# Layer IV Result 1/3 and P1: structured CS corruption differs from unstructured noise. Accept:
# back-projection residual matches the Beta law, and recovery-SNR draw-variance is lower for the
# random-convolution operator than for an iid Gaussian operator at matched rho.

GAMMA = 2
DRAWS = 6


def _snr(x, xhat):
    num = x.abs().pow(2).sum(-1)
    den = (x - xhat).abs().pow(2).sum(-1).clamp_min(1e-12)
    return 10 * torch.log10(num / den)


def _fista_dense(y, a, lam, iters=150):
    ah = a.conj().transpose(-1, -2)
    lip = torch.linalg.svdvals(a)[0].item() ** 2
    step = 1.0 / lip
    alpha = torch.zeros(*y.shape[:-1], a.shape[1], dtype=a.dtype, device=y.device)
    z, t = alpha.clone(), 1.0
    for _ in range(iters):
        grad = (z @ a.transpose(-1, -2) - y) @ ah.transpose(-1, -2)
        nxt = complex_soft_threshold(z - step * grad, lam * step)
        t2 = 0.5 * (1 + (1 + 4 * t * t) ** 0.5)
        z = nxt + ((t - 1) / t2) * (nxt - alpha)
        alpha, t = nxt, t2
    return alpha


def run(frames, meta, ratios, seed, device) -> list[dict]:
    n = frames.shape[-1]
    frame = gabor_frame(n, GAMMA, device=device)
    dmat = frame.d
    x = frames / frames.abs().pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    records = []
    for rho in ratios:
        m = max(1, round(rho * n))
        snr_conv, snr_gauss = [], []
        bp_fracs = []
        for j in range(DRAWS):
            g = torch.Generator(device=device).manual_seed(seed * 100 + j)
            op = random_convolution(n, m, g, device=device)
            a_conv = to_dense(op) @ dmat
            gmat = torch.randn(m, n, dtype=torch.complex64, device=device, generator=g)
            a_gauss = gmat @ dmat
            a_gauss = a_gauss * (torch.linalg.svdvals(a_conv)[0] / torch.linalg.svdvals(a_gauss)[0])
            xt_c = synthesis(_fista_dense(measure(x, op), a_conv, 0.02), frame)
            xt_g = synthesis(_fista_dense(x @ gmat.transpose(-1, -2), a_gauss, 0.02), frame)
            snr_conv.append(_snr(x, xt_c).mean().item())
            snr_gauss.append(_snr(x, xt_g).mean().item())
            e = x - backproject(measure(x, op), op)
            bp_fracs.append((e.abs().pow(2).sum(-1) / x.abs().pow(2).sum(-1)).mean().item())
        vc = torch.tensor(snr_conv).var().item()
        vg = torch.tensor(snr_gauss).var().item()
        records.append({
            "rho": rho,
            "m": m,
            "var_conv": vc,
            "var_gauss": vg,
            "var_ratio": vc / max(vg, 1e-9),
            "bp_frac_mean": float(sum(bp_fracs) / len(bp_fracs)),
            "bp_frac_pred": beta_law_moments(n, m)["mean"],
        })
    return records
