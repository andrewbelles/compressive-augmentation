import torch

from dsp.frames import synthesis
from dsp.operators import random_convolution, measure, to_dense
from dsp.recovery import complex_soft_threshold
from rf.hypotheses._artifacts import normalize_power, snr_strata
from rf.signal_model import DEFAULT_DICTIONARY, build_dictionary

# Layer IV P1: recovery-SNR variance across operator draws is 11-17% lower for the structured
# random convolution than for an iid Gaussian at matched rho. Accept: the bootstrap CI on the
# variance ratio lies below 1.

DRAWS = 64
LAM = 0.02
BOOTSTRAP = 400
SUBSET = 32


def _snr(x, xhat):
    num = x.abs().pow(2).sum(-1)
    den = (x - xhat).abs().pow(2).sum(-1).clamp_min(1e-12)
    return 10 * torch.log10(num / den)


def _fista(y, a, lam, lip, iters=150):
    ah = a.conj().transpose(-1, -2)
    step = 1.0 / lip
    alpha = torch.zeros(*y.shape[:-1], a.shape[1], dtype=a.dtype, device=y.device)
    z, t = alpha.clone(), 1.0
    for _ in range(iters):
        grad = (z @ a.transpose(-1, -2) - y) @ ah.transpose(-1, -2)
        nxt = complex_soft_threshold(z - step * grad, lam * step)
        t2 = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        z = nxt + ((t - 1.0) / t2) * (nxt - alpha)
        alpha, t = nxt, t2
    return alpha


def _spectral_norm_sq(a, iters=20):
    v = torch.randn(a.shape[1], dtype=a.dtype, device=a.device)
    for _ in range(iters):
        v = a.conj().transpose(-1, -2) @ (a @ v)
        v = v / v.norm().clamp_min(1e-12)
    return ((a @ v).norm() ** 2 / v.norm().clamp_min(1e-12) ** 2).item()


def _ratio_ci(conv, gauss, gen):
    """Bootstrap CI on var(conv)/var(gauss) over operator draws."""
    c, g = torch.tensor(conv), torch.tensor(gauss)
    idx = torch.randint(0, c.numel(), (BOOTSTRAP, c.numel()), generator=gen)
    r = c[idx].var(dim=1) / g[idx].var(dim=1).clamp_min(1e-12)
    q = r.quantile(torch.tensor([0.025, 0.975]))
    return q[0].item(), q[1].item()


def run(frames, meta, ratios, seed, device, draws=DRAWS, dictionary=DEFAULT_DICTIONARY) -> list[dict]:
    n = frames.shape[-1]
    frame = build_dictionary(dictionary, n, device)
    x = normalize_power(frames)
    boot = torch.Generator().manual_seed(seed)
    records = []
    for snr_db, rows in snr_strata(meta, x.shape[0]).items():
        sel = x[torch.tensor(rows[:SUBSET], device=device)]
        for rho in ratios:
            m = max(1, round(rho * n))
            snr_conv, snr_gauss = [], []
            for j in range(draws):
                gen = torch.Generator(device=device).manual_seed(seed * 1000 + j)
                op = random_convolution(n, m, gen, device=device)
                a_conv = to_dense(op) @ frame.d
                gmat = torch.randn(m, n, dtype=torch.complex64, device=device, generator=gen)
                # match total measurement energy: ||Phi||_F^2 = n since Phi Phi* = (n/m) I
                gmat = gmat * (n ** 0.5 / gmat.norm())
                a_gauss = gmat @ frame.d
                lip_c = frame.gamma * n / m
                xt_c = synthesis(_fista(measure(sel, op), a_conv, LAM, lip_c), frame)
                xt_g = synthesis(_fista(sel @ gmat.transpose(-1, -2), a_gauss, LAM,
                                        _spectral_norm_sq(a_gauss)), frame)
                snr_conv.append(_snr(sel, xt_c).mean().item())
                snr_gauss.append(_snr(sel, xt_g).mean().item())
            vc = torch.tensor(snr_conv).var().item()
            vg = torch.tensor(snr_gauss).var().item()
            lo, hi = _ratio_ci(snr_conv, snr_gauss, boot)
            records.append({
                "snr": snr_db,
                "rho": rho,
                "m": m,
                "draws": draws,
                "var_conv": vc,
                "var_gauss": vg,
                "var_ratio": vc / max(vg, 1e-9),
                "var_ratio_lo": lo,
                "var_ratio_hi": hi,
                "mean_snr_conv": float(sum(snr_conv) / len(snr_conv)),
                "mean_snr_gauss": float(sum(snr_gauss) / len(snr_gauss)),
            })
    return records
