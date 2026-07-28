import torch

from dsp.frames import analysis
from dsp.operators import random_convolution
from dsp.recovery import sparse_code
from dsp.state_evolution import optimal_kappa
from rf.hypotheses._artifacts import normalize_power, snr_strata
from rf.hypotheses._matched import (
    awgn_view,
    backprojection_view,
    cs_view,
    distortion,
    dropout_view,
)
from rf.signal_model import DEFAULT_DICTIONARY, build_dictionary, k_eff

# H0: at matched distortion the CS view distribution is an isotropic Gaussian. Rejecting it is what
# makes the kernel a new analytic object rather than an expensive way to write down AWGN, whose
# kernel is already known. Accept: zeta_perp exceeds both the AWGN and the back-projection arm.

EPS = 1e-12
LAM = 0.05
BOOTSTRAP = 1000


def _inner(a, b):
    return (a.conj() * b).sum(-1).real


def _perp(e, x):
    """Remove the component along x, so shrinkage gain cannot masquerade as shared structure."""
    coef = _inner(x, e) / x.abs().pow(2).sum(-1).clamp_min(EPS)
    return e - coef.unsqueeze(-1) * x


def _zeta(e1, e2):
    return _inner(e1, e2) / (e1.norm(dim=-1) * e2.norm(dim=-1)).clamp_min(EPS)


def _support_fraction(e, alpha, frame, k):
    """Share of the error's coefficient energy on the signal's own top-k atoms."""
    coeffs = analysis(e, frame).abs().pow(2)
    idx = alpha.abs().topk(k, dim=-1).indices
    return coeffs.gather(-1, idx).sum(-1) / coeffs.sum(-1).clamp_min(EPS)


def _stats(x, v1, v2, alpha, frame, k):
    e1, e2 = v1 - x, v2 - x
    p1, p2 = _perp(e1, x), _perp(e2, x)
    return {
        "gain": (_inner(v1, x) / x.abs().pow(2).sum(-1).clamp_min(EPS)),
        "zeta": _zeta(e1, e2),
        "zeta_perp": _zeta(p1, p2),
        "support_fraction": _support_fraction(e1, alpha, frame, k),
        "achieved_distortion": distortion(x, v1),
    }


def _bootstrap_ci(a, b, gen):
    """95% CI on mean(a) - mean(b), resampling frames."""
    idx = torch.randint(0, a.numel(), (BOOTSTRAP, a.numel()), generator=gen)
    diff = a[idx].mean(dim=1) - b[idx].mean(dim=1)
    q = diff.quantile(torch.tensor([0.025, 0.975]))
    return q[0].item(), q[1].item()


def run(frames, meta, ratios, seed, device, dictionary=DEFAULT_DICTIONARY,
        measurement_snr=None) -> list[dict]:
    n = frames.shape[-1]
    frame = build_dictionary(dictionary, n, device)
    x = normalize_power(frames)
    eps = k_eff(n) / frame.n_atoms
    sigma = 0.0 if measurement_snr is None else 10.0 ** (-measurement_snr / 20.0)
    k_top = max(1, int(round(k_eff(n))))
    boot = torch.Generator().manual_seed(seed)
    records = []
    for snr_db, rows in snr_strata(meta, x.shape[0]).items():
        sel = x[torch.tensor(rows, device=device)]
        alpha, _ = sparse_code(sel, frame, LAM)
        for rho in ratios:
            m = max(1, round(rho * n))
            kappa = optimal_kappa(rho, sigma, eps, frame.gamma, n)
            gens = [torch.Generator(device=device).manual_seed(seed * 100 + j) for j in (0, 1)]
            ops = [random_convolution(n, m, g, device=device) for g in gens]
            cs = [cs_view(sel, ops[j], frame, kappa, measurement_snr, gens[j]) for j in (0, 1)]
            target = distortion(sel, cs[0])
            tmean = target.mean().item()

            arms = {"cs": cs}
            arms["awgn"] = [awgn_view(sel, target, gens[j]) for j in (0, 1)]
            arms["backprojection"] = [backprojection_view(sel, tmean, seed * 100 + j, device)[0]
                                      for j in (0, 1)]
            arms["dropout"] = [dropout_view(sel, alpha, frame, tmean, seed * 100 + j, device)[0]
                               for j in (0, 1)]

            stats = {a: _stats(sel, v[0], v[1], alpha, frame, k_top) for a, v in arms.items()}
            zp = {a: s["zeta_perp"].cpu() for a, s in stats.items()}
            lo_a, hi_a = _bootstrap_ci(zp["cs"], zp["awgn"], boot)
            lo_b, hi_b = _bootstrap_ci(zp["cs"], zp["backprojection"], boot)
            for arm, s in stats.items():
                records.append({
                    "arm": arm,
                    "dictionary": dictionary,
                    "snr": snr_db,
                    "measurement_snr": measurement_snr if measurement_snr is not None else float("inf"),
                    "rho": rho,
                    "m": m,
                    "kappa": kappa,
                    "target_distortion": tmean,
                    **{key: val.mean().item() for key, val in s.items()},
                    # contrasts are carried on every row so analysis can group freely
                    "zeta_perp_vs_awgn_lo": lo_a,
                    "zeta_perp_vs_awgn_hi": hi_a,
                    "zeta_perp_vs_bp_lo": lo_b,
                    "zeta_perp_vs_bp_hi": hi_b,
                })
    return records
