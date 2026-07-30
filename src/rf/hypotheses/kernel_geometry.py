import math

import torch

from common.statistics import scatter_ratio
from dsp.cumulants import cumulant_features
from dsp.frames import analysis
from dsp.operators import random_convolution
from dsp.recovery import sparse_code
from dsp.state_evolution import optimal_kappa
from rf.augment import dither
from rf.hypotheses._artifacts import mods_at, normalize_power, snr_strata
from rf.hypotheses._matched import (
    awgn_view,
    backprojection_view,
    cs_mmse_view,
    cs_view,
    distortion,
    dropout_view,
    shrunk_view,
)
from rf.signal_model import (
    DEFAULT_DICTIONARY,
    build_dictionary,
    dither_sigma,
    frame_sparsity,
    k_eff,
)

# H0: at matched distortion the CS view distribution is an isotropic Gaussian. Rejecting it is what
# makes the kernel a new analytic object rather than an expensive way to write down AWGN, whose
# kernel is already known. Accept: zeta_perp exceeds both the AWGN and the back-projection arm.
#
# The same two views carry the Layer V trade-off, so retention and diversity are recorded here and
# label_nuisance_tradeoff is scored from these artifacts rather than recomputing them.

EPS = 1e-12
LAM = 0.05
BOOTSTRAP = 1000
NUISANCE_CFO = 3.5


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


def _stats(x, v1, v2, alpha, frame, k, mods, base_sep):
    e1, e2 = v1 - x, v2 - x
    p1, p2 = _perp(e1, x), _perp(e2, x)
    sep = scatter_ratio(cumulant_features(v1), mods)
    return {
        "gain": _inner(v1, x) / x.abs().pow(2).sum(-1).clamp_min(EPS),
        "zeta": _zeta(e1, e2),
        "zeta_perp": _zeta(p1, p2),
        "support_fraction": _support_fraction(e1, alpha, frame, k),
        "achieved_distortion": distortion(x, v1),
        "view_diversity": distortion(v1, v2),
        "view_sep": torch.full_like(distortion(x, v1), sep),
        "label_retention": torch.full_like(distortion(x, v1),
                                           sep / base_sep if base_sep else float("nan")),
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
    eps = frame_sparsity(frame, n)
    sigma = dither_sigma(measurement_snr)
    k_top = max(1, int(round(k_eff(n))))
    kgrid = torch.arange(n, device=device)
    boot = torch.Generator().manual_seed(seed)
    records = []
    for snr_db, rows in snr_strata(meta, x.shape[0]).items():
        sel = x[torch.tensor(rows, device=device)]
        mods = mods_at(meta, rows)
        alpha, _ = sparse_code(sel, frame, LAM)
        base_sep = scatter_ratio(cumulant_features(sel), mods)
        nuis = sel * torch.exp(2j * math.pi * NUISANCE_CFO * kgrid / n)
        base_nuis = distortion(sel, nuis).mean()
        for rho in ratios:
            m = max(1, round(rho * n))
            kappa = optimal_kappa(rho, sigma, eps, frame.gamma, n)
            gens = [torch.Generator(device=device).manual_seed(seed * 100 + j) for j in (0, 1)]
            ops = [random_convolution(n, m, g, device=device) for g in gens]
            noise = [dither(sel, ops[j], measurement_snr, gens[j]) for j in (0, 1)]
            cs = [cs_view(sel, ops[j], frame, kappa, noise[j]) for j in (0, 1)]
            target = distortion(sel, cs[0])
            tmean = target.mean().item()

            arms = {"cs": cs}
            # the debiased refit is the better estimator but removes shrinkage structure from the
            # error, which is what zeta_perp measures; carrying both lets analysis compare the two
            # at matched distortion instead of at matched rho
            arms["cs_shrunk"] = [shrunk_view(sel, ops[j], frame, kappa, noise[j]) for j in (0, 1)]
            # the matched denoiser drops nothing, so this arm separates anisotropy that survives a
            # smooth estimator from anisotropy that is soft-threshold shrinkage structure
            arms["cs_mmse"] = [cs_mmse_view(sel, ops[j], frame, eps, noise[j]) for j in (0, 1)]
            arms["awgn"] = [awgn_view(sel, target, gens[j]) for j in (0, 1)]
            arms["backprojection"] = [backprojection_view(sel, tmean, seed * 100 + j, device)[0]
                                      for j in (0, 1)]
            arms["dropout"] = [dropout_view(sel, alpha, frame, tmean, seed * 100 + j, device)[0]
                               for j in (0, 1)]

            # the nuisance view reuses the base view's noise, so the ratio isolates the nuisance
            # instead of picking up an independent 2 sigma^2
            nuis_view = cs_view(nuis, ops[0], frame, kappa, noise[0])
            collapse = (distortion(cs[0], nuis_view).mean() / base_nuis.clamp_min(EPS)).item()

            stats = {a: _stats(sel, v[0], v[1], alpha, frame, k_top, mods, base_sep)
                     for a, v in arms.items()}
            zp = {a: s["zeta_perp"].cpu() for a, s in stats.items()}
            lo_a, hi_a = _bootstrap_ci(zp["cs"], zp["awgn"], boot)
            lo_b, hi_b = _bootstrap_ci(zp["cs"], zp["backprojection"], boot)
            # drawn after the two gate contrasts so their resampling stream is unchanged
            own = {a: _bootstrap_ci(zp[a], zp["awgn"], boot) for a in stats}
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
                    "base_sep": base_sep,
                    "nuisance_collapse": collapse if arm == "cs" else float("nan"),
                    **{key: val.mean().item() for key, val in s.items()},
                    # contrasts are carried on every row so analysis can group freely
                    "zeta_perp_vs_awgn_lo": lo_a,
                    "zeta_perp_vs_awgn_hi": hi_a,
                    "zeta_perp_vs_bp_lo": lo_b,
                    "zeta_perp_vs_bp_hi": hi_b,
                    # the row's own arm against awgn, so every arm carries its own isotropy test
                    "arm_vs_awgn_lo": own[arm][0],
                    "arm_vs_awgn_hi": own[arm][1],
                })
    return records
