import torch

from dsp.operators import random_convolution, measure, backproject
from dsp.state_evolution import beta_law_moments
from rf.hypotheses._artifacts import normalize_power, snr_strata

# Layer IV Result 3 Eq. 9: back-projection uses no sparsity, so its residual energy fraction is
# Beta(N-m, m) with no assumption on x. A control on Phi, not evidence about the augmentation:
# it is solver-independent and falsifies "Phi is a Haar-random projection" directly.

DRAWS = 8


def run(frames, meta, ratios, seed, device) -> list[dict]:
    n = frames.shape[-1]
    x = normalize_power(frames)
    records = []
    for snr_db, rows in snr_strata(meta, x.shape[0]).items():
        sel = x[torch.tensor(rows, device=device)]
        for rho in ratios:
            m = max(1, round(rho * n))
            fracs = []
            for j in range(DRAWS):
                gen = torch.Generator(device=device).manual_seed(seed * 1000 + j)
                op = random_convolution(n, m, gen, device=device)
                e = sel - backproject(measure(sel, op), op)
                fracs.append(e.abs().pow(2).sum(-1) / sel.abs().pow(2).sum(-1))
            f = torch.cat(fracs)
            law = beta_law_moments(n, m)
            records.append({
                "snr": snr_db,
                "rho": rho,
                "m": m,
                "bp_frac_mean": f.mean().item(),
                "bp_frac_pred": law["mean"],
                "bp_frac_var": f.var().item(),
                "bp_var_pred": law["var"],
                "snr_bp": (-10 * torch.log10(f.mean().clamp_min(1e-12))).item(),
            })
    return records
