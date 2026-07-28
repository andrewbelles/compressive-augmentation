import math

import torch

from common.statistics import scatter_ratio
from dsp.cumulants import cumulant_features
from dsp.operators import random_convolution, measure
from dsp.recovery import oamp, reconstruct
from rf.hypotheses._artifacts import mods_at, normalize_power, snr_strata
from rf.hypotheses._matched import awgn_view, distortion
from dsp.state_evolution import optimal_kappa
from rf.signal_model import DEFAULT_DICTIONARY, build_dictionary, k_eff

# Layer V Result 4 measured rather than derived: an augmentation must keep the label and destroy
# the nuisance. Fidelity alone is maximized at rho -> 1 by the identity map, so retention is scored
# against view diversity. Accept: a rho range with high retention, non-vanishing diversity, C < 1.


def _view(x, op, frame, kappa):
    return reconstruct(oamp(measure(x, op), op, frame, kappa=kappa), frame)


def _rel_dist(a, b):
    return (a - b).abs().pow(2).sum(-1) / a.abs().pow(2).sum(-1).clamp_min(1e-12)


def run(frames, meta, ratios, seed, device, dictionary=DEFAULT_DICTIONARY,
        measurement_snr=None) -> list[dict]:
    n = frames.shape[-1]
    frame = build_dictionary(dictionary, n, device)
    x = normalize_power(frames)
    eps = k_eff(n) / frame.n_atoms
    sigma = 0.0 if measurement_snr is None else 10.0 ** (-measurement_snr / 20.0)
    k = torch.arange(n, device=device)
    records = []
    for snr_db, rows in snr_strata(meta, x.shape[0]).items():
        sel = x[torch.tensor(rows, device=device)]
        mods = mods_at(meta, rows)
        base_sep = scatter_ratio(cumulant_features(sel), mods)
        nuis = sel * torch.exp(2j * math.pi * 3.5 * k / n)
        base_nuis = _rel_dist(sel, nuis).mean()
        for rho in ratios:
            m = max(1, round(rho * n))
            gens = [torch.Generator(device=device).manual_seed(seed * 100 + j) for j in (0, 1)]
            ops = [random_convolution(n, m, g, device=device) for g in gens]
            kappa = optimal_kappa(rho, sigma, eps, frame.gamma, n)
            v1, v2 = _view(sel, ops[0], frame, kappa), _view(sel, ops[1], frame, kappa)
            collapse = _rel_dist(v1, _view(nuis, ops[0], frame, kappa)).mean() / base_nuis.clamp_min(1e-12)
            # AWGN at the same distortion: rho is a parameter, not a strength, so retention is only
            # comparable at matched view diversity
            target = distortion(sel, v1)
            a1, a2 = (awgn_view(sel, target, gens[j]) for j in (0, 1))
            for arm, w1, w2 in (("cs", v1, v2), ("awgn", a1, a2)):
                sep = scatter_ratio(cumulant_features(w1), mods)
                records.append({
                    "arm": arm,
                    "snr": snr_db,
                    "measurement_snr": measurement_snr if measurement_snr is not None else float("inf"),
                    "rho": rho,
                    "m": m,
                    "label_retention": sep / base_sep if base_sep else float("nan"),
                    "view_sep": sep,
                    "base_sep": base_sep,
                    "view_diversity": _rel_dist(w1, w2).mean().item(),
                    "fidelity": _rel_dist(sel, w1).mean().item(),
                    "nuisance_collapse": collapse.item() if arm == "cs" else float("nan"),
                })
    return records
