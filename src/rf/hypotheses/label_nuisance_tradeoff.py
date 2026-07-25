import math

import torch

from common.statistics import scatter_ratio
from dsp.cumulants import cumulant_features
from dsp.operators import random_convolution, measure
from dsp.recovery import oamp, reconstruct
from rf.hypotheses._artifacts import mods_at, normalize_power, snr_strata
from rf.signal_model import DEFAULT_DICTIONARY, build_dictionary

# Layer V Result 4 measured rather than derived: an augmentation must keep the label and destroy
# the nuisance. Fidelity alone is maximized at rho -> 1 by the identity map, so retention is scored
# against view diversity. Accept: a rho range with high retention, non-vanishing diversity, C < 1.


def _view(x, op, frame):
    return reconstruct(oamp(measure(x, op), op, frame), frame)


def _rel_dist(a, b):
    return (a - b).abs().pow(2).sum(-1) / a.abs().pow(2).sum(-1).clamp_min(1e-12)


def run(frames, meta, ratios, seed, device, dictionary=DEFAULT_DICTIONARY) -> list[dict]:
    n = frames.shape[-1]
    frame = build_dictionary(dictionary, n, device)
    x = normalize_power(frames)
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
            ops = [random_convolution(n, m, torch.Generator(device=device).manual_seed(seed * 100 + j),
                                      device=device) for j in (0, 1)]
            v1, v2 = _view(sel, ops[0], frame), _view(sel, ops[1], frame)
            view_sep = scatter_ratio(cumulant_features(v1), mods)
            collapse = _rel_dist(v1, _view(nuis, ops[0], frame)).mean() / base_nuis.clamp_min(1e-12)
            records.append({
                "snr": snr_db,
                "rho": rho,
                "m": m,
                "label_retention": view_sep / base_sep if base_sep else float("nan"),
                "view_sep": view_sep,
                "base_sep": base_sep,
                "view_diversity": _rel_dist(v1, v2).mean().item(),
                "fidelity": _rel_dist(sel, v1).mean().item(),
                "nuisance_collapse": collapse.item(),
            })
    return records
