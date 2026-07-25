import torch

from dsp.cumulants import cumulant_features, cumulant_distance
from dsp.operators import random_convolution, measure
from dsp.recovery import oamp, reconstruct
from rf.hypotheses._artifacts import mods_at, normalize_power, snr_strata
from rf.signal_model import DEFAULT_DICTIONARY, build_dictionary

# Layer V Prop 5 Eq. 11: the CS view preserves the label-discriminating cumulants inside the band.
# Class means are taken within an SNR stratum, since pooling collapses every class onto the noise
# centroid. Accept: mean distortion stays below the class separation at large rho.


def _margins(feats, mods, device):
    """Minimum and 10th-percentile pairwise distance between class means."""
    keys = sorted(set(mods))
    if len(keys) < 2:
        return float("nan"), float("nan")
    index: dict[str, list[int]] = {k: [] for k in keys}
    for i, mod in enumerate(mods):
        index[mod].append(i)
    stacked = torch.stack([feats[torch.tensor(index[k], device=device)].mean(0) for k in keys])
    dist = torch.cdist(stacked, stacked)
    iu = torch.triu_indices(len(keys), len(keys), offset=1, device=device)
    pairs = dist[iu[0], iu[1]]
    return pairs.min().item(), pairs.quantile(0.1).item()


def run(frames, meta, ratios, seed, device, dictionary=DEFAULT_DICTIONARY) -> list[dict]:
    n = frames.shape[-1]
    frame = build_dictionary(dictionary, n, device)
    x = normalize_power(frames)
    records = []
    for snr_db, rows in snr_strata(meta, x.shape[0]).items():
        sel = x[torch.tensor(rows, device=device)]
        mods = mods_at(meta, rows)
        delta_min, delta_q10 = _margins(cumulant_features(sel), mods, device)
        for rho in ratios:
            m = max(1, round(rho * n))
            gen = torch.Generator(device=device).manual_seed(seed)
            op = random_convolution(n, m, gen, device=device)
            xt = reconstruct(oamp(measure(sel, op), op, frame), frame)
            dist = cumulant_distance(sel, xt)
            records.append({
                "snr": snr_db,
                "rho": rho,
                "m": m,
                "mean_cumulant_dist": dist.mean().item(),
                "median_cumulant_dist": dist.median().item(),
                "delta_min": delta_min,
                "delta_q10": delta_q10,
            })
    return records
