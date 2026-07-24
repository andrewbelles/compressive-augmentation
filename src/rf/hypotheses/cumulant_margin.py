import torch

from dsp.operators import random_convolution, measure
from dsp.frames import gabor_frame
from dsp.recovery import oamp, reconstruct
from dsp.cumulants import cumulant_features, cumulant_distance

# Layer V Prop 5 Eq. 11: the CS view preserves the label-discriminating cumulants inside the band.
# Accept: mean cumulant distance stays below the class margin Delta_min for large rho and grows as
# rho falls.

GAMMA = 2


def run(frames, meta, ratios, seed, device) -> list[dict]:
    n = frames.shape[-1]
    frame = gabor_frame(n, GAMMA, device=device)
    x = frames
    mods = meta.get("mod", [""] * frames.shape[0])
    feats = cumulant_features(x)
    class_means = {}
    for mod in sorted(set(mods)):
        rows = torch.tensor([i for i, mm in enumerate(mods) if mm == mod], device=device)
        class_means[mod] = feats[rows].mean(0)
    stacked = torch.stack(list(class_means.values())) if len(class_means) > 1 else None
    delta_min = float("nan")
    if stacked is not None and stacked.shape[0] > 1:
        dists = torch.cdist(stacked, stacked)
        dists = dists + torch.eye(stacked.shape[0], device=device) * 1e9
        delta_min = dists.min().item()
    records = []
    for rho in ratios:
        m = max(1, round(rho * n))
        op = random_convolution(n, m, torch.Generator(device=device).manual_seed(seed), device=device)
        xt = reconstruct(oamp(measure(x, op), op, frame), frame)
        records.append({
            "rho": rho,
            "m": m,
            "mean_cumulant_dist": cumulant_distance(x, xt).mean().item(),
            "delta_min": delta_min,
        })
    return records
