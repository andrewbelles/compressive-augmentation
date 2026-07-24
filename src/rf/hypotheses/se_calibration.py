import torch

from dsp.operators import random_convolution, measure
from dsp.frames import gabor_frame
from dsp.recovery import oamp, reconstruct
from dsp.state_evolution import calibration_snr
from rf.signal_model import k_eff

# Layer IV Result 2: state evolution predicts realized OAMP recovery SNR of the CS pipeline.
# Accept: SE upper-bounds realized SNR and the gap closes with rho (fit in the admissible band).

GAMMA = 2


def _snr(x, xhat):
    num = x.abs().pow(2).sum(-1)
    den = (x - xhat).abs().pow(2).sum(-1).clamp_min(1e-12)
    return (10 * torch.log10(num / den)).mean().item()


def run(frames, meta, ratios, seed, device) -> list[dict]:
    n = frames.shape[-1]
    frame = gabor_frame(n, GAMMA, device=device)
    eps = k_eff(n) / frame.n_atoms
    x = frames / frames.abs().pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    records = []
    for rho in ratios:
        m = max(1, round(rho * n))
        op = random_convolution(n, m, torch.Generator(device=device).manual_seed(seed), device=device)
        realized = _snr(x, reconstruct(oamp(measure(x, op), op, frame), frame))
        se = calibration_snr(rho, 0.0, eps, GAMMA, n)
        records.append({
            "rho": rho,
            "m": m,
            "realized_snr": realized,
            "se_snr": se,
            "gap": se - realized,
        })
    return records
