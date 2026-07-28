import torch

from common.statistics import scatter_ratio
from dsp.frames import analysis
from dsp.recovery import sparse_code
from rf.hypotheses._artifacts import mods_at, normalize_power, snr_strata
from rf.signal_model import CONFUSABLE_MODS, DICTIONARIES, build_dictionary, k_eff

# Layer II Eq. 2: frames admit a sparse synthesis whose support carries the modulation class.
# Accept: a dictionary reaches k near k_eff and its coefficient support separates classes; the DFT
# arm is degenerate because its support is shared across classes.

LAM = 0.05
MAX_ITERS = 400


def _k_energy(coeffs, frac):
    p = coeffs.abs().pow(2)
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
    cum = p.sort(dim=-1, descending=True).values.cumsum(-1)
    return (cum < frac).sum(-1).float() + 1.0


def _circular_span(coeffs, frac=0.9):
    """Smallest circular arc covering the kept support, over its size; 1.0 is contiguous."""
    n = coeffs.shape[-1]
    p = coeffs.abs().pow(2)
    order = p.argsort(dim=-1, descending=True)
    cum = p.gather(-1, order).cumsum(-1) / p.sum(-1, keepdim=True).clamp_min(1e-12)
    kept = cum < frac
    spans = []
    for i in range(coeffs.shape[0]):
        idx = order[i][kept[i]].sort().values
        if idx.numel() < 2:
            spans.append(1.0)
            continue
        gaps = torch.diff(torch.cat([idx, idx[:1] + n]))
        spans.append(((n - gaps.max()).item() + 1) / idx.numel())
    return torch.tensor(spans, device=coeffs.device)


def run(frames, meta, ratios, seed, device) -> list[dict]:
    n = frames.shape[-1]
    x = normalize_power(frames)
    pred = k_eff(n)
    records = []
    for name in DICTIONARIES:
        frame = build_dictionary(name, n, device)
        for snr, rows in snr_strata(meta, x.shape[0]).items():
            sel = x[torch.tensor(rows, device=device)]
            mods = mods_at(meta, rows)
            coeffs, iters = sparse_code(sel, frame, LAM, max_iters=MAX_ITERS)
            k90, k99 = _k_energy(coeffs, 0.9), _k_energy(coeffs, 0.99)
            span = _circular_span(analysis(sel, frame) if name == "dft" else coeffs)
            sep = scatter_ratio(coeffs.abs(), mods)
            # sub-hypothesis: the 15 RRC-PSD-sharing classes, where the label actually lives
            conf = [j for j, mm in enumerate(mods) if mm in CONFUSABLE_MODS]
            sep_conf = float("nan")
            if len(conf) > 1:
                rows_conf = torch.tensor(conf, device=device)
                sep_conf = scatter_ratio(coeffs[rows_conf].abs(), [mods[j] for j in conf])
            by_mod: dict[str, list[int]] = {}
            for j, mod in enumerate(mods):
                by_mod.setdefault(mod, []).append(j)
            for mod, js in sorted(by_mod.items()):
                sub = torch.tensor(js, device=device)
                records.append({
                    "dictionary": name,
                    "snr": snr,
                    "mod": mod,
                    "n_atoms": frame.n_atoms,
                    "k90": k90[sub].mean().item(),
                    "k99": k99[sub].mean().item(),
                    "k90_over_n": k90[sub].mean().item() / n,
                    "k90_over_d": k90[sub].mean().item() / frame.n_atoms,
                    "k_eff_pred": pred,
                    "k99_over_k_eff": k99[sub].mean().item() / pred,
                    "circular_span_ratio": span[sub].mean().item(),
                    "class_scatter_ratio": sep,
                    "class_scatter_confusable": sep_conf,
                    "fista_iters": iters,
                    "fista_converged": iters < MAX_ITERS,
                })
    return records
