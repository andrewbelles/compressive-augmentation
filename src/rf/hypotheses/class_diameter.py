import torch

from common.statistics import class_margins, scatter_ratio
from dsp.cumulants import cumulant_distance, cumulant_features
from dsp.operators import random_convolution
from dsp.recovery import sparse_code
from dsp.state_evolution import kappa_pinned, optimal_kappa
from rf.augment import dither
from rf.hypotheses._artifacts import mods_at, normalize_power, snr_strata
from rf.hypotheses._matched import (
    awgn_view,
    backprojection_view,
    cs_mmse_view,
    cs_view,
    distortion,
    dropout_view,
    retained_energy,
    shrunk_view,
)
from rf.signal_model import (
    DEFAULT_DICTIONARY,
    build_dictionary,
    dither_sigma,
    frame_sparsity,
    k_eff,
)

# Pair geometry of one modulation under augmentation: kernel_geometry contrasts two views of one
# frame, this contrasts views of two frames drawn through independent operators.
#
# diameter_ratio reduces to g^2 + eps/(1 - corr) for a view g x + e, which equals retained_energy
# whenever same-class frames are uncorrelated, so it is a gain statistic and carries no verdict.
# The contraction statistic is class_alignment minus class_alignment_between: each term divides the
# gain out and is mean-zero under isotropic error at any energy, and their difference is zero for a
# map that has no directional preference. No verdict scores these; the criterion is not registered.

EPS = 1e-12
LAM = 0.05
QUANTILE = 0.9
BOOTSTRAP = 1000
MIN_PER_HALF = 2
BETWEEN_CLASSES = 2


def _halves(mods, device):
    """Split each modulation's frames alternately, so both halves carry every class in balance."""
    sides = ([], [])
    seen: dict[str, int] = {}
    for i, mod in enumerate(mods):
        k = seen.get(mod, 0)
        sides[k % 2].append(i)
        seen[mod] = k + 1
    return [torch.tensor(s, dtype=torch.long, device=device) for s in sides]


def _members(mods, idx, device):
    """Local positions of each modulation inside the two halves, dropping classes too thin to pair."""
    out: dict[str, tuple[list[int], list[int]]] = {}
    for side, rows in enumerate(idx):
        for pos, i in enumerate(rows.tolist()):
            out.setdefault(mods[i], ([], []))[side].append(pos)
    return {mod: tuple(torch.tensor(s, dtype=torch.long, device=device) for s in sides)
            for mod, sides in out.items()
            if min(len(sides[0]), len(sides[1])) >= MIN_PER_HALF}


def _pair_energy(a, b):
    """Squared distance over the cross product of two disjoint sets."""
    # raw energy, not distortion: a collapsing view shrinks its own denominator, so a relative
    # distance would turn the degenerate case into 0/0 instead of a ratio that reads as collapse
    return (a.unsqueeze(1) - b.unsqueeze(0)).abs().pow(2).sum(-1).reshape(-1)


def _diameter(d):
    return d.quantile(QUANTILE).item()


def _partners(keys, mod):
    """A fixed small set of other classes, so the between-class term does not cost a full product."""
    i = keys.index(mod)
    return [keys[(i + j) % len(keys)] for j in range(1, min(BETWEEN_CLASSES, len(keys) - 1) + 1)]


def _alignment(xa, xb, va, xb_view):
    """Share of the within-class direction the view removes, over the same cross product."""
    # at matched distortion every arm carries the same gain and error energy, so a magnitude
    # statistic cannot separate them; projecting the error onto x_a - x_b is mean-zero under
    # isotropic error at any energy and positive only when the error follows that direction
    d = xa.unsqueeze(1) - xb.unsqueeze(0)
    e = (va - xa).unsqueeze(1) - (xb_view - xb).unsqueeze(0)
    return (-(d.conj() * e).sum(-1).real
            / d.abs().pow(2).sum(-1).clamp_min(EPS)).reshape(-1)


def _bootstrap_ratio(aug, clean, gen):
    """95% CI on the quantile ratio, resampling the pair set both diameters are read over."""
    a, c = aug.cpu(), clean.cpu()
    idx = torch.randint(0, a.numel(), (BOOTSTRAP, a.numel()), generator=gen)
    r = a[idx].quantile(QUANTILE, dim=1) / c[idx].quantile(QUANTILE, dim=1).clamp_min(EPS)
    q = r.quantile(torch.tensor([0.025, 0.975]))
    return q[0].item(), q[1].item()


def run(frames, meta, ratios, seed, device, dictionary=DEFAULT_DICTIONARY,
        measurement_snr=None) -> list[dict]:
    n = frames.shape[-1]
    frame = build_dictionary(dictionary, n, device)
    x = normalize_power(frames)
    eps = frame_sparsity(frame, n)
    sigma = dither_sigma(measurement_snr)
    boot = torch.Generator().manual_seed(seed)
    records = []
    for snr_db, rows in snr_strata(meta, x.shape[0]).items():
        sel = x[torch.tensor(rows, device=device)]
        mods = mods_at(meta, rows)
        # the margin is a between-class quantity read once over the stratum and repeated on its
        # rows; a single-modulation row has no pair to take it over
        delta_min, delta_q10 = class_margins(cumulant_features(sel), mods)
        idx = _halves(mods, device)
        parts = [sel[i] for i in idx]
        alphas = [sparse_code(p, frame, LAM)[0] for p in parts]
        members = _members(mods, idx, device)
        partners = {mod: _partners(sorted(members), mod) for mod in members}
        # the label is the outcome the contraction is meant to buy, so it is read on these same rows
        # rather than joined from kernel_geometry across two sweeps of different rho grids
        half_mods = [mods[i] for j in (0, 1) for i in idx[j].tolist()]
        base_sep = scatter_ratio(cumulant_features(sel), mods)
        clean = {mod: _pair_energy(parts[0][a], parts[1][b]) for mod, (a, b) in members.items()}
        clean_d = {mod: _diameter(d) for mod, d in clean.items()}
        for rho in ratios:
            m = max(1, round(rho * n))
            kappa = optimal_kappa(rho, sigma, eps, frame.gamma, n)
            # one operator and one dither per half, so the two members of any pair pass through
            # independent randomness exactly as two deployed views would
            gens = [torch.Generator(device=device).manual_seed(seed * 100 + j) for j in (0, 1)]
            ops = [random_convolution(n, m, g, device=device) for g in gens]
            noise = [dither(parts[j], ops[j], measurement_snr, gens[j]) for j in (0, 1)]
            cs = [cs_view(parts[j], ops[j], frame, kappa, noise[j]) for j in (0, 1)]
            # each half carries its own matched target: one pooled scalar would leave the null arms
            # matched to a distortion neither half has
            target = [distortion(parts[j], cs[j]) for j in (0, 1)]
            tmean = [t.mean().item() for t in target]

            arms = {"cs": cs}
            arms["cs_shrunk"] = [shrunk_view(parts[j], ops[j], frame, kappa, noise[j])
                                 for j in (0, 1)]
            arms["cs_mmse"] = [cs_mmse_view(parts[j], ops[j], frame, eps, noise[j]) for j in (0, 1)]
            arms["awgn"] = [awgn_view(parts[j], target[j], gens[j]) for j in (0, 1)]
            arms["backprojection"] = [backprojection_view(parts[j], tmean[j], seed * 100 + j,
                                                          device)[0] for j in (0, 1)]
            arms["dropout"] = [dropout_view(parts[j], alphas[j], frame, tmean[j], seed * 100 + j,
                                            device)[0] for j in (0, 1)]

            for arm, views in arms.items():
                sep = scatter_ratio(cumulant_features(torch.cat(views)), half_mods)
                energy = [retained_energy(parts[j], views[j]) for j in (0, 1)]
                dist = [distortion(parts[j], views[j]) for j in (0, 1)]
                cdist = [cumulant_distance(parts[j], views[j]) for j in (0, 1)]
                for mod, (a, b) in members.items():
                    aug = _pair_energy(views[0][a], views[1][b])
                    lo, hi = _bootstrap_ratio(aug, clean[mod], boot)
                    cd = torch.cat([cdist[0][a], cdist[1][b]])
                    records.append({
                        "arm": arm,
                        "dictionary": dictionary,
                        "snr": snr_db,
                        "mod": mod,
                        "measurement_snr": measurement_snr if measurement_snr is not None
                        else float("inf"),
                        "rho": rho,
                        "m": m,
                        "m_over_k_eff": m / k_eff(n),
                        "kappa": kappa,
                        "kappa_pinned": kappa_pinned(rho, sigma, eps, frame.gamma, n),
                        # over the same frames as the achieved column, so the two can be compared
                        # to audit the matching that the null arms rest on
                        "target_distortion": torch.cat([target[0][a],
                                                        target[1][b]]).mean().item(),
                        "achieved_distortion": torch.cat([dist[0][a], dist[1][b]]).mean().item(),
                        "retained_energy": torch.cat([energy[0][a],
                                                      energy[1][b]]).mean().item(),
                        "n_pairs": aug.numel(),
                        "aug_diameter": _diameter(aug),
                        "clean_diameter": clean_d[mod],
                        "diameter_ratio": _diameter(aug) / max(clean_d[mod], EPS),
                        "diameter_ratio_lo": lo,
                        "diameter_ratio_hi": hi,
                        "class_alignment": _alignment(parts[0][a], parts[1][b],
                                                      views[0][a], views[1][b]).mean().item(),
                        # the same statistic on pairs that cross classes, kept separate so a reader
                        # sees which term moved; a direction-agnostic arm removes the same share of
                        # both and only a projection preferring the within-class direction splits them
                        "class_alignment_between": sum(
                            _alignment(parts[0][a], parts[1][members[o][1]],
                                       views[0][a], views[1][members[o][1]]).mean().item()
                            for o in partners[mod]) / len(partners[mod]),
                        "view_sep": sep,
                        # carried beside the ratio because base_sep spans orders of magnitude across
                        # strata, and a retention read against a small one is two noise floors
                        "base_sep": base_sep,
                        "label_retention": sep / base_sep if base_sep else float("nan"),
                        "mean_cumulant_dist": cd.mean().item(),
                        "median_cumulant_dist": cd.median().item(),
                        "delta_min": delta_min,
                        "delta_q10": delta_q10,
                    })
    return records
