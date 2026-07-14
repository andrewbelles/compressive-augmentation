from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from csmath.dictlearn import ksvd, lc_ksvd
from rf.frames import load_complex_frames, load_manifest
from rf.nuisance import orbit_augment
from rf.preprocess.manifests import MOD_CLASSES

EPS = 1e-12
VARIANTS = ("v0", "v1", "v2", "v3")


@dataclass
class Dictionary:
    """A multi-class SRC dictionary: unit-norm complex atoms with class labels."""
    atoms:       torch.Tensor   # [1024, K_total] complex64, unit-norm columns
    atom_labels: torch.Tensor   # [K_total] long, index into MOD_CLASSES
    variant:     str


def _normalize_columns(A: torch.Tensor) -> torch.Tensor:
    return A / A.norm(dim=0, keepdim=True).clamp_min(EPS)


def build_gallery(
    hdf5_path: Path,
    manifest_dir: Path,
    per_class: int,
    snr_min: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample per_class clean (snr >= snr_min) training-split exemplars per modulation.

    Returns (frames [P, 1024] unit-norm complex64, labels [P] long). The
    per-class RNG is keyed on (seed, class index) so galleries are deterministic.
    """
    manifest = load_manifest(manifest_dir, "training")
    manifest = manifest[manifest["snr"] >= snr_min]
    indices, labels = [], []
    for mod_i, mod in enumerate(MOD_CLASSES):
        pool = manifest[manifest["mod"] == mod]["frame_idx"].sort_values().to_numpy()
        if len(pool) == 0:
            raise ValueError(f"no training frames with snr >= {snr_min} for class {mod}")
        rng = np.random.default_rng([seed, mod_i])
        n   = min(per_class, len(pool))
        indices.append(np.sort(rng.choice(pool, size=n, replace=False)))
        labels.extend([mod_i] * n)
    x = load_complex_frames(hdf5_path, np.concatenate(indices), device, normalize=True)
    return x, torch.tensor(labels, dtype=torch.long, device=device)


def build_v0(gallery_x: torch.Tensor, labels: torch.Tensor) -> Dictionary:
    """V0: raw exemplar atoms (one atom per gallery frame)."""
    return Dictionary(_normalize_columns(gallery_x.T.clone()), labels.clone(), "v0")


def build_v1(
    gallery_x: torch.Tensor,
    labels: torch.Tensor,
    n_orbit: int,
    max_eps: float,
    max_shift: int,
    gen: torch.Generator,
) -> Dictionary:
    """V1: exemplar atoms expanded over the nuisance orbit (copy 0 = identity).

    Callers pass a gallery of atoms_per_class // n_orbit base exemplars per class
    so V0 and V1 share the same total atom budget.
    """
    orbit = orbit_augment(gallery_x, n_orbit, max_eps, max_shift, gen)
    return Dictionary(_normalize_columns(orbit.T), labels.repeat(n_orbit), "v1")


def build_v2(
    gallery_x: torch.Tensor,
    labels: torch.Tensor,
    atoms_per_class: int,
    sparsity: int,
    n_iter: int,
    gen: torch.Generator,
) -> Dictionary:
    """V2: per-class complex K-SVD dictionaries, concatenated."""
    parts, atom_labels = [], []
    for c in labels.unique(sorted=True).tolist():
        Xc = gallery_x[labels == c].T
        D, _, _ = ksvd(Xc, atoms_per_class, sparsity, n_iter, gen)
        parts.append(D)
        atom_labels.extend([c] * atoms_per_class)
    atoms = _normalize_columns(torch.cat(parts, dim=1))
    return Dictionary(atoms, torch.tensor(atom_labels, dtype=torch.long, device=atoms.device), "v2")


def build_v3(
    gallery_x: torch.Tensor,
    labels: torch.Tensor,
    atoms_per_class: int,
    sparsity: int,
    alpha_lc: float,
    n_iter: int,
    gen: torch.Generator,
) -> Dictionary:
    """V3: one shared LC-KSVD1 dictionary with label-consistent atoms."""
    D, atom_labels = lc_ksvd(gallery_x.T, labels, atoms_per_class, sparsity, alpha_lc, n_iter, gen)
    return Dictionary(D, atom_labels, "v3")


def get_or_build_dictionary(
    variant: str,
    cache_dir: Path,
    hdf5_path: Path,
    manifest_dir: Path,
    atoms_per_class: int,
    snr_min: int,
    seed: int,
    device: torch.device,
    n_orbit: int = 4,
    max_eps: float = 0.5,
    max_shift: int = 32,
    ksvd_sparsity: int = 8,
    ksvd_iters: int = 30,
    lc_alpha: float = 1.0,
    train_per_class: int | None = None,
) -> Dictionary:
    """Build (or load from cache) one dictionary variant; K-SVD variants resume for free.

    Cache key: dicts/dict_{variant}_k{K}_snr{snr_min}_s{seed}.pt with
    K = atoms_per_class * n_classes.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown dictionary variant: {variant!r}")
    K = atoms_per_class * len(MOD_CLASSES)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"dict_{variant}_k{K}_snr{snr_min}_s{seed}.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location=device, weights_only=True)
        return Dictionary(payload["atoms"], payload["atom_labels"], payload["variant"])

    gen = torch.Generator(device=device).manual_seed(seed)
    if variant == "v0":
        x, labels = build_gallery(hdf5_path, manifest_dir, atoms_per_class, snr_min, seed, device)
        d = build_v0(x, labels)
    elif variant == "v1":
        base = max(1, atoms_per_class // n_orbit)
        x, labels = build_gallery(hdf5_path, manifest_dir, base, snr_min, seed, device)
        d = build_v1(x, labels, n_orbit, max_eps, max_shift, gen)
    else:
        per_class = train_per_class if train_per_class is not None else 4 * atoms_per_class
        x, labels = build_gallery(hdf5_path, manifest_dir, per_class, snr_min, seed, device)
        if variant == "v2":
            d = build_v2(x, labels, atoms_per_class, ksvd_sparsity, ksvd_iters, gen)
        else:
            d = build_v3(x, labels, atoms_per_class, ksvd_sparsity, lc_alpha, ksvd_iters, gen)

    tmp = cache_path.with_suffix(".pt.tmp")
    torch.save({"atoms": d.atoms, "atom_labels": d.atom_labels, "variant": d.variant}, tmp)
    tmp.rename(cache_path)
    return d
