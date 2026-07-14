import math

import torch

from csmath.solvers import omp_batch

# Complex K-SVD / LC-KSVD. Signals are columns of X [n, P]; dictionaries are
# [n, K] with unit-norm atoms. Rank-1 atom updates use torch.linalg.svd on the
# complex residual directly (C-native -- no real/imag stacking, which would
# decouple I and Q).

EPS = 1e-12


def _normalize_columns(D: torch.Tensor) -> torch.Tensor:
    return D / D.norm(dim=0, keepdim=True).clamp_min(EPS)


def _random_atoms(n: int, k: int, gen: torch.Generator, device, dtype) -> torch.Tensor:
    re = torch.randn(n, k, generator=gen, device=device)
    im = torch.randn(n, k, generator=gen, device=device)
    return _normalize_columns((re + 1j * im).to(dtype))


def _init_from_columns(X: torch.Tensor, n_atoms: int, gen: torch.Generator) -> torch.Tensor:
    """Initialize atoms from random distinct signal columns, gaussian-padded if short."""
    n, P = X.shape
    take = min(n_atoms, P)
    cols = torch.randperm(P, generator=gen, device=X.device)[:take]
    D = X[:, cols]
    if take < n_atoms:
        D = torch.cat([D, _random_atoms(n, n_atoms - take, gen, X.device, X.dtype)], dim=1)
    return _normalize_columns(D.clone())


def _ksvd_iterations(
    X: torch.Tensor,
    D: torch.Tensor,
    sparsity: int,
    n_iter: int,
    replace_pools: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Run approximate K-SVD sweeps (OMP coding + sequential rank-1 atom updates).

    replace_pools optionally restricts, per atom, which signal columns a dead
    atom may be re-seeded from (used by LC-KSVD to keep atoms class-pure).
    """
    x_norm = X.norm().clamp_min(EPS)
    codes  = torch.zeros(D.shape[1], X.shape[1], dtype=X.dtype, device=X.device)
    err_history: list[float] = []
    for _ in range(n_iter):
        codes = omp_batch(D, X.T, sparsity).T
        for k in range(D.shape[1]):
            omega = codes[k].abs().nonzero(as_tuple=True)[0]
            if omega.numel() == 0:
                col_err = (X - D @ codes).norm(dim=0)
                pool = replace_pools[k] if replace_pools is not None else None
                worst = pool[col_err[pool].argmax()] if pool is not None else col_err.argmax()
                D[:, k] = X[:, worst] / X[:, worst].norm().clamp_min(EPS)
                continue
            E = X[:, omega] - D @ codes[:, omega] + torch.outer(D[:, k], codes[k, omega])
            U, S, Vh = torch.linalg.svd(E, full_matrices=False)
            D[:, k] = U[:, 0]
            codes[k, omega] = S[0].to(X.dtype) * Vh[0]
        err_history.append(((X - D @ codes).norm() / x_norm).item())
    return D, codes, err_history


def ksvd(
    X: torch.Tensor,
    n_atoms: int,
    sparsity: int,
    n_iter: int,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Learn a complex dictionary D [n, n_atoms] from signal columns X [n, P].

    Returns (D, codes [n_atoms, P], per-iteration relative Frobenius error).
    """
    D = _init_from_columns(X, n_atoms, gen)
    return _ksvd_iterations(X, D, sparsity, n_iter)


def lc_ksvd(
    X: torch.Tensor,
    labels: torch.Tensor,
    atoms_per_class: int,
    sparsity: int,
    alpha_lc: float,
    n_iter: int,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LC-KSVD1: K-SVD on [X; sqrt(alpha) Q] with Q the 0/1 discriminative-code target.

    Q anchors each atom to one class, so the returned atom_labels stay valid and
    the decision rule can remain residual-based (no LC-KSVD2 classifier W).
    Returns (D [n, atoms_per_class * n_classes] unit-norm, atom_labels).
    """
    n, P = X.shape
    classes = labels.unique(sorted=True)
    n_classes = classes.numel()
    K = atoms_per_class * n_classes
    atom_labels = classes.repeat_interleave(atoms_per_class)

    Q = (atom_labels.unsqueeze(1) == labels.unsqueeze(0)).to(X.dtype)     # [K, P]
    X_aug = torch.cat([X, math.sqrt(alpha_lc) * Q], dim=0)

    parts, pools = [], []
    for c in classes:
        cols = (labels == c).nonzero(as_tuple=True)[0]
        parts.append(_init_from_columns(X_aug[:, cols], atoms_per_class, gen))
        pools.extend([cols] * atoms_per_class)
    D_aug = torch.cat(parts, dim=1)

    D_aug, _, _ = _ksvd_iterations(X_aug, D_aug, sparsity, n_iter, replace_pools=pools)
    return _normalize_columns(D_aug[:n]), atom_labels
