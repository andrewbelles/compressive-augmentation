from dataclasses import dataclass

import torch

from csmath.solvers import bpdn_batch, fista_batch, operator_norm_sq

EPS = 1e-12


@dataclass
class SRCResult:
    """Per-frame SRC outputs: predicted class, residual margin, solver residual."""
    pred:     torch.Tensor   # [B] long
    margin:   torch.Tensor   # [B] float32, runner-up minus best class residual
    res_norm: torch.Tensor   # [B] float32, final solver residual ||y - A_full alpha||


def class_residuals(
    A: torch.Tensor,
    atom_labels: torch.Tensor,
    Y: torch.Tensor,
    alpha: torch.Tensor,
    n_classes: int,
) -> torch.Tensor:
    """Return [B, C] residuals r_c = ||y - A delta_c(alpha)|| for each class c."""
    res = torch.empty(Y.shape[0], n_classes, device=Y.device)
    for c in range(n_classes):
        cols = atom_labels == c
        recon = alpha[:, cols] @ A[:, cols].T
        res[:, c] = (Y - recon).norm(dim=1)
    return res


def src_classify(
    A: torch.Tensor,
    atom_labels: torch.Tensor,
    Y: torch.Tensor,
    solver: str = "fista",
    lam_rel: float = 0.05,
    n_iter: int = 200,
    eps: torch.Tensor | None = None,
    E: torch.Tensor | None = None,
    n_classes: int | None = None,
) -> SRCResult:
    """Wright SRC: one global sparse code over the multi-class dictionary A, then argmin
    of per-class residuals.

    solver is "fista" (lam = lam_rel * ||A^H y||_inf per frame) or "bpdn"
    (requires per-frame eps). An optional error dictionary E is appended for the
    solve; its contribution is subtracted from y before class residuals and its
    coefficients are excluded from them.
    """
    if n_classes is None:
        n_classes = int(atom_labels.max().item()) + 1
    K = A.shape[1]
    A_full = A if E is None else torch.cat([A, E], dim=1)

    L    = operator_norm_sq(A_full)
    gram = A_full.conj().T @ A_full
    AhY  = Y @ A_full.conj()
    if solver == "fista":
        lam = lam_rel * AhY.abs().amax(dim=1)
        alpha, res_norm = fista_batch(A_full, Y, lam, n_iter, L=L, gram=gram, AhY=AhY)
    elif solver == "bpdn":
        if eps is None:
            raise ValueError("bpdn solver requires per-frame eps")
        alpha, res_norm = bpdn_batch(A_full, Y, eps, n_iter, L=L)
    else:
        raise ValueError(f"unknown solver: {solver!r}")

    Y_eff = Y if E is None else Y - alpha[:, K:] @ E.T
    res   = class_residuals(A, atom_labels, Y_eff, alpha[:, :K], n_classes)

    best2  = res.topk(2, dim=1, largest=False)
    pred   = best2.indices[:, 0]
    margin = (best2.values[:, 1] - best2.values[:, 0]).float()
    return SRCResult(pred=pred, margin=margin, res_norm=res_norm.float())
