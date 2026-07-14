import torch

# Batched complex-valued sparse solvers. Signals are rows of Y [B, m]; the
# dictionary/sensing product A is [m, K]. Codes alpha are rows [B, K]. All
# math is C-linear: the l1 prox acts on complex magnitudes (joint I/Q), never
# on real and imaginary parts independently.

EPS = 1e-12


def complex_soft_threshold(z: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
    """Magnitude soft-threshold (z/|z|) * relu(|z| - t); zero-safe at z = 0."""
    mag = z.abs()
    return z * ((mag - t).clamp_min(0.0) / mag.clamp_min(EPS))


def operator_norm_sq(A: torch.Tensor, n_iter: int = 50) -> float:
    """Estimate ||A||_2^2 (largest eigenvalue of A^H A) via power iteration."""
    gen = torch.Generator(device=A.device).manual_seed(0)
    v = torch.randn(A.shape[1], 2, generator=gen, device=A.device)
    v = (v[:, 0] + 1j * v[:, 1]).to(A.dtype)
    for _ in range(n_iter):
        v = A.conj().T @ (A @ v)
        v = v / v.norm().clamp_min(EPS)
    return (A @ v).norm().pow(2).item()


def _as_col(lam: torch.Tensor | float, batch: int, device: torch.device) -> torch.Tensor:
    """Broadcast a scalar or per-frame lambda to shape [B, 1]."""
    if not torch.is_tensor(lam):
        lam = torch.tensor(float(lam), device=device)
    return lam.reshape(-1, 1).expand(batch, 1) if lam.numel() == 1 else lam.reshape(batch, 1)


def fista_batch(
    A: torch.Tensor,
    Y: torch.Tensor,
    lam: torch.Tensor | float,
    n_iter: int = 200,
    L: float | None = None,
    gram: torch.Tensor | None = None,
    AhY: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve min_a 0.5||y - A a||^2 + lam ||a||_1 per row of Y with complex FISTA.

    lam may be a scalar or per-frame [B]. gram (A^H A) and AhY (rows A^H y_b)
    can be precomputed once per run spec. Returns (alpha [B, K], res_norms [B]).
    """
    B = Y.shape[0]
    if gram is None:
        gram = A.conj().T @ A
    if AhY is None:
        AhY = Y @ A.conj()
    if L is None:
        L = operator_norm_sq(A)
    step   = 1.0 / L
    thresh = _as_col(lam, B, Y.device) * step
    alpha  = torch.zeros_like(AhY)
    z      = alpha.clone()
    t      = 1.0
    for _ in range(n_iter):
        grad      = z @ gram.T - AhY
        alpha_new = complex_soft_threshold(z - step * grad, thresh)
        t_new     = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        z         = alpha_new + ((t - 1.0) / t_new) * (alpha_new - alpha)
        alpha, t  = alpha_new, t_new
    res_norms = (Y - alpha @ A.T).norm(dim=1)
    return alpha, res_norms


def bpdn_batch(
    A: torch.Tensor,
    Y: torch.Tensor,
    eps: torch.Tensor | float,
    n_iter: int = 200,
    n_bisect: int = 6,
    L: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve BPDN min ||a||_1 s.t. ||y - A a|| <= eps via per-frame bisection on lambda.

    Each bisection step is one batched fista_batch call. Returns (alpha, res_norms)
    at the largest feasible lambda found per frame (lower bound if never feasible).
    """
    B    = Y.shape[0]
    gram = A.conj().T @ A
    AhY  = Y @ A.conj()
    if L is None:
        L = operator_norm_sq(A)
    eps_col  = _as_col(eps, B, Y.device).squeeze(1)
    lam_hi   = AhY.abs().amax(dim=1).clamp_min(EPS)
    lam_lo   = lam_hi * 1e-3
    lam_best = lam_lo.clone()
    for _ in range(n_bisect):
        lam_mid = (lam_lo * lam_hi).sqrt()
        _, res  = fista_batch(A, Y, lam_mid, n_iter, L=L, gram=gram, AhY=AhY)
        feasible = res <= eps_col
        lam_best = torch.where(feasible, torch.maximum(lam_best, lam_mid), lam_best)
        lam_lo   = torch.where(feasible, lam_mid, lam_lo)
        lam_hi   = torch.where(feasible, lam_hi, lam_mid)
    return fista_batch(A, Y, lam_best, n_iter, L=L, gram=gram, AhY=AhY)


def debias_batch(
    A: torch.Tensor, Y: torch.Tensor, alpha: torch.Tensor, k_max: int
) -> torch.Tensor:
    """Least-squares refit on the top-k support of alpha (standard lasso debiasing).

    Removes the soft-threshold shrinkage bias from a FISTA/BPDN solution before
    the code is used for reconstruction-quality metrics.
    """
    B, K = alpha.shape
    k    = min(k_max, K, A.shape[0])
    idx  = alpha.abs().topk(k, dim=1).indices
    A_sel = A.T[idx].transpose(1, 2)                     # [B, m, k]
    coef  = torch.linalg.lstsq(A_sel, Y.unsqueeze(-1)).solution
    out = torch.zeros_like(alpha)
    out.scatter_(1, idx, coef.squeeze(-1))
    return out


def omp_batch(A: torch.Tensor, Y: torch.Tensor, sparsity: int) -> torch.Tensor:
    """Complex orthogonal matching pursuit; returns alpha [B, K] with <= sparsity nonzeros per row.

    Training-time solver (the sparse-coding step inside K-SVD); not used in the
    per-frame classification path.
    """
    B, m = Y.shape
    K    = A.shape[1]
    sparsity = min(sparsity, K, m)
    support  = torch.zeros(B, sparsity, dtype=torch.long, device=Y.device)
    selected = torch.zeros(B, K, dtype=torch.bool, device=Y.device)
    residual = Y.clone()
    coef     = None
    for s in range(sparsity):
        corr = (residual @ A.conj()).abs()
        corr[selected] = -1.0
        idx = corr.argmax(dim=1)
        support[:, s] = idx
        selected[torch.arange(B, device=Y.device), idx] = True
        A_sel = A.T[support[:, : s + 1]].transpose(1, 2)          # [B, m, s+1]
        coef  = torch.linalg.lstsq(A_sel, Y.unsqueeze(-1)).solution
        residual = Y - (A_sel @ coef).squeeze(-1)
    alpha = torch.zeros(B, K, dtype=A.dtype, device=Y.device)
    alpha.scatter_(1, support, coef.squeeze(-1))
    return alpha
