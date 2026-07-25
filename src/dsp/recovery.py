import torch

from .operators import ConvOperator, measure, adjoint
from .frames import Frame, synthesis, analysis

# complex sparse recovery in a Gabor frame: OAMP (primary) and FISTA (cross-check)

EPS = 1e-12


def complex_soft_threshold(u: torch.Tensor, theta) -> torch.Tensor:
    """Complex soft-threshold eta(u) = (|u| - theta)_+ u/|u|."""
    mag = u.abs()
    shrunk = (mag - theta).clamp_min(0.0)
    return torch.where(mag > EPS, u * (shrunk / mag.clamp_min(EPS)), torch.zeros_like(u))


def apply_A(alpha: torch.Tensor, op: ConvOperator, frame: Frame) -> torch.Tensor:
    """A alpha = Phi D alpha."""
    return measure(synthesis(alpha, frame), op)


def apply_AH(y: torch.Tensor, op: ConvOperator, frame: Frame) -> torch.Tensor:
    """A* y = D* Phi* y."""
    return analysis(adjoint(y, op), frame)


def lasso_fista(
    y: torch.Tensor,
    op: ConvOperator,
    frame: Frame,
    lam: float,
    iters: int = 200,
) -> torch.Tensor:
    """Solve min 0.5||y - A alpha||^2 + lam ||alpha||_1 by batched FISTA."""
    step = op.m / (frame.gamma * op.n)  # 1/L, L = gamma N/m
    alpha = torch.zeros(*y.shape[:-1], frame.n_atoms, dtype=frame.d.dtype, device=y.device)
    z = alpha.clone()
    t = 1.0
    for _ in range(iters):
        grad = apply_AH(apply_A(z, op, frame) - y, op, frame)
        alpha_next = complex_soft_threshold(z - step * grad, lam * step)
        t_next = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        z = alpha_next + ((t - 1.0) / t_next) * (alpha_next - alpha)
        alpha, t = alpha_next, t_next
    return alpha


def sparse_code(
    x: torch.Tensor,
    frame: Frame,
    lam: float = 0.05,
    max_iters: int = 400,
    tol: float = 1e-5,
) -> tuple[torch.Tensor, int]:
    """Sparse-code x in the frame by FISTA, returning coefficients and iterations used."""
    step = 1.0 / frame.gamma  # 1/L, L = ||D||^2 = gamma
    alpha = torch.zeros(*x.shape[:-1], frame.n_atoms, dtype=frame.d.dtype, device=x.device)
    z, t, used = alpha.clone(), 1.0, max_iters
    for i in range(max_iters):
        grad = analysis(synthesis(z, frame) - x, frame)
        nxt = complex_soft_threshold(z - step * grad, lam * step)
        t2 = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        z = nxt + ((t - 1.0) / t2) * (nxt - alpha)
        shift = (nxt - alpha).norm() / nxt.norm().clamp_min(EPS)
        alpha, t = nxt, t2
        if shift.item() < tol:
            used = i + 1
            break
    return alpha, used


def soft_threshold_divergence(r: torch.Tensor, theta) -> torch.Tensor:
    """Mean Wirtinger divergence of the complex soft-threshold, for the Onsager term."""
    mag = r.abs()
    active = (mag > theta).to(r.real.dtype)
    return (active * (1.0 - theta / (2.0 * mag.clamp_min(EPS)))).mean(dim=-1, keepdim=True)


def oamp(
    y: torch.Tensor,
    op: ConvOperator,
    frame: Frame,
    iters: int = 40,
    kappa: float = 1.6,
    damping: float = 0.6,
) -> torch.Tensor:
    """Complex OAMP recovery: divergence-free soft-threshold with matched linear stage."""
    s = torch.zeros(*y.shape[:-1], frame.n_atoms, dtype=frame.d.dtype, device=y.device)
    for _ in range(iters):
        resid = y - apply_A(s, op, frame)
        innov = apply_AH(resid, op, frame)  # W = A* is de-correlated for tight D
        r = s + innov
        tau = innov.abs().pow(2).mean(dim=-1, keepdim=True).clamp_min(EPS).sqrt()
        theta = kappa * tau
        eta = complex_soft_threshold(r, theta)
        div = soft_threshold_divergence(r, theta).clamp(0.0, 0.95)
        s = damping * ((eta - div * r) / (1.0 - div)) + (1.0 - damping) * s
    return s


def reconstruct(alpha: torch.Tensor, frame: Frame) -> torch.Tensor:
    """Map recovered coefficients back to a signal view, x_tilde = D alpha."""
    return synthesis(alpha, frame)
