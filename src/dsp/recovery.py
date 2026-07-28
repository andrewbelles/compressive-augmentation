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


def denoise_step(r: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Divergence-free soft-threshold update, sharing one magnitude pass over r."""
    mag = r.abs()
    active = (mag > theta).to(r.real.dtype)
    div = (active * (1.0 - theta / (2.0 * mag.clamp_min(EPS)))).mean(dim=-1, keepdim=True)
    div = div.clamp(0.0, 0.95)
    shrink = (mag - theta).clamp_min(0.0) / mag.clamp_min(EPS)
    return r * ((shrink - div) / (1.0 - div))


def _oamp_update_real(sr: torch.Tensor, ir: torch.Tensor, kappa: float, gain: float) -> torch.Tensor:
    """Whole update over (..., 2) real views so inductor can fuse it; it rejects complex ops."""
    ire, iim = ir[..., 0] / gain, ir[..., 1] / gain
    re, im = sr[..., 0] + ire, sr[..., 1] + iim
    tau = (ire * ire + iim * iim).mean(dim=-1, keepdim=True).clamp_min(EPS).sqrt()
    theta = kappa * tau
    mag = torch.sqrt(re * re + im * im)
    active = (mag > theta).to(sr.dtype)
    div = (active * (1.0 - theta / (2.0 * mag.clamp_min(EPS)))).mean(dim=-1, keepdim=True)
    div = div.clamp(0.0, 0.95)
    shrink = (mag - theta).clamp_min(0.0) / mag.clamp_min(EPS)
    scale = (shrink - div) / (1.0 - div)
    return torch.stack([re * scale, im * scale], dim=-1)


_compiled_update = torch.compile(_oamp_update_real, dynamic=True)
_compile_ok = True


def _oamp_update(s: torch.Tensor, innov: torch.Tensor, kappa: float, gain: float) -> torch.Tensor:
    """One OAMP step, compiled when available and eager complex otherwise."""
    global _compile_ok
    if _compile_ok:
        try:
            out = _compiled_update(torch.view_as_real(s).contiguous(),
                                   torch.view_as_real(innov).contiguous(), kappa, gain)
            return torch.view_as_complex(out.contiguous())
        except Exception:
            _compile_ok = False
    scaled = innov / gain
    tau = scaled.abs().pow(2).mean(dim=-1, keepdim=True).clamp_min(EPS).sqrt()
    return denoise_step(s + scaled, kappa * tau)


def oamp(
    y: torch.Tensor,
    op: ConvOperator,
    frame: Frame,
    kappa: float,
    iters: int = 120,
    damping: float = 1.0,
) -> torch.Tensor:
    """Complex OAMP recovery: divergence-free soft-threshold with matched linear stage."""
    gain = frame.gamma * op.n / op.m  # nonzero eigenvalue of A*A, so W = A*/gain makes WA a projection
    s = torch.zeros(*y.shape[:-1], frame.n_atoms, dtype=frame.d.dtype, device=y.device)
    for _ in range(iters):
        innov = apply_AH(y - apply_A(s, op, frame), op, frame)
        nxt = _oamp_update(s, innov, kappa, gain)
        s = nxt if damping == 1.0 else damping * nxt + (1.0 - damping) * s
    return s


def reconstruct(alpha: torch.Tensor, frame: Frame) -> torch.Tensor:
    """Map recovered coefficients back to a signal view, x_tilde = D alpha."""
    return synthesis(alpha, frame)
