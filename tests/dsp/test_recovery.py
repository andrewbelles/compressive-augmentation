import torch

from dsp.frames import gabor_frame, synthesis, analysis
from dsp.operators import random_convolution, measure, backproject
from dsp.recovery import (
    apply_A,
    apply_AH,
    complex_soft_threshold,
    lasso_fista,
    oamp,
    reconstruct,
    soft_threshold_divergence,
)

N = 96
GAMMA = 2


def _gen(device, seed=0):
    return torch.Generator(device=device).manual_seed(seed)


def _snr_db(x, xhat):
    num = x.abs().pow(2).sum()
    den = (x - xhat).abs().pow(2).sum().clamp_min(1e-12)
    return (10 * torch.log10(num / den)).item()


def _planted(device, d, k=6, b=16, seed=7):
    g = _gen(device, seed)
    alpha = torch.zeros(b, d, dtype=torch.complex64, device=device)
    for i in range(b):
        idx = torch.randperm(d, generator=g, device=device)[:k]
        alpha[i, idx] = torch.randn(k, dtype=torch.complex64, device=device, generator=g)
    return alpha


class TestSoftThreshold:
    def test_kills_small_and_shrinks_large(self, device):
        u = torch.tensor([0.5, 2.0], dtype=torch.complex64, device=device)
        out = complex_soft_threshold(u, 1.0)
        assert out[0].abs().item() == 0.0
        assert abs(out[1].abs().item() - 1.0) < 1e-5

    def test_preserves_phase(self, device):
        u = torch.exp(1j * torch.tensor([0.3, 1.1], device=device)) * 3.0
        out = complex_soft_threshold(u, 1.0)
        assert torch.allclose(out.angle(), u.angle(), atol=1e-5)


class TestRecovery:
    def test_both_recover_planted_sparse(self, device):
        frame = gabor_frame(N, N, N // 2, device=device)
        op = random_convolution(N, round(0.8 * N), _gen(device, 1), device=device)
        alpha = _planted(device, frame.n_atoms)
        x = synthesis(alpha, frame)
        y = measure(x, op)
        assert _snr_db(x, reconstruct(oamp(y, op, frame), frame)) > 40.0
        assert _snr_db(x, reconstruct(lasso_fista(y, op, frame, lam=0.02, iters=400), frame)) > 15.0

    def test_oamp_reaches_high_snr_when_exactly_sparse(self, device):
        # separates a signal-side compressibility floor from a solver that cannot converge
        frame = gabor_frame(N, N, N // 2, device=device)
        op = random_convolution(N, round(0.9 * N), _gen(device, 1), device=device)
        x = synthesis(_planted(device, frame.n_atoms), frame)
        assert _snr_db(x, reconstruct(oamp(measure(x, op), op, frame), frame)) > 40.0

    def test_divergence_clamp_does_not_bind(self, device):
        frame = gabor_frame(N, N, N // 2, device=device)
        op = random_convolution(N, round(0.9 * N), _gen(device, 1), device=device)
        y = measure(synthesis(_planted(device, frame.n_atoms), frame), op)
        s = oamp(y, op, frame)
        innov = apply_AH(y - apply_A(s, op, frame), op, frame)
        tau = innov.abs().pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        div = soft_threshold_divergence(s + innov, 1.6 * tau)
        assert div.max().item() < 0.95

    def test_incompressible_recovers_worse(self, device):
        # null: white-noise frames are not sparse in the Gabor frame
        frame = gabor_frame(N, N, N // 2, device=device)
        op = random_convolution(N, round(0.5 * N), _gen(device, 1), device=device)
        alpha = _planted(device, frame.n_atoms)
        x_struct = synthesis(alpha, frame)
        x_white = torch.randn(16, N, dtype=torch.complex64, device=device, generator=_gen(device, 4))
        s = _snr_db(x_struct, reconstruct(lasso_fista(measure(x_struct, op), op, frame, 0.02, 400), frame))
        w = _snr_db(x_white, reconstruct(lasso_fista(measure(x_white, op), op, frame, 0.02, 400), frame))
        assert s - w > 8.0

    def test_solvers_agree_at_high_ratio(self, device):
        frame = gabor_frame(N, N, N // 2, device=device)
        op = random_convolution(N, round(0.9 * N), _gen(device, 1), device=device)
        alpha = _planted(device, frame.n_atoms)
        y = measure(synthesis(alpha, frame), op)
        xo = reconstruct(oamp(y, op, frame), frame)
        xf = reconstruct(lasso_fista(y, op, frame, 0.01, 500), frame)
        corr = (xo.flatten().conj() @ xf.flatten()).abs() / (xo.norm() * xf.norm())
        assert corr.item() > 0.9

    def test_backprojection_residual(self, device):
        frame = gabor_frame(N, N, N // 2, device=device)
        op = random_convolution(N, round(0.6 * N), _gen(device, 1), device=device)
        x = torch.randn(8, N, dtype=torch.complex64, device=device, generator=_gen(device, 2))
        # e = x - P x lies in the null space: measuring it gives zero
        e = x - backproject(measure(x, op), op)
        assert measure(e, op).abs().max().item() < 1e-3


class TestDeviceParity:
    def test_oamp_cpu_cuda_match(self, device):
        if device.type != "cuda":
            return
        frame_c = gabor_frame(N, N, N // 2, device="cpu")
        op_c = random_convolution(N, round(0.7 * N), _gen("cpu", 1), device="cpu")
        alpha = _planted(torch.device("cpu"), frame_c.n_atoms)
        y_c = measure(synthesis(alpha, frame_c), op_c)
        out_c = oamp(y_c, op_c, frame_c)
        out_g = oamp(y_c.cuda(), op_c.to("cuda"), frame_c.to("cuda"))
        assert torch.allclose(out_c, out_g.cpu(), atol=1e-3)
