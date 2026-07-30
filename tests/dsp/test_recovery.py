import torch

from dsp.frames import gabor_frame, synthesis, analysis
from dsp.operators import random_convolution, measure, backproject
from dsp.recovery import (
    _oamp_update,
    apply_A,
    apply_AH,
    bg_mmse,
    bg_mmse_divergence,
    bg_mmse_step,
    denoise_step,
    complex_soft_threshold,
    lasso_fista,
    oamp,
    oamp_bg_mmse,
    reconstruct,
    soft_threshold_divergence,
)

N = 96
GAMMA = 2
KAPPA = 1.6


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


EPS_BG = 0.05


def _bg(device, shape, eps=EPS_BG, seed=3):
    g = _gen(device, seed)
    mask = (torch.rand(shape, generator=g, device=device) < eps).to(torch.complex64)
    z = torch.randn(shape, dtype=torch.complex64, device=device, generator=g)
    return mask * z * (1.0 / eps) ** 0.5


class TestBernoulliGaussianMmse:
    def test_keeps_every_atom_and_stays_monotone(self, device):
        # the property soft-threshold lacks: no interval collapses to zero, so the map is invertible
        mag = torch.linspace(1e-4, 6.0, 512, device=device)
        out = bg_mmse(mag.to(torch.complex64), torch.tensor([0.4], device=device), EPS_BG).abs()
        assert out.min().item() > 0.0
        assert (out.diff() > 0).all()

    def test_preserves_phase(self, device):
        u = torch.exp(1j * torch.tensor([0.3, 1.1, -2.0], device=device)) * 2.5
        out = bg_mmse(u, torch.tensor([0.5], device=device), EPS_BG)
        assert torch.allclose(out.angle(), u.angle(), atol=1e-5)

    def test_divergence_matches_finite_differences(self, device):
        # pins the closed-form Onsager term: d eta / d r = (d/dx - i d/dy) / 2 at each coordinate
        r = torch.randn(1, 64, dtype=torch.complex64, device=device, generator=_gen(device, 5))
        tau = torch.tensor([[0.6]], device=device)
        h = 1e-3
        base = bg_mmse(r, tau, EPS_BG)
        dx = (bg_mmse(r + h, tau, EPS_BG) - base) / h
        dy = (bg_mmse(r + 1j * h, tau, EPS_BG) - base) / h
        expect = (0.5 * (dx - 1j * dy)).real.mean(dim=-1, keepdim=True)
        got = bg_mmse_divergence(r, tau, EPS_BG)
        assert (got - expect).abs().max().item() < 5e-3

    def test_beats_soft_threshold_under_its_own_prior(self, device):
        # null: the matched denoiser is no better than a tuned threshold on Bernoulli-Gaussian draws
        alpha = _bg(device, (1, 40000))
        tau = 0.5
        z = torch.randn(alpha.shape, dtype=torch.complex64, device=device, generator=_gen(device, 6))
        r = alpha + tau * z
        t = torch.full((1, 1), tau, device=device)
        mmse = (bg_mmse(r, t, EPS_BG) - alpha).abs().pow(2).mean().item()
        best = min((complex_soft_threshold(r, k * tau) - alpha).abs().pow(2).mean().item()
                   for k in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0))
        assert mmse < best

    def test_recovers_planted_sparse(self, device):
        frame = gabor_frame(N, N, N // 2, device=device)
        op = random_convolution(N, round(0.9 * N), _gen(device, 1), device=device)
        x = synthesis(_planted(device, frame.n_atoms), frame)
        eps = 6.0 / frame.n_atoms
        got = reconstruct(oamp_bg_mmse(measure(x, op), op, frame, eps), frame)
        assert _snr_db(x, got) > 20.0

    def test_divergence_free_step_matches_the_helpers(self, device):
        r = torch.randn(8, 128, dtype=torch.complex64, device=device, generator=_gen(device, 13))
        tau = torch.full((8, 1), 0.7, device=device)
        div = bg_mmse_divergence(r, tau, EPS_BG).clamp(0.0, 0.95)
        expect = (bg_mmse(r, tau, EPS_BG) - div * r) / (1.0 - div)
        got = bg_mmse_step(r, tau, EPS_BG)
        assert ((got - expect).abs().max() / expect.abs().mean()).item() < 1e-5

    def test_cpu_cuda_match(self, device):
        if device.type != "cuda":
            return
        frame_c = gabor_frame(N, N, N // 2, device="cpu")
        op_c = random_convolution(N, round(0.7 * N), _gen("cpu", 1), device="cpu")
        y_c = measure(synthesis(_planted(torch.device("cpu"), frame_c.n_atoms), frame_c), op_c)
        eps = 6.0 / frame_c.n_atoms
        out_c = oamp_bg_mmse(y_c, op_c, frame_c, eps)
        out_g = oamp_bg_mmse(y_c.cuda(), op_c.to("cuda"), frame_c.to("cuda"), eps)
        assert torch.allclose(out_c, out_g.cpu(), atol=1e-3)


class TestRecovery:
    def test_both_recover_planted_sparse(self, device):
        frame = gabor_frame(N, N, N // 2, device=device)
        op = random_convolution(N, round(0.8 * N), _gen(device, 1), device=device)
        alpha = _planted(device, frame.n_atoms)
        x = synthesis(alpha, frame)
        y = measure(x, op)
        assert _snr_db(x, reconstruct(oamp(y, op, frame, kappa=KAPPA), frame)) > 40.0
        assert _snr_db(x, reconstruct(lasso_fista(y, op, frame, lam=0.02, iters=400), frame)) > 15.0

    def test_oamp_reaches_high_snr_when_exactly_sparse(self, device):
        # separates a signal-side compressibility floor from a solver that cannot converge
        frame = gabor_frame(N, N, N // 2, device=device)
        op = random_convolution(N, round(0.9 * N), _gen(device, 1), device=device)
        x = synthesis(_planted(device, frame.n_atoms), frame)
        assert _snr_db(x, reconstruct(oamp(measure(x, op), op, frame, kappa=KAPPA), frame)) > 40.0

    def test_output_stays_bounded_on_near_tone(self, device):
        # null: an unnormalized linear stage compounds on frames whose atoms never fall below
        # threshold, so a tone in a redundant frame diverges instead of merely failing
        for window, hop in ((N, N // 2), (32, 8)):
            frame = gabor_frame(N, window, hop, device=device)
            k = torch.arange(N, device=device, dtype=torch.float32)
            x = torch.stack([torch.exp(2j * torch.pi * (5 + i) * k / N) for i in range(4)])
            x = x / x.abs().pow(2).mean(-1, keepdim=True).sqrt()
            for rho in (0.3, 0.7):
                op = random_convolution(N, round(rho * N), _gen(device, 1), device=device)
                xhat = reconstruct(oamp(measure(x, op), op, frame, kappa=KAPPA), frame)
                assert xhat.norm().item() < 2.0 * x.norm().item()
                assert _snr_db(x, xhat) > 0.0

    def test_divergence_clamp_does_not_bind(self, device):
        frame = gabor_frame(N, N, N // 2, device=device)
        op = random_convolution(N, round(0.9 * N), _gen(device, 1), device=device)
        y = measure(synthesis(_planted(device, frame.n_atoms), frame), op)
        s = oamp(y, op, frame, kappa=KAPPA)
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
        xo = reconstruct(oamp(y, op, frame, kappa=KAPPA), frame)
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


class TestFusedUpdate:
    def test_matches_the_exported_helpers(self, device):
        # pins the algebra so the fused loop and the published primitives cannot drift
        g = _gen(device, 11)
        r = torch.randn(8, 128, dtype=torch.complex64, device=device, generator=g)
        th = torch.full((8, 1), 0.2, device=device)
        div = soft_threshold_divergence(r, th).clamp(0.0, 0.95)
        expect = (complex_soft_threshold(r, th) - div * r) / (1.0 - div)
        got = denoise_step(r, th)
        assert ((got - expect).abs().max() / expect.abs().mean()).item() < 1e-5

    def test_compiled_matches_eager(self, device):
        g = _gen(device, 12)
        s = torch.randn(8, 128, dtype=torch.complex64, device=device, generator=g)
        innov = torch.randn(8, 128, dtype=torch.complex64, device=device, generator=g)
        scaled = innov / 3.0
        tau = scaled.abs().pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        expect = denoise_step(s + scaled, 1.1 * tau)
        got = _oamp_update(s, innov, 1.1, 3.0)
        assert ((got - expect).abs().max() / expect.abs().mean()).item() < 1e-4

    def test_tf32_does_not_move_recovery(self, device):
        if device.type != "cuda":
            return
        frame = gabor_frame(N, N, N // 2, device=device)
        op = random_convolution(N, round(0.8 * N), _gen(device, 1), device=device)
        y = measure(synthesis(_planted(device, frame.n_atoms), frame), op)
        x = synthesis(_planted(device, frame.n_atoms), frame)
        prev = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            exact = _snr_db(x, reconstruct(oamp(y, op, frame, kappa=KAPPA), frame))
            torch.backends.cuda.matmul.allow_tf32 = True
            fast = _snr_db(x, reconstruct(oamp(y, op, frame, kappa=KAPPA), frame))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        assert abs(exact - fast) / abs(exact) < 1e-3


class TestDeviceParity:
    def test_oamp_cpu_cuda_match(self, device):
        if device.type != "cuda":
            return
        frame_c = gabor_frame(N, N, N // 2, device="cpu")
        op_c = random_convolution(N, round(0.7 * N), _gen("cpu", 1), device="cpu")
        alpha = _planted(torch.device("cpu"), frame_c.n_atoms)
        y_c = measure(synthesis(alpha, frame_c), op_c)
        out_c = oamp(y_c, op_c, frame_c, kappa=KAPPA)
        out_g = oamp(y_c.cuda(), op_c.to("cuda"), frame_c.to("cuda"), kappa=KAPPA)
        assert torch.allclose(out_c, out_g.cpu(), atol=1e-3)
