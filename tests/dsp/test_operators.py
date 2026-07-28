import math

import torch

from dsp.operators import (
    ConvOperator,
    SRHTOperator,
    _fwht,
    random_convolution,
    measure,
    measurement_noise,
    adjoint,
    backproject,
    srht,
    srht_measure,
    to_dense,
)
from dsp.frames import gabor_frame, dft_frame, synthesis

N = 64
M = 24


def _gen(device, seed=0):
    return torch.Generator(device=device).manual_seed(seed)


def _op(device, seed=0, n=N, m=M):
    return random_convolution(n, m, _gen(device, seed), device=device)


class TestRowOrthogonality:
    def test_phi_phi_star_is_scaled_identity(self, device):
        # Prop 2: Phi Phi* = (N/m) I_m exactly
        op = _op(device)
        phi = to_dense(op)
        gram = phi @ phi.conj().transpose(-1, -2)
        target = (N / M) * torch.eye(M, dtype=gram.dtype, device=device)
        assert torch.allclose(gram, target, atol=1e-4)

    def test_projection_spectrum_is_binary(self, device):
        # (m/N) Phi* Phi is an orthogonal projector: eigenvalues in {0, 1}
        op = _op(device)
        phi = to_dense(op)
        proj = (M / N) * phi.conj().transpose(-1, -2) @ phi
        eig = torch.linalg.eigvalsh(proj)
        ones = (eig > 0.5).sum().item()
        assert ones == M
        assert torch.allclose(eig.clamp(0, 1).round(), eig, atol=1e-4)


class TestAdjoint:
    def test_inner_product_identity(self, device):
        # <Phi x, y> = <x, Phi* y>
        op = _op(device)
        x = torch.randn(N, dtype=torch.complex64, device=device)
        y = torch.randn(M, dtype=torch.complex64, device=device)
        lhs = torch.vdot(measure(x, op), y)
        rhs = torch.vdot(x, adjoint(y, op))
        assert torch.allclose(lhs, rhs, atol=1e-4)

    def test_backprojection_is_projection(self, device):
        # backproject(measure(x)) = P x, and P is idempotent
        op = _op(device)
        x = torch.randn(4, N, dtype=torch.complex64, device=device)
        px = backproject(measure(x, op), op)
        ppx = backproject(measure(px, op), op)
        assert torch.allclose(px, ppx, atol=1e-4)


class TestPartialIsometry:
    def test_tight_frame_gives_equal_singular_values(self, device):
        # Prop 4: A = Phi D has m equal nonzero singular values for tight D
        op = _op(device)
        frame = gabor_frame(N, N, N // 2, device=device)
        a = measure(synthesis(torch.eye(frame.n_atoms, dtype=torch.complex64, device=device), frame), op)
        a = a.transpose(-1, -2)
        sv = torch.linalg.svdvals(a)
        nz = sv[sv > 1e-3]
        assert nz.numel() == M
        assert torch.allclose(nz, nz.mean().expand_as(nz), rtol=1e-3)

    def test_nontight_frame_spreads_spectrum(self, device):
        # Null: perturbing tightness spreads A's singular values
        op = _op(device)
        frame = gabor_frame(N, N, N // 2, device=device)
        d = frame.d.clone()
        d[:, ::7] *= 3.0
        a = (op_dense := to_dense(op)) @ d
        sv = torch.linalg.svdvals(a)
        nz = sv[sv > 1e-3]
        assert nz.std() / nz.mean() > 0.05


class TestEquivarianceIdentities:
    def test_conv_timing_identity_is_exact(self, device):
        # C circulant commutes with T_a, and Omega selected from a shifted signal is Omega - a
        op = _op(device)
        a = N // 4
        x = torch.randn(4, N, dtype=torch.complex64, device=device, generator=_gen(device, 3))
        shifted_op = ConvOperator(N, M, (op.omega - a) % N, op.theta)
        lhs = measure(torch.roll(x, a, dims=-1), op)
        assert (lhs - measure(x, shifted_op)).abs().max().item() < 1e-5

    def test_conv_ongrid_cfo_identity_is_exact(self, device):
        # Phi M_b = U Phi' with U unimodular and Phi' carrying cyclically shifted phases
        op = _op(device)
        b = 3
        k = torch.arange(N, device=device)
        x = torch.randn(4, N, dtype=torch.complex64, device=device, generator=_gen(device, 3))
        mod = x * torch.exp(2j * math.pi * b * k / N)
        cfo_op = ConvOperator(N, M, op.omega, torch.roll(op.theta, -b))
        err = (measure(mod, op).abs() - measure(x, cfo_op).abs()).abs().max().item()
        assert err < 1e-5

    def test_srht_timing_identity_fails(self, device):
        # the null: random signs do not absorb cyclic shifts, so SRHT is not G-equivariant
        op = srht(N, M, _gen(device, 5), device=device)
        a = N // 4
        x = torch.randn(4, N, dtype=torch.complex64, device=device, generator=_gen(device, 3))
        shifted_op = SRHTOperator(N, M, (op.omega - a) % N, op.signs)
        lhs = srht_measure(torch.roll(x, a, dims=-1), op)
        rel = (lhs - srht_measure(x, shifted_op)).abs().max() / lhs.abs().mean()
        assert rel.item() > 0.5


class TestSRHT:
    def test_is_scaled_partial_isometry(self, device):
        op = srht(N, M, _gen(device, 5), device=device)
        eye = torch.eye(N, dtype=torch.complex64, device=device)
        phi = srht_measure(eye, op).transpose(-1, -2)
        gram = phi @ phi.conj().transpose(-1, -2)
        target = (N / M) * torch.eye(M, dtype=gram.dtype, device=device)
        assert (gram - target).abs().max().item() < 1e-4

    def test_hadamard_is_orthonormal(self, device):
        eye = torch.eye(N, dtype=torch.complex64, device=device)
        assert (_fwht(_fwht(eye)) - eye).abs().max().item() < 1e-5

    def test_rejects_non_power_of_two(self, device):
        try:
            srht(63, 20, _gen(device, 0), device=device)
            assert False
        except ValueError:
            pass


class TestMeasurementNoise:
    def test_power_scales_inversely_with_rho(self, device):
        # per-measurement power is n/m = 1/rho, so sigma^2 tracks it at fixed target SNR
        x = torch.randn(64, N, dtype=torch.complex64, device=device, generator=_gen(device, 1))
        x = x / x.abs().pow(2).mean(-1, keepdim=True).sqrt()
        powers = []
        for m in (16, 48):
            op = _op(device, 0, N, m)
            w = measurement_noise(measure(x, op), 20.0, op, _gen(device, 7))
            powers.append(w.abs().pow(2).mean().item() * (m / N))
        assert abs(powers[0] - powers[1]) / powers[0] < 0.15

    def test_none_is_noiseless(self, device):
        op = _op(device)
        x = torch.randn(4, N, dtype=torch.complex64, device=device, generator=_gen(device, 1))
        assert measurement_noise(measure(x, op), None, op, _gen(device, 7)).abs().max().item() == 0.0

    def test_hits_target_snr(self, device):
        x = torch.randn(64, N, dtype=torch.complex64, device=device, generator=_gen(device, 1))
        x = x / x.abs().pow(2).mean(-1, keepdim=True).sqrt()
        op = _op(device)
        y = measure(x, op)
        w = measurement_noise(y, 10.0, op, _gen(device, 7))
        got = 10 * math.log10(y.abs().pow(2).mean().item() / w.abs().pow(2).mean().item())
        assert abs(got - 10.0) < 1.5


class TestDeterminism:
    def test_same_seed_same_operator(self, device):
        o1 = _op(device, 0)
        o2 = _op(device, 0)
        assert torch.equal(o1.omega, o2.omega)
        assert torch.allclose(o1.theta, o2.theta)

    def test_different_seed_differs(self, device):
        o1 = _op(device, 0)
        o2 = _op(device, 1)
        assert not torch.allclose(o1.theta, o2.theta)
