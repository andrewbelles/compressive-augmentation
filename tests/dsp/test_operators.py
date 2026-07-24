import math

import torch

from dsp.operators import (
    random_convolution,
    measure,
    adjoint,
    backproject,
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
        frame = gabor_frame(N, gamma=2, device=device)
        a = measure(synthesis(torch.eye(frame.n_atoms, dtype=torch.complex64, device=device), frame), op)
        a = a.transpose(-1, -2)
        sv = torch.linalg.svdvals(a)
        nz = sv[sv > 1e-3]
        assert nz.numel() == M
        assert torch.allclose(nz, nz.mean().expand_as(nz), rtol=1e-3)

    def test_nontight_frame_spreads_spectrum(self, device):
        # Null: perturbing tightness spreads A's singular values
        op = _op(device)
        frame = gabor_frame(N, gamma=2, device=device)
        d = frame.d.clone()
        d[:, ::7] *= 3.0
        a = (op_dense := to_dense(op)) @ d
        sv = torch.linalg.svdvals(a)
        nz = sv[sv > 1e-3]
        assert nz.std() / nz.mean() > 0.05


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
