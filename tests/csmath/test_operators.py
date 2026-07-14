import math

import pytest
import torch

from csmath.operators import (
    build_sensing_matrix,
    complex_gaussian_matrix,
    identity_operator,
    mutual_coherence,
    partial_fourier_matrix,
    random_demodulator_matrix,
)

N = 128
M = 32
FAMILIES = ["gaussian", "demod", "fourier"]


def _gen(device, seed=0):
    return torch.Generator(device=device).manual_seed(seed)


def _rand_complex(device, *shape, seed=1):
    g = _gen(device, seed)
    re = torch.randn(*shape, generator=g, device=device)
    im = torch.randn(*shape, generator=g, device=device)
    return (re + 1j * im).to(torch.complex64)


class TestShapesAndDtypes:
    @pytest.mark.parametrize("family", FAMILIES)
    def test_shape_dtype(self, device, family):
        phi = build_sensing_matrix(family, M / N, N, seed=0, device=device)
        assert phi.shape == (M, N)
        assert phi.dtype == torch.complex64

    def test_identity(self, device):
        phi = build_sensing_matrix("identity", 1.0, N, seed=0, device=device)
        assert torch.equal(phi, identity_operator(N, device))
        x = _rand_complex(device, N)
        assert torch.allclose(phi @ x, x)

    def test_any_m_works_for_demod(self, device):
        for m in (1, 7, 100, N):
            phi = random_demodulator_matrix(m, N, _gen(device), device)
            assert phi.shape == (m, N)

    def test_unknown_family_raises(self, device):
        with pytest.raises(ValueError):
            build_sensing_matrix("wavelet", 0.5, N, seed=0, device=device)


class TestDeterminism:
    @pytest.mark.parametrize("family", FAMILIES)
    def test_same_seed_same_matrix(self, device, family):
        a = build_sensing_matrix(family, 0.25, N, seed=3, device=device)
        b = build_sensing_matrix(family, 0.25, N, seed=3, device=device)
        assert torch.equal(a, b)

    @pytest.mark.parametrize("family", FAMILIES)
    def test_different_seed_differs(self, device, family):
        a = build_sensing_matrix(family, 0.25, N, seed=3, device=device)
        b = build_sensing_matrix(family, 0.25, N, seed=4, device=device)
        assert not torch.equal(a, b)


class TestAdjoint:
    @pytest.mark.parametrize("family", FAMILIES)
    def test_adjoint_identity(self, device, family):
        phi = build_sensing_matrix(family, M / N, N, seed=0, device=device)
        x = _rand_complex(device, N, seed=2)
        y = _rand_complex(device, M, seed=3)
        lhs = torch.vdot(y, phi @ x)
        rhs = torch.vdot(phi.conj().T @ y, x)
        assert torch.allclose(lhs, rhs, atol=1e-3)


class TestPhaseEquivariance:
    """The I/Q-jointness check: op(e^{j theta} x) == e^{j theta} op(x)."""

    @pytest.mark.parametrize("family", FAMILIES + ["identity"])
    def test_phase_equivariance(self, device, family):
        phi = build_sensing_matrix(family, M / N, N, seed=0, device=device)
        x = _rand_complex(device, N, seed=5)
        theta = torch.exp(torch.tensor(1j * 0.7, device=device)).to(torch.complex64)
        assert torch.allclose(phi @ (theta * x), theta * (phi @ x), atol=1e-4)


class TestStatistics:
    def test_gaussian_column_energy(self, device):
        phi = complex_gaussian_matrix(512, 256, _gen(device), device)
        energies = phi.abs().pow(2).sum(dim=0)
        assert torch.allclose(energies.mean(), torch.tensor(1.0, device=device), atol=0.05)

    def test_partial_fourier_rows_orthonormal(self, device):
        phi = partial_fourier_matrix(M, N, _gen(device), device)
        eye = (N / M) * torch.eye(M, dtype=torch.complex64, device=device)
        assert torch.allclose(phi @ phi.conj().T, eye, atol=1e-3)

    def test_demod_matches_bruteforce(self, device):
        m, n = 8, 32
        gen = _gen(device, 7)
        phi = random_demodulator_matrix(m, n, gen, device)
        chips = phi.sum(dim=0)                       # one nonzero per column
        assert torch.allclose(chips.abs(), torch.ones(n, device=device), atol=1e-6)
        x = _rand_complex(device, n, seed=9)
        expected = torch.zeros(m, dtype=torch.complex64, device=device)
        for i in range(n):
            expected[(i * m) // n] += chips[i] * x[i]
        assert torch.allclose(phi @ x, expected, atol=1e-4)


class TestMutualCoherence:
    def test_orthonormal_self_coherence_zero(self, device):
        eye = identity_operator(N, device)
        assert mutual_coherence(eye) < 1e-6

    def test_range_and_scale_invariance(self, device):
        A = _rand_complex(device, M, 16, seed=1)
        mu = mutual_coherence(A)
        assert 0.0 <= mu <= 1.0 + 1e-5
        assert math.isclose(mu, mutual_coherence(3.0 * A), rel_tol=1e-4)

    def test_cross_coherence_identical_bases(self, device):
        A = _rand_complex(device, M, 16, seed=1)
        assert mutual_coherence(A, A) == pytest.approx(1.0, abs=1e-4)
