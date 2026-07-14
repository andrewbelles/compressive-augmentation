import torch

from csmath.operators import complex_gaussian_matrix
from csmath.solvers import (
    bpdn_batch,
    complex_soft_threshold,
    debias_batch,
    fista_batch,
    omp_batch,
    operator_norm_sq,
)

B = 8
M = 64
K = 128
SPARSITY = 4


def _gen(device, seed=0):
    return torch.Generator(device=device).manual_seed(seed)


def _sparse_problem(device, seed=0, noise=0.0):
    """Build a synthetic k-sparse complex recovery problem y = A alpha (+ noise)."""
    gen = _gen(device, seed)
    A = complex_gaussian_matrix(M, K, gen, device)
    alpha = torch.zeros(B, K, dtype=torch.complex64, device=device)
    for b in range(B):
        idx = torch.randperm(K, generator=gen, device=device)[:SPARSITY]
        re = torch.randn(SPARSITY, generator=gen, device=device)
        im = torch.randn(SPARSITY, generator=gen, device=device)
        alpha[b, idx] = ((re + 1j * im) + 2.0 * torch.sign(re)).to(torch.complex64)
    Y = alpha @ A.T
    if noise > 0:
        re = torch.randn(B, M, generator=gen, device=device)
        im = torch.randn(B, M, generator=gen, device=device)
        Y = Y + noise * (re + 1j * im).to(torch.complex64)
    return A, alpha, Y


def _objective(A, Y, alpha, lam):
    return (0.5 * (Y - alpha @ A.T).norm(dim=1).pow(2) + lam * alpha.abs().sum(dim=1))


class TestSoftThreshold:
    def test_preserves_phase(self, device):
        z = torch.tensor([3.0 * torch.exp(torch.tensor(1j * 0.9))], device=device).to(torch.complex64)
        out = complex_soft_threshold(z, 1.0)
        assert torch.allclose(out.abs(), torch.tensor([2.0], device=device), atol=1e-5)
        assert torch.allclose(out.angle(), z.angle(), atol=1e-5)

    def test_zero_safe(self, device):
        z = torch.zeros(4, dtype=torch.complex64, device=device)
        out = complex_soft_threshold(z, 1.0)
        assert torch.isfinite(out.real).all() and torch.isfinite(out.imag).all()
        assert torch.equal(out, z)

    def test_kills_small_entries(self, device):
        z = torch.tensor([0.5 + 0.5j, 4.0 + 0.0j], dtype=torch.complex64, device=device)
        out = complex_soft_threshold(z, 1.0)
        assert out[0].abs() == 0.0
        assert out[1].abs() > 0.0


class TestOperatorNorm:
    def test_matches_svd(self, device):
        A = complex_gaussian_matrix(M, K, _gen(device), device)
        est = operator_norm_sq(A, n_iter=100)
        exact = torch.linalg.matrix_norm(A, ord=2).pow(2).item()
        assert abs(est - exact) / exact < 1e-2


class TestFista:
    def test_exact_recovery(self, device):
        A, alpha_true, Y = _sparse_problem(device)
        lam = 1e-3 * (Y @ A.conj()).abs().amax()
        alpha, _ = fista_batch(A, Y, lam, n_iter=1000)
        rel = (alpha - alpha_true).norm(dim=1) / alpha_true.norm(dim=1)
        assert (rel < 0.05).all()

    def test_support_identification(self, device):
        A, alpha_true, Y = _sparse_problem(device)
        lam = 1e-3 * (Y @ A.conj()).abs().amax()
        alpha, _ = fista_batch(A, Y, lam, n_iter=1000)
        top = alpha.abs().topk(SPARSITY, dim=1).indices.sort(dim=1).values
        true = alpha_true.abs().topk(SPARSITY, dim=1).indices.sort(dim=1).values
        assert torch.equal(top, true)

    def test_objective_decreases_with_iterations(self, device):
        A, _, Y = _sparse_problem(device, noise=0.05)
        lam = 0.05 * (Y @ A.conj()).abs().amax()
        few, _ = fista_batch(A, Y, lam, n_iter=10)
        many, _ = fista_batch(A, Y, lam, n_iter=200)
        assert (_objective(A, Y, many, lam) <= _objective(A, Y, few, lam) + 1e-4).all()

    def test_per_frame_lambda(self, device):
        A, _, Y = _sparse_problem(device, noise=0.05)
        base = (Y @ A.conj()).abs().amax()
        lam = torch.full((B,), 0.05 * base.item(), device=device)
        lam[0] = 0.9 * base.item()          # near lam_max: frame 0 should be ~all zero
        alpha, _ = fista_batch(A, Y, lam, n_iter=200)
        n_active = (alpha.abs() > 1e-4).sum(dim=1)
        assert n_active[0] < n_active[1:].min()

    def test_precomputed_gram_matches(self, device):
        A, _, Y = _sparse_problem(device, noise=0.02)
        lam = 0.05
        a1, r1 = fista_batch(A, Y, lam, n_iter=100)
        a2, r2 = fista_batch(A, Y, lam, n_iter=100, L=operator_norm_sq(A),
                             gram=A.conj().T @ A, AhY=Y @ A.conj())
        assert torch.allclose(a1, a2, atol=1e-5)
        assert torch.allclose(r1, r2, atol=1e-5)


class TestBpdn:
    def test_residual_within_ball(self, device):
        A, _, Y = _sparse_problem(device, noise=0.05)
        eps = 0.3 * Y.norm(dim=1)
        _, res = bpdn_batch(A, Y, eps, n_iter=300)
        assert (res <= eps * 1.05).all()

    def test_sparser_than_tiny_lambda(self, device):
        A, _, Y = _sparse_problem(device, noise=0.05)
        eps = 0.3 * Y.norm(dim=1)
        alpha_bpdn, _ = bpdn_batch(A, Y, eps, n_iter=300)
        alpha_tiny, _ = fista_batch(A, Y, 1e-6, n_iter=300)
        assert alpha_bpdn.abs().sum() < alpha_tiny.abs().sum()


class TestDebias:
    def test_removes_shrinkage_bias(self, device):
        A, alpha_true, Y = _sparse_problem(device)
        lam = 0.05 * (Y @ A.conj()).abs().amax()
        alpha, _ = fista_batch(A, Y, lam, n_iter=500)
        biased = (alpha - alpha_true).norm(dim=1) / alpha_true.norm(dim=1)
        alpha_db = debias_batch(A, Y, alpha, k_max=SPARSITY)
        debiased = (alpha_db - alpha_true).norm(dim=1) / alpha_true.norm(dim=1)
        assert (debiased < 1e-4).all()
        assert (debiased < biased).all()

    def test_support_cap(self, device):
        A, _, Y = _sparse_problem(device, noise=0.05)
        alpha, _ = fista_batch(A, Y, 1e-4, n_iter=200)
        alpha_db = debias_batch(A, Y, alpha, k_max=6)
        assert ((alpha_db.abs() > 0).sum(dim=1) <= 6).all()


class TestOmp:
    def test_exact_recovery(self, device):
        A, alpha_true, Y = _sparse_problem(device)
        alpha = omp_batch(A, Y, SPARSITY)
        rel = (alpha - alpha_true).norm(dim=1) / alpha_true.norm(dim=1)
        assert (rel < 1e-3).all()

    def test_sparsity_respected(self, device):
        A, _, Y = _sparse_problem(device, noise=0.1)
        alpha = omp_batch(A, Y, SPARSITY)
        assert ((alpha.abs() > 0).sum(dim=1) <= SPARSITY).all()
