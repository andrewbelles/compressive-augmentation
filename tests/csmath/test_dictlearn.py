import torch

from csmath.dictlearn import ksvd, lc_ksvd
from csmath.solvers import omp_batch

N = 32
K_TRUE = 16
P = 200
SPARSITY = 3


def _gen(device, seed=0):
    return torch.Generator(device=device).manual_seed(seed)


def _planted_problem(device, seed=0):
    """Signals that are exact sparse combinations of a planted complex dictionary."""
    gen = _gen(device, seed)
    re = torch.randn(N, K_TRUE, generator=gen, device=device)
    im = torch.randn(N, K_TRUE, generator=gen, device=device)
    D_true = (re + 1j * im).to(torch.complex64)
    D_true = D_true / D_true.norm(dim=0, keepdim=True)
    X = torch.zeros(N, P, dtype=torch.complex64, device=device)
    for p in range(P):
        idx = torch.randperm(K_TRUE, generator=gen, device=device)[:SPARSITY]
        re = torch.randn(SPARSITY, generator=gen, device=device)
        im = torch.randn(SPARSITY, generator=gen, device=device)
        X[:, p] = D_true[:, idx] @ ((re + 1j * im) + torch.sign(re)).to(torch.complex64)
    return D_true, X


class TestKsvd:
    def test_unit_norm_atoms(self, device):
        _, X = _planted_problem(device)
        D, _, _ = ksvd(X, K_TRUE, SPARSITY, n_iter=3, gen=_gen(device))
        assert D.shape == (N, K_TRUE)
        assert D.dtype == torch.complex64
        assert torch.allclose(D.norm(dim=0), torch.ones(K_TRUE, device=device), atol=1e-4)

    def test_error_decreases(self, device):
        _, X = _planted_problem(device)
        _, _, hist = ksvd(X, K_TRUE, SPARSITY, n_iter=8, gen=_gen(device))
        assert len(hist) == 8
        assert hist[-1] < 0.5 * hist[0]
        for a, b in zip(hist, hist[1:]):
            assert b <= a * 1.05

    def test_beats_random_dictionary(self, device):
        _, X = _planted_problem(device)
        D, codes, hist = ksvd(X, K_TRUE, SPARSITY, n_iter=10, gen=_gen(device))
        gen = _gen(device, 99)
        re = torch.randn(N, K_TRUE, generator=gen, device=device)
        im = torch.randn(N, K_TRUE, generator=gen, device=device)
        D_rand = (re + 1j * im).to(torch.complex64)
        D_rand = D_rand / D_rand.norm(dim=0, keepdim=True)
        codes_rand = omp_batch(D_rand, X.T, SPARSITY).T
        err_rand = (X - D_rand @ codes_rand).norm() / X.norm()
        assert hist[-1] < 0.5 * err_rand.item()

    def test_codes_shape_and_reconstruction(self, device):
        _, X = _planted_problem(device)
        D, codes, hist = ksvd(X, K_TRUE, SPARSITY, n_iter=10, gen=_gen(device))
        assert codes.shape == (K_TRUE, P)
        rel = ((X - D @ codes).norm() / X.norm()).item()
        assert abs(rel - hist[-1]) < 1e-5


class TestLcKsvd:
    def _labeled_problem(self, device, seed=0):
        """Two classes living in disjoint coordinate halves."""
        gen = _gen(device, seed)
        X = torch.zeros(N, P, dtype=torch.complex64, device=device)
        labels = torch.zeros(P, dtype=torch.long, device=device)
        half = N // 2
        for p in range(P):
            c = p % 2
            re = torch.randn(half, generator=gen, device=device)
            im = torch.randn(half, generator=gen, device=device)
            sl = slice(0, half) if c == 0 else slice(half, N)
            X[sl, p] = (re + 1j * im).to(torch.complex64)
            labels[p] = c
        return X / X.norm(dim=0, keepdim=True), labels

    def test_shapes_and_labels(self, device):
        X, labels = self._labeled_problem(device)
        D, atom_labels = lc_ksvd(X, labels, atoms_per_class=4, sparsity=2,
                                 alpha_lc=1.0, n_iter=3, gen=_gen(device))
        assert D.shape == (N, 8)
        assert atom_labels.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
        assert torch.allclose(D.norm(dim=0), torch.ones(8, device=device), atol=1e-4)

    def test_same_class_code_concentration(self, device):
        X, labels = self._labeled_problem(device)
        D, atom_labels = lc_ksvd(X, labels, atoms_per_class=4, sparsity=2,
                                 alpha_lc=1.0, n_iter=5, gen=_gen(device))
        codes = omp_batch(D, X.T, 2)                      # [P, K]
        energy = codes.abs().pow(2)
        same  = energy[:, :4].sum(dim=1) * (labels == 0) + energy[:, 4:].sum(dim=1) * (labels == 1)
        other = energy[:, 4:].sum(dim=1) * (labels == 0) + energy[:, :4].sum(dim=1) * (labels == 1)
        assert (same.sum() > 10.0 * other.sum()).item()
