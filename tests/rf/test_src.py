import torch

from csmath.operators import complex_gaussian_matrix, identity_operator
from rf.corruption import inject_impulsive
from rf.src_classify import class_residuals, src_classify

N = 64
N_CLASSES = 3
ATOMS_PER_CLASS = 4
FRAMES_PER_CLASS = 10


def _gen(device, seed=0):
    return torch.Generator(device=device).manual_seed(seed)


def _orthogonal_class_problem(device, seed=0):
    """Three classes spanning disjoint coordinate blocks: perfectly separable by SRC."""
    gen = _gen(device, seed)
    blocks = [(0, 20), (20, 40), (40, 64)]
    atoms, atom_labels = [], []
    for c, (lo, hi) in enumerate(blocks):
        re = torch.randn(hi - lo, ATOMS_PER_CLASS, generator=gen, device=device)
        im = torch.randn(hi - lo, ATOMS_PER_CLASS, generator=gen, device=device)
        q, _ = torch.linalg.qr((re + 1j * im).to(torch.complex64))
        block = torch.zeros(N, ATOMS_PER_CLASS, dtype=torch.complex64, device=device)
        block[lo:hi] = q
        atoms.append(block)
        atom_labels.extend([c] * ATOMS_PER_CLASS)
    D = torch.cat(atoms, dim=1)
    labels_true, signals = [], []
    for c in range(N_CLASSES):
        re = torch.randn(FRAMES_PER_CLASS, ATOMS_PER_CLASS, generator=gen, device=device)
        im = torch.randn(FRAMES_PER_CLASS, ATOMS_PER_CLASS, generator=gen, device=device)
        coef = ((re + 1j * im) + torch.sign(re)).to(torch.complex64)
        x = coef @ D[:, c * ATOMS_PER_CLASS:(c + 1) * ATOMS_PER_CLASS].T
        signals.append(x / x.norm(dim=1, keepdim=True))
        labels_true.extend([c] * FRAMES_PER_CLASS)
    X = torch.cat(signals, dim=0)
    return D, torch.tensor(atom_labels, device=device), X, torch.tensor(labels_true, device=device)


class TestClassResiduals:
    def test_shape_and_true_class_smallest(self, device):
        D, atom_labels, X, y_true = _orthogonal_class_problem(device)
        result = src_classify(D, atom_labels, X, n_iter=300)
        res = class_residuals(D, atom_labels, X,
                              torch.zeros(X.shape[0], D.shape[1], dtype=X.dtype, device=device),
                              N_CLASSES)
        assert res.shape == (X.shape[0], N_CLASSES)
        assert torch.allclose(res, X.norm(dim=1, keepdim=True).expand_as(res).float(), atol=1e-5)
        assert torch.equal(result.pred, y_true)


class TestSrcClassify:
    def test_identity_operator_perfect(self, device):
        D, atom_labels, X, y_true = _orthogonal_class_problem(device)
        phi = identity_operator(N, device)
        result = src_classify(phi @ D, atom_labels, X @ phi.T, n_iter=300)
        assert torch.equal(result.pred, y_true)
        assert (result.margin > 0).all()

    def test_gaussian_half_rho_perfect(self, device):
        D, atom_labels, X, y_true = _orthogonal_class_problem(device)
        phi = complex_gaussian_matrix(N // 2, N, _gen(device, 9), device)
        A = phi @ D
        A = A / A.norm(dim=0, keepdim=True)
        result = src_classify(A, atom_labels, X @ phi.T, n_iter=300)
        assert torch.equal(result.pred, y_true)
        assert (result.margin > 0).all()

    def test_bpdn_solver(self, device):
        D, atom_labels, X, y_true = _orthogonal_class_problem(device)
        eps = 0.05 * X.norm(dim=1)
        result = src_classify(D, atom_labels, X, solver="bpdn", eps=eps, n_iter=300)
        assert torch.equal(result.pred, y_true)
        assert (result.res_norm <= eps * 1.05).all()

    def test_bpdn_requires_eps(self, device):
        D, atom_labels, X, _ = _orthogonal_class_problem(device)
        try:
            src_classify(D, atom_labels, X, solver="bpdn")
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestErrorDictionary:
    def test_error_dict_fixes_impulsive(self, device):
        D, atom_labels, X, y_true = _orthogonal_class_problem(device)
        Xc = inject_impulsive(X, n_bursts=2, burst_len=4, amp_rel=8.0, gen=_gen(device, 3))
        plain = src_classify(D, atom_labels, Xc, n_iter=300)
        robust = src_classify(D, atom_labels, Xc, n_iter=300, E=identity_operator(N, device))
        acc_plain  = (plain.pred == y_true).float().mean().item()
        acc_robust = (robust.pred == y_true).float().mean().item()
        assert acc_plain < 1.0
        assert acc_robust == 1.0
        assert acc_robust > acc_plain
