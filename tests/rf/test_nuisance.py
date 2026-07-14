import torch

from rf.corruption import inject_cci, inject_impulsive
from rf.nuisance import orbit_augment, random_cfo, random_cyclic_shift, random_global_phase

B = 6
N = 256


def _gen(device, seed=0):
    return torch.Generator(device=device).manual_seed(seed)


def _rand_complex(device, seed=1):
    g = _gen(device, seed)
    re = torch.randn(B, N, generator=g, device=device)
    im = torch.randn(B, N, generator=g, device=device)
    return (re + 1j * im).to(torch.complex64)


class TestPhaseAndCfo:
    def test_phase_preserves_magnitude(self, device):
        x = _rand_complex(device)
        out = random_global_phase(x, _gen(device))
        assert torch.allclose(out.abs(), x.abs(), atol=1e-5)

    def test_cfo_preserves_magnitude(self, device):
        x = _rand_complex(device)
        out = random_cfo(x, 0.5, _gen(device))
        assert torch.allclose(out.abs(), x.abs(), atol=1e-5)

    def test_cfo_zero_eps_is_identity(self, device):
        x = _rand_complex(device)
        assert torch.allclose(random_cfo(x, 0.0, _gen(device)), x, atol=1e-6)


class TestCyclicShift:
    def test_value_multiset_preserved(self, device):
        x = _rand_complex(device)
        out = random_cyclic_shift(x, 32, _gen(device))
        a = torch.view_as_real(x).reshape(B, -1).sort(dim=1).values
        b = torch.view_as_real(out).reshape(B, -1).sort(dim=1).values
        assert torch.allclose(a, b, atol=1e-6)

    def test_known_shift(self, device):
        x = torch.zeros(1, N, dtype=torch.complex64, device=device)
        x[0, 0] = 1.0 + 1.0j
        gen = _gen(device, 0)
        out = random_cyclic_shift(x, 4, gen)
        assert out.abs().sum() == x.abs().sum()
        assert out[0].abs().argmax().item() in {(s % N) for s in range(-4, 5)}


class TestOrbitAugment:
    def test_shape_and_identity_copy(self, device):
        x = _rand_complex(device)
        out = orbit_augment(x, 4, 0.5, 32, _gen(device))
        assert out.shape == (4 * B, N)
        assert torch.equal(out[:B], x)

    def test_phase_equivariance(self, device):
        """Orbit of e^{j theta} x equals e^{j theta} orbit of x (same generator state)."""
        x = _rand_complex(device)
        theta = torch.exp(torch.tensor(1j * 1.3, device=device)).to(torch.complex64)
        a = orbit_augment(theta * x, 4, 0.5, 32, _gen(device, 5))
        b = theta * orbit_augment(x, 4, 0.5, 32, _gen(device, 5))
        assert torch.allclose(a, b, atol=1e-4)

    def test_magnitude_preserved_per_copy(self, device):
        x = _rand_complex(device)
        out = orbit_augment(x, 3, 0.5, 32, _gen(device))
        assert torch.allclose(out.norm(dim=1), x.norm(dim=1).repeat(3), atol=1e-4)


class TestCorruption:
    def test_impulsive_time_sparse(self, device):
        x = _rand_complex(device)
        out = inject_impulsive(x, n_bursts=2, burst_len=8, amp_rel=10.0, gen=_gen(device))
        diff = (out - x).abs()
        assert ((diff > 0).sum(dim=1) <= 2 * 8).all()
        assert (diff.sum(dim=1) > 0).all()

    def test_cci_adds_expected_power(self, device):
        x = _rand_complex(device)
        out = inject_cci(x, n_tones=2, sir_db=0.0, gen=_gen(device))
        p_sig = x.abs().pow(2).mean(dim=1)
        p_int = (out - x).abs().pow(2).mean(dim=1)
        assert torch.allclose(p_int, p_sig, rtol=0.3)

    def test_corruptions_phase_equivariant_in_signal(self, device):
        """Injected error is independent of x's phase only through RMS, so
        corrupting e^{j t} x with the same generator equals e^{j t}-rotating
        the signal part plus the same error term."""
        x = _rand_complex(device)
        e1 = inject_impulsive(x, 2, 8, 5.0, _gen(device, 7)) - x
        e2 = inject_impulsive(x * 1j, 2, 8, 5.0, _gen(device, 7)) - x * 1j
        assert torch.allclose(e1, e2, atol=1e-5)
