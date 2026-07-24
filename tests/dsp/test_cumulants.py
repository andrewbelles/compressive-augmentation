import torch

from dsp.cumulants import cumulant_features, cumulant_distance

NSAMP = 4096
B = 32


def _gen(device, seed=0):
    return torch.Generator(device=device).manual_seed(seed)


def _bpsk(device, seed):
    g = _gen(device, seed)
    bits = torch.randint(0, 2, (B, NSAMP), generator=g, device=device) * 2 - 1
    return bits.to(torch.complex64)


def _qpsk(device, seed):
    g = _gen(device, seed)
    re = torch.randint(0, 2, (B, NSAMP), generator=g, device=device) * 2 - 1
    im = torch.randint(0, 2, (B, NSAMP), generator=g, device=device) * 2 - 1
    return (re + 1j * im).to(torch.complex64) / (2 ** 0.5)


class TestCumulants:
    def test_features_finite(self, device):
        feats = cumulant_features(_qpsk(device, 0))
        assert torch.isfinite(feats).all()
        assert feats.shape == (B, 6)

    def test_classes_separate(self, device):
        # between-class distance exceeds within-class spread
        fb = cumulant_features(_bpsk(device, 1)).mean(0)
        fq = cumulant_features(_qpsk(device, 2)).mean(0)
        within_b = (cumulant_features(_bpsk(device, 3)) - fb).norm(dim=-1).mean()
        between = (fb - fq).norm()
        assert between > 3.0 * within_b

    def test_stable_across_seeds(self, device):
        f1 = cumulant_features(_qpsk(device, 4)).mean(0)
        f2 = cumulant_features(_qpsk(device, 5)).mean(0)
        assert torch.allclose(f1, f2, atol=0.1)

    def test_distance_zero_for_identical(self, device):
        x = _qpsk(device, 6)
        assert cumulant_distance(x, x).abs().max().item() < 1e-4
