import pytest
import torch

from csmath.losses import barlow_twins_loss, off_diagonal, supcon_loss

D = 32
B = 8


def _rand_feats(device):
    return torch.randn(B, D, device=device)


class TestOffDiagonal:
    def test_count(self, device):
        n   = 5
        mat = torch.ones(n, n, device=device)
        assert off_diagonal(mat).numel() == n * n - n

    def test_non_square_raises(self, device):
        with pytest.raises(ValueError):
            off_diagonal(torch.ones(3, 4, device=device))

    def test_identity_is_zero(self, device):
        mat = torch.eye(6, device=device)
        assert off_diagonal(mat).eq(0).all()


class TestBarlowTwinsLoss:
    def test_returns_three_scalars(self, device):
        z = _rand_feats(device)
        loss, on_diag, off_diag = barlow_twins_loss(z, z.clone(), lambd=5e-5)
        for t in (loss, on_diag, off_diag):
            assert t.shape == ()

    def test_identical_views_lower_loss(self, device):
        torch.manual_seed(0)
        z  = _rand_feats(device)
        z2 = _rand_feats(device)
        loss_same, _, _ = barlow_twins_loss(z, z.clone(), lambd=1.0)
        loss_diff, _, _ = barlow_twins_loss(z, z2,       lambd=1.0)
        assert loss_same.item() < loss_diff.item()

    def test_lambda_scales_off_diag(self, device):
        z1 = _rand_feats(device)
        z2 = _rand_feats(device)
        loss_low, _, _ = barlow_twins_loss(z1, z2, lambd=0.0)
        loss_hi,  _, _ = barlow_twins_loss(z1, z2, lambd=1.0)
        assert loss_hi.item() >= loss_low.item()

    def test_non_negative(self, device):
        z1 = _rand_feats(device)
        z2 = _rand_feats(device)
        loss, _, _ = barlow_twins_loss(z1, z2, lambd=5e-5)
        assert loss.item() >= 0.0


class TestSupConLoss:
    def test_non_negative(self, device):
        feats  = _rand_feats(device)
        labels = torch.zeros(B, dtype=torch.long, device=device)
        assert supcon_loss(feats, labels).item() >= 0.0

    def test_aligned_cluster_lower_loss(self, device):
        # two tight clusters perfectly separated should have lower loss than random features
        c1 = torch.zeros(B // 2, D, device=device)
        c2 = torch.ones(B // 2, D, device=device) * 10.0
        feats_clustered = torch.cat([c1, c2])
        feats_random    = _rand_feats(device)
        labels = torch.cat([
            torch.zeros(B // 2, dtype=torch.long, device=device),
            torch.ones(B // 2, dtype=torch.long, device=device),
        ])
        assert supcon_loss(feats_clustered, labels).item() < supcon_loss(feats_random, labels).item()

    def test_prenormalized_same_result(self, device):
        feats = _rand_feats(device)
        norm  = torch.nn.functional.normalize(feats, dim=1)
        labels = torch.zeros(B, dtype=torch.long, device=device)
        assert torch.allclose(supcon_loss(feats, labels), supcon_loss(norm, labels), atol=1e-5)
