import math

import pytest
import torch

from csmath.cs import (
    _gpu_dct_batch,
    _gpu_idct_batch,
    _gpu_wht_batch,
    gpu_dct_cs_view_batch,
    gpu_srht_batch,
)
from csmath.losses import barlow_twins_loss, off_diagonal, supcon_loss

B = 4
T = 512
D = 32


def _gen(device, seed=0):
    return torch.Generator(device=device).manual_seed(seed)


def _rand(device):
    return torch.randn(B, T, device=device)


def _rand_feats(device):
    return torch.randn(B, D, device=device)


class TestDCT:
    def test_round_trip(self, device):
        x = _rand(device)
        assert torch.allclose(_gpu_idct_batch(_gpu_dct_batch(x)), x, atol=1e-4)

    def test_energy_conservation(self, device):
        x = _rand(device)
        energy_in  = x.pow(2).sum(dim=-1)
        energy_out = _gpu_dct_batch(x).pow(2).sum(dim=-1)
        assert torch.allclose(energy_in, energy_out, atol=1e-3)


class TestWHT:
    def test_output_shape(self, device):
        p2 = 1 << math.ceil(math.log2(T))
        x  = torch.randn(B, p2, device=device)
        assert _gpu_wht_batch(x).shape == (B, p2)

    def test_involutory(self, device):
        p2 = 1 << math.ceil(math.log2(T))
        x  = torch.randn(B, p2, device=device)
        assert torch.allclose(_gpu_wht_batch(_gpu_wht_batch(x.clone())) / p2, x, atol=1e-4)


class TestDCTCS:
    def test_full_ratio_identity(self, device):
        x   = _rand(device)
        out = gpu_dct_cs_view_batch(x, ratio=100, gen=_gen(device), uniform=True)
        assert torch.allclose(out, x, atol=1e-4)

    def test_output_shape(self, device):
        for ratio in [1, 10, 50, 80]:
            out = gpu_dct_cs_view_batch(_rand(device), ratio=ratio, gen=_gen(device))
            assert out.shape == (B, T)

    def test_rng_determinism(self, device):
        x  = _rand(device)
        o1 = gpu_dct_cs_view_batch(x, ratio=20, gen=_gen(device, 0))
        o2 = gpu_dct_cs_view_batch(x, ratio=20, gen=_gen(device, 0))
        assert torch.allclose(o1, o2)

    def test_rng_independence(self, device):
        x  = _rand(device)
        o1 = gpu_dct_cs_view_batch(x, ratio=20, gen=_gen(device, 0))
        o2 = gpu_dct_cs_view_batch(x, ratio=20, gen=_gen(device, 1))
        assert not torch.allclose(o1, o2)

    def test_energy_bound(self, device):
        x     = _rand(device)
        out   = gpu_dct_cs_view_batch(x, ratio=20, gen=_gen(device))
        e_in  = x.pow(2).mean()
        e_out = out.pow(2).mean()
        assert e_out <= e_in * 4

    def test_biased_uniform_differ(self, device):
        x    = _rand(device)
        bias = gpu_dct_cs_view_batch(x, ratio=20, gen=_gen(device, 0), uniform=False)
        unif = gpu_dct_cs_view_batch(x, ratio=20, gen=_gen(device, 0), uniform=True)
        assert not torch.allclose(bias, unif)


class TestSRHT:
    def test_output_shape(self, device):
        for ratio in [1, 10, 50, 80]:
            out = gpu_srht_batch(_rand(device), ratio=ratio, gen=_gen(device))
            assert out.shape == (B, T)

    def test_full_ratio_energy(self, device):
        x   = _rand(device)
        out = gpu_srht_batch(x, ratio=100, gen=_gen(device))
        assert torch.allclose(x.pow(2).mean(), out.pow(2).mean(), rtol=0.15)

    def test_rng_determinism(self, device):
        x  = _rand(device)
        o1 = gpu_srht_batch(x, ratio=20, gen=_gen(device, 0))
        o2 = gpu_srht_batch(x, ratio=20, gen=_gen(device, 0))
        assert torch.allclose(o1, o2)

    @pytest.mark.parametrize("ratio", [1, 5, 10, 20, 50, 80])
    def test_no_nan_inf(self, device, ratio):
        out = gpu_srht_batch(_rand(device), ratio=ratio, gen=_gen(device))
        assert torch.isfinite(out).all()


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
        loss_diff, _, _ = barlow_twins_loss(z, z2,        lambd=1.0)
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
        c1 = torch.zeros(B // 2, D, device=device)
        c2 = torch.ones(B // 2, D, device=device) * 10.0
        feats_clustered = torch.cat([c1, c2])
        feats_random    = _rand_feats(device)
        labels = torch.cat([
            torch.zeros(B // 2, dtype=torch.long, device=device),
            torch.ones(B // 2,  dtype=torch.long, device=device),
        ])
        assert supcon_loss(feats_clustered, labels).item() < supcon_loss(feats_random, labels).item()

    def test_prenormalized_same_result(self, device):
        feats  = _rand_feats(device)
        norm   = torch.nn.functional.normalize(feats, dim=1)
        labels = torch.zeros(B, dtype=torch.long, device=device)
        assert torch.allclose(supcon_loss(feats, labels), supcon_loss(norm, labels), atol=1e-5)
