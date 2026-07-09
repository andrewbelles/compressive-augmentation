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

B = 4
T = 512


def _gen(device, seed=0):
    return torch.Generator(device=device).manual_seed(seed)


def _rand(device):
    return torch.randn(B, T, device=device)


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
        x      = _rand(device)
        out    = gpu_dct_cs_view_batch(x, ratio=20, gen=_gen(device))
        e_in   = x.pow(2).mean()
        e_out  = out.pow(2).mean()
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
