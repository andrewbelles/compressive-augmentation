import torch

from dsp.frames import gabor_frame, synthesis
from dsp.operators import random_convolution
from dsp.recovery import sparse_code
from rf.hypotheses._matched import (
    awgn_view,
    backprojection_view,
    cs_view,
    distortion,
    dropout_view,
)

# the matched-strength arms are load-bearing for both GO gates, so they are tested on their own:
# a bug here would fail kernel_geometry and label_nuisance_tradeoff identically and look like a result

N = 256


def _gen(device, seed=0):
    return torch.Generator(device=device).manual_seed(seed)


def _frames(device, b=24):
    x = torch.randn(b, N, dtype=torch.complex64, device=device, generator=_gen(device, 1))
    return x / x.abs().pow(2).mean(-1, keepdim=True).sqrt()


class TestMatching:
    def test_awgn_hits_target_exactly(self, device):
        x = _frames(device)
        target = torch.full((x.shape[0],), 0.25, device=device)
        got = distortion(x, awgn_view(x, target, _gen(device, 2)))
        assert torch.allclose(got, target, rtol=1e-4)

    def test_awgn_matches_per_frame_not_just_on_average(self, device):
        x = _frames(device)
        target = torch.linspace(0.05, 0.5, x.shape[0], device=device)
        got = distortion(x, awgn_view(x, target, _gen(device, 2)))
        assert torch.allclose(got, target, rtol=1e-4)

    def test_backprojection_matches_via_beta_law(self, device):
        x = _frames(device)
        view, rho = backprojection_view(x, 0.3, 0, device)
        assert abs(rho - 0.7) < 1e-6
        assert abs(distortion(x, view).mean().item() - 0.3) < 0.05

    def test_dropout_bisection_converges(self, device):
        x = _frames(device)
        frame = gabor_frame(N, 32, 8, device=device)
        alpha, _ = sparse_code(x, frame, 0.05, max_iters=200)
        floor = distortion(x, synthesis(alpha, frame)).mean().item()
        target = floor + 0.25
        view, p = dropout_view(x, alpha, frame, target, 0, device)
        assert 0.0 <= p <= 1.0
        assert abs(distortion(x, view).mean().item() - target) < 0.05

    def test_cs_view_runs_with_and_without_noise(self, device):
        x = _frames(device)
        frame = gabor_frame(N, 32, 8, device=device)
        op = random_convolution(N, round(0.7 * N), _gen(device, 4), device=device)
        clean = cs_view(x, op, frame, 1.0, None, _gen(device, 5))
        noisy = cs_view(x, op, frame, 1.0, 15.0, _gen(device, 5))
        assert distortion(x, noisy).mean() > distortion(x, clean).mean()
