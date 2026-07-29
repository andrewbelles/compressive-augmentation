import torch

from dsp.frames import synthesis
from rf.signal_model import build_dictionary
from dsp.operators import draw_coherence, measure, random_convolution, screened_convolution
from dsp.recovery import debias, oamp, reconstruct, support_of

# debiasing must restore the amplitude the soft threshold shrinks away, without moving the support


def _planted(n, d, k, gen, device):
    """A k-sparse coefficient vector and its signal, amplitudes spread over a decade."""
    frame = build_dictionary("gabor_symbol", n, device)
    alpha = torch.zeros(2, frame.n_atoms, dtype=torch.complex64, device=device)
    for b in range(2):
        idx = torch.randperm(frame.n_atoms, generator=gen, device=device)[:k]
        mag = torch.linspace(1.0, 10.0, k, device=device)
        ph = torch.exp(2j * torch.pi * torch.rand(k, generator=gen, device=device))
        alpha[b, idx] = mag.to(torch.complex64) * ph.to(torch.complex64)
    return frame, alpha, synthesis(alpha, frame)


class TestDebias:
    def test_recovers_planted_amplitudes_better_than_the_shrunk_estimate(self, device):
        gen = torch.Generator(device=device).manual_seed(0)
        frame, alpha, x = _planted(256, None, 8, gen, device)
        op = random_convolution(256, 96, gen, device=device)
        y = measure(x, op)
        shrunk = oamp(y, op, frame, kappa=1.2)
        fixed = debias(shrunk, y, op, frame)
        # gain is the projection of the view onto the truth; shrinkage pulls it below 1
        def gain(a):
            v = reconstruct(a, frame)
            return ((v.conj() * x).sum(-1).real / x.abs().pow(2).sum(-1)).mean().item()
        assert gain(shrunk) < 0.99, "planted problem is too easy to exhibit shrinkage"
        assert abs(gain(fixed) - 1.0) < abs(gain(shrunk) - 1.0)

    def test_stays_inside_the_recovered_support(self, device):
        gen = torch.Generator(device=device).manual_seed(1)
        frame, _, x = _planted(256, None, 8, gen, device)
        op = random_convolution(256, 96, gen, device=device)
        y = measure(x, op)
        shrunk = oamp(y, op, frame, kappa=1.2)
        fixed = debias(shrunk, y, op, frame)
        # the refit may only rescale atoms the recovery already selected, never introduce one
        assert torch.equal(fixed.abs() > 0, support_of(shrunk, op.m))
        assert not (support_of(shrunk, op.m) & ~(shrunk.abs() > 0)).any()

    def test_reduces_measurement_residual(self, device):
        gen = torch.Generator(device=device).manual_seed(2)
        frame, _, x = _planted(256, None, 8, gen, device)
        op = random_convolution(256, 96, gen, device=device)
        y = measure(x, op)
        shrunk = oamp(y, op, frame, kappa=1.2)
        fixed = debias(shrunk, y, op, frame)
        res = lambda a: (y - measure(reconstruct(a, frame), op)).abs().pow(2).sum(-1).mean().item()
        assert res(fixed) <= res(shrunk) + 1e-6

    def test_empty_support_is_a_passthrough(self, device):
        gen = torch.Generator(device=device).manual_seed(3)
        frame, _, x = _planted(256, None, 8, gen, device)
        op = random_convolution(256, 96, gen, device=device)
        zero = torch.zeros(2, frame.n_atoms, dtype=torch.complex64, device=device)
        assert torch.equal(debias(zero, measure(x, op), op, frame), zero)


class TestScreenedConvolution:
    def test_screening_lowers_composite_coherence(self, device):
        gen = torch.Generator(device=device).manual_seed(0)
        frame = build_dictionary("gabor_symbol", 256, device)
        plain = [draw_coherence(random_convolution(256, 128, gen, device=device), frame)
                 for _ in range(8)]
        screened = screened_convolution(256, 128, gen, frame, tries=8, device=device)
        assert draw_coherence(screened, frame) <= sum(plain) / len(plain)

    def test_screened_draw_is_still_a_convolution(self, device):
        gen = torch.Generator(device=device).manual_seed(0)
        frame = build_dictionary("gabor_symbol", 256, device)
        op = screened_convolution(256, 128, gen, frame, tries=4, device=device)
        # unit-modulus spectrum is what makes Phi Phi* exact, and what the equivariance rests on
        assert torch.allclose(op.theta.abs(), torch.ones_like(op.theta.abs()), atol=1e-5)
        assert op.omega.numel() == 128
