import torch

from dsp.frames import (
    gabor_frame,
    dft_frame,
    synthesis,
    analysis,
    is_tight,
)

N = 64


class TestGaborTightness:
    def test_dd_star_is_gamma_identity(self, device):
        for gamma in (2, 4):
            frame = gabor_frame(N, gamma=gamma, device=device)
            assert is_tight(frame)
            assert frame.redundancy == gamma

    def test_synthesis_analysis_round_trip(self, device):
        # tight frame: D D* x = gamma x
        frame = gabor_frame(N, gamma=2, device=device)
        x = torch.randn(4, N, dtype=torch.complex64, device=device)
        recon = synthesis(analysis(x, frame), frame) / frame.gamma
        assert torch.allclose(recon, x, atol=1e-4)

    def test_indivisible_gamma_raises(self, device):
        try:
            gabor_frame(63, gamma=2, device=device)
            assert False
        except ValueError:
            pass


class TestDFTFrame:
    def test_is_orthonormal(self, device):
        frame = dft_frame(N, device=device)
        assert is_tight(frame)
        assert frame.gamma == 1.0

    def test_tone_has_contiguous_support(self, device):
        # a pure tone is one atom in DFT: recovery in DFT degenerates to a band
        frame = dft_frame(N, device=device)
        x = torch.exp(2j * torch.pi * 5 * torch.arange(N, device=device) / N)
        coeffs = analysis(x.unsqueeze(0), frame).abs().squeeze(0)
        support = (coeffs > 0.1 * coeffs.max()).nonzero().flatten()
        assert support.numel() <= 2
