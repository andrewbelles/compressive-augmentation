import torch

from dsp.frames import (
    gabor_frame,
    dft_frame,
    synthesis,
    analysis,
    is_tight,
)
from dsp.operators import mutual_coherence

N = 64


def _eff_support(frame):
    """Median participation ratio of atoms, in samples."""
    p = frame.d.abs().pow(2)
    p = p / p.sum(0, keepdim=True).clamp_min(1e-12)
    return (1.0 / p.pow(2).sum(0)).median().item()


class TestGaborTightness:
    def test_dd_star_is_gamma_identity(self, device):
        for window, hop in ((N, N // 2), (32, 8), (16, 8)):
            frame = gabor_frame(N, window, hop, device=device)
            assert is_tight(frame)
            assert frame.redundancy == frame.gamma

    def test_redundancy_is_nfreq_over_hop(self, device):
        frame = gabor_frame(N, 16, 8, device=device)
        assert frame.gamma == 2.0
        assert frame.n_atoms == (N // 8) * 16

    def test_synthesis_analysis_round_trip(self, device):
        # tight frame: D D* x = gamma x
        frame = gabor_frame(N, N, N // 2, device=device)
        x = torch.randn(4, N, dtype=torch.complex64, device=device)
        recon = synthesis(analysis(x, frame), frame) / frame.gamma
        assert torch.allclose(recon, x, atol=1e-4)

    def test_indivisible_hop_raises(self, device):
        for args in ((63, 32, 8), (N, 8, 16), (N, 32, 8, 16)):
            try:
                gabor_frame(*args, device=device)
                assert False
            except ValueError:
                pass


class TestTimeLocalization:
    def test_effective_support_tracks_window(self, device):
        # regression: window length must be independent of redundancy
        narrow = gabor_frame(N, 16, 8, device=device)
        wide = gabor_frame(N, N, N // 2, device=device)
        assert narrow.gamma == wide.gamma
        assert _eff_support(narrow) < 0.5 * _eff_support(wide)
        assert _eff_support(narrow) < 2.0 * 16

    def test_narrow_window_is_incoherent_with_dft(self, device):
        # null: a full-length window makes the frame a windowed DFT, coherence near 1
        dft = dft_frame(N, device=device)
        cols = lambda f: f.d / f.d.norm(dim=0, keepdim=True)
        overlap = lambda f: (cols(dft).conj().T @ cols(f)).abs().max(0).values.median().item()
        assert overlap(gabor_frame(N, 16, 8, device=device)) < 0.5
        assert overlap(gabor_frame(N, N, N // 2, device=device)) > 0.7


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

    def test_orthobasis_beats_redundant_frame_on_coherence(self, device):
        # mu is driven by the dictionary's own atom overlap, not by Phi
        assert mutual_coherence(dft_frame(N, device=device).d) < 1e-3
        assert mutual_coherence(gabor_frame(N, 16, 8, device=device).d) > 0.1
