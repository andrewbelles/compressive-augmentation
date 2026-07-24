import torch

from dsp.frames import gabor_frame, synthesis
from rf.hypotheses import REGISTRY
from rf.signal_model import admissible_band

N = 1024
GAMMA = 2


def _gen(device, seed=0):
    return torch.Generator(device=device).manual_seed(seed)


def _sparse_frames(device, b=16, k=20, seed=0):
    # frames that are genuinely sparse in the Gabor frame (two modulation groups)
    frame = gabor_frame(N, GAMMA, device=device)
    g = _gen(device, seed)
    alpha = torch.zeros(b, frame.n_atoms, dtype=torch.complex64, device=device)
    for i in range(b):
        idx = torch.randperm(frame.n_atoms, generator=g, device=device)[:k]
        alpha[i, idx] = torch.randn(k, dtype=torch.complex64, device=device, generator=g)
    return synthesis(alpha, frame)


def _meta(b):
    return {"mod": ["A" if i % 2 == 0 else "B" for i in range(b)],
            "snr": [10 if i < b // 2 else 20 for i in range(b)]}


class TestDriversRun:
    def test_all_produce_records(self, device):
        frames = _sparse_frames(device)
        meta = _meta(frames.shape[0])
        for name, fn in REGISTRY.items():
            ratios = [0.7] if name == "structure_vs_unstructured" else [0.6, 0.9]
            recs = fn(frames, meta, ratios, seed=0, device=device)
            assert len(recs) >= 1
            assert all(isinstance(r, dict) and r for r in recs)


class TestNullRejection:
    def test_operator_isometry_exact(self, device):
        recs = REGISTRY["operator_isometry"](_sparse_frames(device), {}, [0.5, 0.8], 0, device)
        for r in recs:
            assert r["gram_max_err"] < 1e-3
            assert r["sv_cv"] < 1e-2

    def test_gabor_beats_dft_on_structured(self, device):
        recs = REGISTRY["gabor_compressibility"](_sparse_frames(device), _meta(16), [], 0, device)
        assert all(r["k90_gabor"] < r["k90_dft"] for r in recs)

    def test_gabor_flat_on_white_noise(self, device):
        white = torch.randn(16, N, dtype=torch.complex64, device=device, generator=_gen(device, 9))
        recs = REGISTRY["gabor_compressibility"](white, _meta(16), [], 0, device)
        # incompressible: Gabor gives no advantage over DFT
        assert all(r["k90_gabor"] >= r["k90_dft"] for r in recs)

    def test_backprojection_beta_law_holds(self, device):
        recs = REGISTRY["structure_vs_unstructured"](_sparse_frames(device), {}, [0.5], 0, device)
        r = recs[0]
        assert abs(r["bp_frac_mean"] - r["bp_frac_pred"]) < 0.05

    def test_empty_band_detected(self):
        assert admissible_band(0.8, 0.7, 0.6)["nonempty"] is False
        assert admissible_band(0.5, 0.5, 0.9)["nonempty"] is True
