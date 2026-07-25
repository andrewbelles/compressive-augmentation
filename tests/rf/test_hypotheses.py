import torch

from dsp.frames import gabor_frame, synthesis
from rf.hypotheses import REGISTRY
from rf.hypotheses.dictionary_compressibility import _circular_span
from rf.signal_model import admissible_band

N = 256


def _gen(device, seed=0):
    return torch.Generator(device=device).manual_seed(seed)


def _sparse_frames(device, b=16, k=20, seed=0):
    # frames that are genuinely sparse in the symbol-scale frame (two modulation groups)
    frame = gabor_frame(N, 32, 8, device=device)
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
    def test_all_produce_stratified_records(self, device):
        frames = _sparse_frames(device)
        meta = _meta(frames.shape[0])
        for name, fn in REGISTRY.items():
            kw = {"draws": 3} if name == "operator_draw_variance" else {}
            recs = fn(frames, meta, [0.6, 0.9], seed=0, device=device, **kw)
            assert len(recs) >= 1
            assert all(isinstance(r, dict) and r for r in recs)
            if name != "operator_isometry":
                assert {r["snr"] for r in recs} == {10, 20}


class TestCircularSpan:
    def test_wrapped_support_is_contiguous(self, device):
        # the wraparound null: a DC-centred band spans both ends of the index range
        n = 64
        c = torch.zeros(1, n, device=device)
        c[0, :4] = 1.0
        c[0, -4:] = 1.0
        assert _circular_span(c, frac=0.9)[0].item() < 1.5

    def test_scattered_support_is_not_contiguous(self, device):
        n = 64
        c = torch.zeros(1, n, device=device)
        c[0, ::8] = 1.0
        assert _circular_span(c, frac=0.9)[0].item() > 4.0


class TestNullRejection:
    def test_operator_isometry_exact(self, device):
        recs = REGISTRY["operator_isometry"](_sparse_frames(device), {}, [0.5, 0.8], 0, device)
        for r in recs:
            assert r["gram_max_err"] < 1e-3
            assert r["sv_cv"] < 1e-2
            assert r["n_nonzero_sv"] == r["n_expected_sv"]

    def test_symbol_frame_sparser_than_dft_on_structured(self, device):
        recs = REGISTRY["dictionary_compressibility"](_sparse_frames(device), _meta(16), [], 0, device)
        by = {}
        for r in recs:
            by.setdefault(r["dictionary"], []).append(r["k90_over_d"])
        mean = {k: sum(v) / len(v) for k, v in by.items()}
        assert mean["gabor_symbol"] < mean["dft"]

    def test_white_noise_is_incompressible(self, device):
        white = torch.randn(16, N, dtype=torch.complex64, device=device, generator=_gen(device, 9))
        noise = REGISTRY["dictionary_compressibility"](white, _meta(16), [], 0, device)
        struct = REGISTRY["dictionary_compressibility"](_sparse_frames(device), _meta(16), [], 0, device)
        pick = lambda rs: sum(r["k90"] for r in rs if r["dictionary"] == "gabor_symbol")
        assert pick(noise) > 3.0 * pick(struct)

    def test_backprojection_beta_law_holds(self, device):
        recs = REGISTRY["backprojection_law"](_sparse_frames(device), _meta(16), [0.5], 0, device)
        for r in recs:
            assert abs(r["bp_frac_mean"] - r["bp_frac_pred"]) < 0.05

    def test_view_diversity_vanishes_at_full_rate(self, device):
        # null: at rho -> 1 the round trip is the identity, which is not an augmentation
        recs = REGISTRY["label_nuisance_tradeoff"](_sparse_frames(device), _meta(16),
                                                   [0.3, 0.95], 0, device)
        low = [r["view_diversity"] for r in recs if r["rho"] == 0.3]
        high = [r["view_diversity"] for r in recs if r["rho"] == 0.95]
        assert sum(high) / len(high) < sum(low) / len(low)

    def test_empty_band_detected(self):
        assert admissible_band(0.8, 0.7, 0.6)["nonempty"] is False
        assert admissible_band(0.5, 0.5, 0.9)["nonempty"] is True
