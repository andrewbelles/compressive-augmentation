import math
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


PLANTED_B = 32
PLANTED_SHARE = 0.7


def _planted_classes(device, seed=3):
    # a shared sparse core per class plus an incompressible per-frame remainder, so the direction
    # separating two frames of one class is exactly the part the sparse model cannot represent
    frame = gabor_frame(N, 32, 8, device=device)
    g = _gen(device, seed)
    mods = _planted_meta()["mod"]
    cores = {}
    for mod in sorted(set(mods)):
        a = torch.zeros(frame.n_atoms, dtype=torch.complex64, device=device)
        idx = torch.randperm(frame.n_atoms, generator=g, device=device)[:20]
        a[idx] = torch.randn(20, dtype=torch.complex64, device=device, generator=g)
        cores[mod] = a
    core = synthesis(torch.stack([cores[m] for m in mods]), frame)
    rest = torch.randn(PLANTED_B, N, dtype=torch.complex64, device=device, generator=g)
    unit = lambda t: t / t.abs().pow(2).mean(-1, keepdim=True).sqrt()
    return PLANTED_SHARE * unit(core) + (1.0 - PLANTED_SHARE) * unit(rest)


def _planted_meta():
    return {"mod": ["A" if i % 2 == 0 else "B" for i in range(PLANTED_B)],
            "snr": [20] * PLANTED_B}


class TestDriversRun:
    def test_all_produce_stratified_records(self, device):
        frames = _sparse_frames(device)
        meta = _meta(frames.shape[0])
        for name, fn in REGISTRY.items():
            kw = {"draws": 3} if name == "operator_draw_variance" else {}
            recs = fn(frames, meta, [0.6, 0.9], seed=0, device=device, **kw)
            assert len(recs) >= 1
            assert all(isinstance(r, dict) and r for r in recs)
            # operator-algebra mechanisms are signal-independent, so they carry no SNR stratum
            if name not in ("operator_isometry", "operator_equivariance"):
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
        recs = [r for r in REGISTRY["kernel_geometry"](_sparse_frames(device), _meta(16),
                                                       [0.3, 0.95], 0, device)
                if r["arm"] == "cs"]
        low = [r["view_diversity"] for r in recs if r["rho"] == 0.3]
        high = [r["view_diversity"] for r in recs if r["rho"] == 0.95]
        assert sum(high) / len(high) < sum(low) / len(low)

    def test_merged_artifacts_carry_the_absorbed_columns(self, device):
        # the deleted mechanisms are scored from these rows, so the columns must survive the merge
        kg = REGISTRY["kernel_geometry"](_sparse_frames(device), _meta(16), [0.6], 0, device)[0]
        for col in ("label_retention", "view_diversity", "nuisance_collapse", "base_sep"):
            assert col in kg
        kn = REGISTRY["se_knee"](_sparse_frames(device), _meta(16), [0.4, 0.6, 0.8], 0, device)[0]
        for col in ("realized_snr", "se_snr", "gap", "eps", "eps_source", "kappa",
                    "mean_cumulant_dist", "delta_min", "delta_q10"):
            assert col in kn

    def test_kernel_geometry_null_holds_under_isotropic_corruption(self, device):
        # zeta_perp does not vanish in finite samples: independent complex noise in C^N correlates
        # at ~1/sqrt(N), so the tolerance is derived from N rather than hardcoded
        x = _sparse_frames(device, b=32)
        recs = REGISTRY["kernel_geometry"](x, _meta(32), [0.6], 0, device)
        awgn = [r["zeta_perp"] for r in recs if r["arm"] == "awgn"]
        cs = [r["zeta_perp"] for r in recs if r["arm"] == "cs"]
        assert abs(sum(awgn) / len(awgn)) < 3.0 / math.sqrt(N)
        assert sum(cs) / len(cs) > 3.0 / math.sqrt(N)

    def test_backprojection_has_no_residual_structure(self, device):
        # a linear arm sharing the operator ensemble isolates "same signal" from "same atoms".
        # rho=0.6 noiseless recovers to the numerical floor, where every error is roundoff and the
        # contrast is undefined, so this is read in a regime with real distortion
        recs = REGISTRY["kernel_geometry"](_sparse_frames(device, b=32), _meta(32), [0.25], 0,
                                           device, measurement_snr=20.0)
        by = {r["arm"]: r for r in recs}
        assert by["cs"]["achieved_distortion"] > 1e-3, "regime too easy to carry structure"
        assert by["backprojection"]["zeta_perp"] < by["cs"]["zeta_perp"]

    def test_debias_removes_the_shrinkage_gain(self, device):
        recs = REGISTRY["kernel_geometry"](_sparse_frames(device, b=32), _meta(32), [0.25], 0,
                                           device, measurement_snr=20.0)
        by = {r["arm"]: r for r in recs}
        # the refit is what the shrunk arm exists to be compared against
        assert abs(by["cs"]["gain"] - 1.0) < abs(by["cs_shrunk"]["gain"] - 1.0)
        assert by["cs"]["achieved_distortion"] < by["cs_shrunk"]["achieved_distortion"]

    def test_matched_denoiser_reconstructs_closer_than_the_shrunk_arm(self, device):
        # above the DT threshold, where recovery works and the estimators can be ranked at all
        recs = REGISTRY["kernel_geometry"](_sparse_frames(device, b=32), _meta(32), [0.6], 0,
                                           device, measurement_snr=20.0)
        by = {r["arm"]: r for r in recs}
        assert "cs_mmse" in by
        # the arm's reason to exist: no threshold to overshoot, so nothing is shrunk past its value
        assert by["cs_mmse"]["achieved_distortion"] < by["cs_shrunk"]["achieved_distortion"]
        assert by["cs_mmse"]["arm_vs_awgn_hi"] >= by["cs_mmse"]["arm_vs_awgn_lo"]

    def test_the_ratio_reduces_to_retained_energy_on_uncorrelated_pairs(self, device):
        # writing a view as g x + e gives ratio = g^2 + eps/(1 - corr) and energy = g^2 + eps, so
        # the two coincide when same-class frames are uncorrelated: the ratio is a gain statistic
        # and a value below 1 says nothing about the class, which is why no verdict scores it
        recs = REGISTRY["class_diameter"](_sparse_frames(device, b=32), _meta(32), [0.25], 0,
                                          device, measurement_snr=20.0)
        for r in recs:
            assert abs(r["diameter_ratio"] - r["retained_energy"]) < 0.25 * r["retained_energy"]

    def test_isotropic_error_leaves_no_alignment_at_matched_distortion(self, device):
        # the null the ratio could not express: awgn carries the same gain and error energy as cs,
        # so only a statistic reading the error's direction can tell them apart
        recs = REGISTRY["class_diameter"](_sparse_frames(device, b=32), _meta(32), [0.3], 0,
                                          device, measurement_snr=20.0)
        by = {r["arm"]: r for r in recs if r["mod"] == "A" and r["snr"] == 20}
        assert abs(by["awgn"]["class_alignment"]) < 0.05
        assert by["cs"]["class_alignment"] > 0.2

    def test_the_projection_outruns_the_linear_arms_on_planted_classes(self, device):
        # each class shares a sparse core and differs by an incompressible remainder, which is what
        # the sparse projection should discard first; at matched distortion the linear and dropout
        # arms have no reason to prefer that direction
        recs = REGISTRY["class_diameter"](_planted_classes(device), _planted_meta(), [0.3], 0,
                                          device, measurement_snr=20.0)
        by = {r["arm"]: r["class_alignment"] for r in recs if r["mod"] == "A"}
        assert by["cs"] > by["backprojection"]
        assert by["cs"] > by["dropout"]
        assert abs(by["awgn"]) < 0.05

    def test_every_class_survives_the_half_split(self, device):
        # a split that put a class in one half would leave its cross product empty and score nothing
        recs = REGISTRY["class_diameter"](_sparse_frames(device, b=32), _meta(32), [0.5], 0, device)
        assert {r["mod"] for r in recs} == {"A", "B"}
        assert all(r["n_pairs"] >= 4 for r in recs)

    def test_the_clean_diameter_is_the_same_denominator_for_every_arm_and_rho(self, device):
        recs = REGISTRY["class_diameter"](_sparse_frames(device, b=32), _meta(32), [0.12, 0.5], 0,
                                          device)
        for mod in ("A", "B"):
            vals = {r["clean_diameter"] for r in recs if r["mod"] == mod and r["snr"] == 10}
            assert len(vals) == 1

    def test_degeneracy_columns_track_the_collapse(self, device):
        # far below the sparsity ratio the solver returns almost nothing, which the columns must say
        recs = REGISTRY["kernel_geometry"](_sparse_frames(device, b=32), _meta(32), [0.03, 0.9], 0,
                                           device, measurement_snr=20.0)
        cs = {r["rho"]: r for r in recs if r["arm"] == "cs"}
        assert cs[0.03]["retained_energy"] < 0.2 < cs[0.9]["retained_energy"]
        assert cs[0.03]["m_over_k_eff"] < 1.0 < cs[0.9]["m_over_k_eff"]

    def test_empty_band_detected(self):
        assert admissible_band(0.8, 0.7, 0.6)["nonempty"] is False
        assert admissible_band(0.5, 0.5, 0.9)["nonempty"] is True
