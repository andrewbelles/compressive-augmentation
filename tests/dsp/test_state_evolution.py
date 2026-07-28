import torch

from dsp.frames import gabor_frame, synthesis
from dsp.operators import random_convolution, measure
from dsp.recovery import oamp, reconstruct
from dsp.state_evolution import (
    se_fixed_point,
    calibration_snr,
    beta_law_moments,
    _bernoulli_gaussian,
)

N = 128
GAMMA = 2
KAPPA = 1.6
EPS_SPIKE = 0.05


def _snr_db(x, xhat):
    num = x.abs().pow(2).sum()
    den = (x - xhat).abs().pow(2).sum().clamp_min(1e-12)
    return (10 * torch.log10(num / den)).item()


class TestCalibrationMonotone:
    def test_snr_increases_with_ratio(self):
        snrs = [calibration_snr(r, 0.1, EPS_SPIKE, GAMMA, N) for r in (0.4, 0.6, 0.8)]
        assert snrs[0] < snrs[1] < snrs[2]

    def test_tau_decreases_with_ratio(self):
        taus = [se_fixed_point(r, 0.1, EPS_SPIKE, GAMMA, N)["tau"] for r in (0.4, 0.6, 0.8)]
        assert taus[0] > taus[1] > taus[2]

    def test_fixed_point_self_consistent(self):
        fp = se_fixed_point(0.7, 0.1, EPS_SPIKE, GAMMA, N)
        coef = (1.0 - fp["delta"]) / fp["delta"]
        assert abs(fp["tau"] ** 2 - (0.1 ** 2 + coef * fp["mse"])) < 1e-3


class TestSEPredictsRecovery:
    def test_se_upper_bounds_realized_and_gap_closes(self, device):
        frame = gabor_frame(N, N, N // 2, device=device)
        d = frame.n_atoms
        sigma = 0.1
        gaps = {}
        for rho in (0.6, 0.9):
            m = round(rho * N)
            op = random_convolution(N, m, torch.Generator(device=device).manual_seed(3), device=device)
            alpha = _bernoulli_gaussian((128, d), EPS_SPIKE, torch.Generator(device=device).manual_seed(7), device)
            x = synthesis(alpha, frame)
            y = measure(x, op) + sigma * torch.randn(128, m, dtype=torch.complex64, device=device)
            realized = _snr_db(x, reconstruct(oamp(y, op, frame, kappa=KAPPA), frame))
            se = calibration_snr(rho, sigma, EPS_SPIKE, GAMMA, N)
            assert se + 1.0 >= realized
            gaps[rho] = se - realized
        assert gaps[0.9] < gaps[0.6]


class TestBetaLaw:
    def test_moments_match_samples(self, device):
        for rho in (0.4, 0.7):
            m = round(rho * N)
            op = random_convolution(N, m, torch.Generator(device=device).manual_seed(5), device=device)
            x = torch.randn(4000, N, dtype=torch.complex64, device=device)
            from dsp.operators import backproject
            e = x - backproject(measure(x, op), op)
            frac = e.abs().pow(2).sum(-1) / x.abs().pow(2).sum(-1)
            mom = beta_law_moments(N, m)
            assert abs(frac.mean().item() - mom["mean"]) < 0.02
            assert abs(frac.var().item() - mom["var"]) < 0.02
