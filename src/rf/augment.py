import torch

from dsp.frames import Frame
from dsp.operators import ConvOperator, measure, measurement_noise, random_convolution
from dsp.recovery import debias, oamp, reconstruct
from dsp.state_evolution import optimal_kappa
from rf.signal_model import DEFAULT_DICTIONARY, build_dictionary, dither_sigma, frame_sparsity

# the compressive augmentation as deployed, one named map so the encoder and the hypothesis arms
# consume the same operator. The dither is a declared component, not a channel property: inside the
# admissible band a sigma = 0 round trip returns x to within 1e-3, so w is what sets view diversity.


def dither(x: torch.Tensor, op: ConvOperator, dither_snr, gen: torch.Generator) -> torch.Tensor:
    """Measurement dither w for one view; share one draw across views whose difference isolates x."""
    return measurement_noise(measure(x, op), dither_snr, op, gen)


def compressive_view(x: torch.Tensor, op: ConvOperator, frame: Frame, kappa: float,
                     w: torch.Tensor | None) -> torch.Tensor:
    """One augmented view x_tilde = D alpha_hat(Phi x + w); w carries no default."""
    y = measure(x, op)
    y = y if w is None else y + w
    return reconstruct(debias(oamp(y, op, frame, kappa=kappa), y, op, frame), frame)


def draw_view(x: torch.Tensor, rho: float, dither_snr, gen: torch.Generator,
              dictionary: str = DEFAULT_DICTIONARY) -> torch.Tensor:
    """Draw an operator and one view at rate rho, the entry point downstream training consumes."""
    n = x.shape[-1]
    frame = build_dictionary(dictionary, n, x.device)
    op = random_convolution(n, max(1, round(rho * n)), gen, device=x.device)
    kappa = optimal_kappa(rho, dither_sigma(dither_snr), frame_sparsity(frame, n),
                          frame.gamma, n)
    return compressive_view(x, op, frame, kappa, dither(x, op, dither_snr, gen))
