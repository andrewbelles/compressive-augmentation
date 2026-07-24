import torch

from dsp.operators import random_convolution, measure, to_dense
from dsp.frames import gabor_frame, synthesis

# Layer III Prop 2/4: Phi is exactly row-orthogonal and A=Phi D is a scaled partial isometry.
# Accept: Phi Phi* = (N/m) I to machine tolerance and A's nonzero singular values are equal.

GAMMA = 2


def run(frames, meta, ratios, seed, device) -> list[dict]:
    n = frames.shape[-1]
    frame = gabor_frame(n, GAMMA, device=device)
    eye_d = torch.eye(frame.n_atoms, dtype=torch.complex64, device=device)
    records = []
    for rho in ratios:
        m = max(1, round(rho * n))
        op = random_convolution(n, m, torch.Generator(device=device).manual_seed(seed), device=device)
        phi = to_dense(op)
        gram = phi @ phi.conj().transpose(-1, -2)
        target = (n / m) * torch.eye(m, dtype=gram.dtype, device=device)
        a = measure(synthesis(eye_d, frame), op).transpose(-1, -2)
        sv = torch.linalg.svdvals(a)
        nz = sv[sv > 1e-3]
        records.append({
            "rho": rho,
            "m": m,
            "gram_max_err": (gram - target).abs().max().item(),
            "sv_cv": (nz.std() / nz.mean()).item(),
            "n_nonzero_sv": int(nz.numel()),
        })
    return records
