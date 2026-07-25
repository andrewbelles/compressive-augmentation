import torch

from dsp.operators import random_convolution, to_dense, mutual_coherence
from rf.signal_model import DICTIONARIES, build_dictionary

# Layer III Prop 2/4: Phi is exactly row-orthogonal and A = Phi D is a scaled partial isometry.
# An implementation control, not evidence about the augmentation. mu(A) is reported against mu(D)
# because for a redundant D the coherence floor is set by the dictionary, not by Phi.


def run(frames, meta, ratios, seed, device) -> list[dict]:
    n = frames.shape[-1]
    records = []
    for name in DICTIONARIES:
        frame = build_dictionary(name, n, device)
        mu_d = mutual_coherence(frame.d)
        for rho in ratios:
            m = max(1, round(rho * n))
            gen = torch.Generator(device=device).manual_seed(seed)
            op = random_convolution(n, m, gen, device=device)
            phi = to_dense(op)
            gram = phi @ phi.conj().transpose(-1, -2)
            target = (n / m) * torch.eye(m, dtype=gram.dtype, device=device)
            a = phi @ frame.d
            sv = torch.linalg.svdvals(a)
            nz = sv[sv > 1e-3]
            records.append({
                "dictionary": name,
                "rho": rho,
                "m": m,
                "gram_max_err": (gram - target).abs().max().item(),
                "sv_cv": (nz.std() / nz.mean()).item(),
                "n_nonzero_sv": int(nz.numel()),
                "n_expected_sv": m,
                "mu_a": mutual_coherence(a),
                "mu_d": mu_d,
            })
    return records
