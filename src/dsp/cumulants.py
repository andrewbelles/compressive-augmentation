import torch

# scale-normalized higher-order cumulant features T(x) for the label-margin test

EPS = 1e-12


def cumulant_features(x: torch.Tensor) -> torch.Tensor:
    """Return normalized cumulant magnitudes [C20,C40,C41,C42,C60,C63] per complex frame."""
    xc = x - x.mean(dim=-1, keepdim=True)
    conj = xc.conj()

    def m(p, q):
        return (xc.pow(p - q) * conj.pow(q)).mean(dim=-1)

    m20, m21 = m(2, 0), m(2, 1)
    m40, m41, m42 = m(4, 0), m(4, 1), m(4, 2)
    m60, m63 = m(6, 0), m(6, 3)

    c20, c21 = m20, m21.real.clamp_min(EPS)
    c40 = m40 - 3 * m20 * m20
    c41 = m41 - 3 * m20 * m21
    c42 = m42 - m20.abs().pow(2) - 2 * m21.pow(2)
    c60 = m60 - 15 * m20 * m40 + 30 * m20.pow(3)
    c63 = m63 - 6 * m20 * m41 - 9 * m21 * m42 + 18 * m20.pow(2) * m21 + 12 * m21.pow(3)

    feats = torch.stack([
        c20.abs() / c21,
        c40.abs() / c21.pow(2),
        c41.abs() / c21.pow(2),
        c42.abs() / c21.pow(2),
        c60.abs() / c21.pow(3),
        c63.abs() / c21.pow(3),
    ], dim=-1)
    return feats.real


def cumulant_distance(x: torch.Tensor, x_tilde: torch.Tensor) -> torch.Tensor:
    """Feature-space distance ||T(x_tilde) - T(x)|| per frame."""
    return (cumulant_features(x_tilde) - cumulant_features(x)).norm(dim=-1)
