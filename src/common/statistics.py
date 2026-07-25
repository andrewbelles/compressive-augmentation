import torch

# class-separability statistics shared by domain measurement drivers

EPS = 1e-12


def scatter_ratio(features: torch.Tensor, labels) -> float:
    """Between-class over within-class scatter; 1.0 means labels carry no separation."""
    keys = sorted(set(labels))
    if len(keys) < 2:
        return float("nan")
    index = torch.as_tensor([keys.index(v) for v in labels], device=features.device)
    means, within, counts = [], [], []
    for c in range(len(keys)):
        rows = features[index == c]
        if rows.shape[0] == 0:
            continue
        mu = rows.mean(0)
        means.append(mu)
        within.append((rows - mu).pow(2).sum(-1).mean())
        counts.append(rows.shape[0])
    mu_all = torch.stack(means).mean(0)
    w = torch.stack(within).mean()
    b = torch.stack([(m - mu_all).pow(2).sum() for m in means]).mean()
    return (b / w.clamp_min(EPS)).item()
