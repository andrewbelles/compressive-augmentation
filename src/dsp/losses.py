import torch
import torch.nn.functional as F

EPS = 1e-12


def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    """Return flattened off-diagonal entries of a square matrix."""
    n, m = matrix.shape
    if n != m:
        raise ValueError("expected square matrix")
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_twins_loss(
    left: torch.Tensor,
    right: torch.Tensor,
    lambd: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the Barlow Twins decorrelation objective and its on/off-diagonal components."""
    batch_size = left.size(0)
    left  = (left  - left.mean(dim=0))  / left.std(dim=0).clamp_min(EPS)
    right = (right - right.mean(dim=0)) / right.std(dim=0).clamp_min(EPS)
    correlation = left.T @ right / batch_size
    on_diag  = torch.diagonal(correlation).add_(-1.0).pow_(2).sum()
    off_diag = off_diagonal(correlation).pow_(2).sum()
    return on_diag + float(lambd) * off_diag, on_diag, off_diag


def supcon_loss(feats: torch.Tensor, labels: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    """Supervised contrastive loss over two-view feature batches."""
    feats   = F.normalize(feats, dim=1)
    sim     = feats @ feats.T / temp
    n       = feats.size(0)
    labels  = labels.view(-1, 1)
    mask    = (labels == labels.T).float()
    mask.fill_diagonal_(0)
    pos_sum = mask.sum(1).clamp_min(1)
    exp_sim = torch.exp(sim) * (1 - torch.eye(n, device=feats.device))
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True).clamp_min(EPS))
    return (-(mask * log_prob).sum(1) / pos_sum).mean()
