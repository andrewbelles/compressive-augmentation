import random

import torch


def enable_fast_matmul() -> None:
    """Allow TF32 in matmuls; complex matmul decomposes to real ones, so it applies there too."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs for a training run."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
