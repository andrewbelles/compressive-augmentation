import random

import torch


def set_seed(seed: int) -> None:
    """
    Seed Python and PyTorch RNGs for a training run.

    Assumptions:
    - CUDA RNG seeding is only needed when CUDA is available.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
