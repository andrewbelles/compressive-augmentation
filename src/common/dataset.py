import numpy as np
import torch
from torch.utils.data import Dataset


class BaseBarlowDataset(Dataset):
    """Abstract base for Barlow Twins datasets.

    Subclasses implement load_sample and make_views. Both views returned by
    make_views must be independent draws from the same augmentation family and
    strength; the Barlow cross-correlation target is only valid under this
    exchangeability constraint.
    """

    _raw_only: bool = False

    def load_sample(self, index: int) -> np.ndarray:
        """Return the raw sample array for index without augmentation."""
        raise NotImplementedError

    def make_views(
        self,
        index: int,
        rng1: np.random.Generator,
        rng2: np.random.Generator,
    ) -> tuple:
        """Return (view1, view2) tensors from independent draws of the same augmentation."""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple:
        epoch_seed = int(torch.initial_seed()) % (2 ** 31) if getattr(self, "is_train", False) else 0
        if self._raw_only:
            y = self.load_sample(index)
            return (torch.from_numpy(np.asarray(y, dtype=np.float32)),)
        rng1 = np.random.default_rng([getattr(self, "seed", 0), index, epoch_seed, 1])
        rng2 = np.random.default_rng([getattr(self, "seed", 0), index, epoch_seed, 2])
        return self.make_views(index, rng1, rng2)
