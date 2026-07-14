from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from rf.preprocess.manifests import MOD_CLASSES

EPS = 1e-12


def load_manifest(manifest_dir: Path, split: str) -> pd.DataFrame:
    """Load manifest_{split}.csv (columns frame_idx, mod, snr, split) as a DataFrame."""
    path = Path(manifest_dir) / f"manifest_{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    df = pd.read_csv(path)
    df["frame_idx"] = df["frame_idx"].astype(np.int64)
    df["snr"]       = df["snr"].astype(np.int64)
    return df


def select_frames(
    manifest: pd.DataFrame,
    per_cell: int,
    seed: int,
    snr_min: int | None = None,
    snr_values: list[int] | None = None,
    mods: list[str] | None = None,
) -> pd.DataFrame:
    """Deterministically subsample up to per_cell frames from each (mod, snr) cell.

    The per-cell RNG is keyed on (seed, mod index, snr) so the selection is
    independent of row/group ordering and of which other cells are requested.
    """
    df = manifest
    if snr_min is not None:
        df = df[df["snr"] >= snr_min]
    if snr_values is not None:
        df = df[df["snr"].isin(snr_values)]
    if mods is not None:
        df = df[df["mod"].isin(mods)]

    picks = []
    for (mod, snr), group in sorted(df.groupby(["mod", "snr"]), key=lambda kv: kv[0]):
        group = group.sort_values("frame_idx")
        n     = min(per_cell, len(group))
        rng   = np.random.default_rng([seed, MOD_CLASSES.index(mod), int(snr) + 1000])
        idx   = rng.choice(len(group), size=n, replace=False)
        picks.append(group.iloc[np.sort(idx)])
    if not picks:
        raise ValueError("select_frames: no frames match the given filters")
    return pd.concat(picks, ignore_index=True)


def load_complex_frames(
    hdf5_path: Path,
    frame_indices: np.ndarray,
    device: torch.device,
    normalize: bool = True,
) -> torch.Tensor:
    """Load frames from /X as complex64 x = I + jQ, shape [B, 1024], in input order.

    h5py fancy indexing requires sorted unique indices, so the slice is read in
    sorted order and the original order restored afterwards. When ``normalize``
    each frame is scaled to unit L2 norm (standard SRC preprocessing; SNR stays
    available as metadata).
    """
    idx = np.asarray(frame_indices, dtype=np.int64)
    order  = np.argsort(idx, kind="stable")
    sorted_idx = idx[order]
    with h5py.File(hdf5_path, "r") as f:
        data = f["X"][sorted_idx]                       # [B, 1024, 2] float32
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    data = data[inverse]
    x = torch.from_numpy(data[..., 0] + 1j * data[..., 1]).to(torch.complex64).to(device)
    if normalize:
        x = x / x.norm(dim=1, keepdim=True).clamp_min(EPS)
    return x
