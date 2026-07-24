import csv
from pathlib import Path

import h5py
import numpy as np
import torch

# real rml2018 frame access: slice /X by manifest index, return complex IQ frames

FRAME_LEN = 1024


def read_manifest(path: Path) -> list[dict]:
    """Read a manifest CSV into row dicts with typed frame_idx and snr."""
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "frame_idx": int(row["frame_idx"]),
                "mod": row["mod"],
                "snr": int(row["snr"]),
                "split": row.get("split", ""),
            })
    return rows


def select_indices(rows: list[dict], per_group: int, snrs=None, seed: int = 0) -> list[dict]:
    """Sample up to per_group frames from each (mod, snr) group, optionally filtered by snr."""
    rng = np.random.default_rng(seed)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if snrs is not None and r["snr"] not in snrs:
            continue
        groups.setdefault((r["mod"], r["snr"]), []).append(r)
    picked = []
    for key in sorted(groups):
        g = groups[key]
        take = min(per_group, len(g))
        sel = rng.choice(len(g), size=take, replace=False)
        picked.extend(g[i] for i in sel)
    return picked


def load_frames(hdf5_path: Path, indices, device=None) -> torch.Tensor:
    """Load complex IQ frames (B, 1024) from /X for the given frame indices."""
    idx = np.asarray(indices, dtype=np.int64)
    order = np.argsort(idx)
    with h5py.File(Path(hdf5_path), "r") as f:
        raw = f["X"][idx[order]]
    restored = np.empty_like(raw)
    restored[order] = raw
    x = torch.from_numpy(restored.astype(np.float32))
    frames = torch.complex(x[..., 0], x[..., 1])
    return frames.to(device) if device is not None else frames
