#!/usr/bin/env python3
"""
Index a RadioML 2018.01A HDF5 file and write deterministic train/val/test
split manifests.

The HDF5 file contains three datasets:
  /X  (2555904, 1024, 2)  float32  I/Q frames
  /Y  (2555904, 24)       float32  one-hot modulation labels
  /Z  (2555904,)          float64  SNR in dB

Only /Y and /Z are read here; /X is left untouched for the dataset loader.

Split strategy
--------------
Within each (mod, snr) group of 4096 frames the split is assigned by
``frame_idx_within_group % 10``:  0-6 → training, 7-8 → validation, 9 → test.
This guarantees every modulation class at every SNR level is represented in
all three splits (stratified split).

GPU prefetch note
-----------------
The manifests carry only integer frame indices into /X.  At training time a
dataset class uses these indices to slice /X directly.  On an H200 (80 GB
VRAM) the full /X tensor (~20 GB float32) can be loaded once at job start and
cached entirely on-device.  On a smaller GPU, per-batch HDF5 streaming is used
instead.  This is controlled by a ``--prefetch-gpu`` flag in the training
script, not here.
"""
import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

MOD_CLASSES = [
    "OOK", "ASK4", "ASK8", "BPSK", "QPSK", "PSK8", "PSK16", "PSK32",
    "APSK16", "APSK32", "APSK64", "APSK128",
    "QAM16", "QAM32", "QAM64", "QAM128", "QAM256",
    "AM_SSB_WC", "AM_SSB_SC", "AM_DSB_WC", "AM_DSB_SC",
    "FM", "GMSK", "OQPSK",
]

_SPLIT_MAP = {**{i: "training" for i in range(7)}, 7: "validation", 8: "validation", 9: "test"}
FIELDS = ["frame_idx", "mod", "snr", "split"]


def load_hdf5_index(hdf5_path: Path) -> list[dict]:
    """Read /Y and /Z from the HDF5 and return a list of row dicts.

    Only label and SNR arrays are loaded — /X (the 21 GB signal data) is
    not touched.
    """
    hdf5_path = Path(hdf5_path)
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"HDF5 not found: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as f:
        Y = f["Y"][:]
        Z = f["Z"][:]
    n = len(Y)
    if Y.shape[1] != len(MOD_CLASSES):
        raise ValueError(
            f"expected {len(MOD_CLASSES)} modulation classes, got {Y.shape[1]}"
        )
    mod_indices = np.argmax(Y, axis=1)
    snrs        = Z.reshape(-1).astype(np.int32)
    rows = []
    for i in range(n):
        rows.append({
            "frame_idx": i,
            "mod":       MOD_CLASSES[mod_indices[i]],
            "snr":       int(snrs[i]),
        })
    return rows


def _assign_splits(rows: list[dict]) -> list[dict]:
    """Assign deterministic stratified splits to each row in-place."""
    group_counter: dict[tuple, int] = {}
    for row in rows:
        key = (row["mod"], row["snr"])
        count = group_counter.get(key, 0)
        row["split"] = _SPLIT_MAP[count % 10]
        group_counter[key] = count + 1
    return rows


def write_manifests(hdf5_path: Path, output_dir: Path) -> dict[str, Path]:
    """Index the HDF5 and write four split manifest CSVs to output_dir.

    Returns a dict mapping split name to written Path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_hdf5_index(hdf5_path)
    rows = _assign_splits(rows)

    buckets: dict[str, list[dict]] = {"all": rows, "training": [], "validation": [], "test": []}
    for row in rows:
        buckets[row["split"]].append(row)

    manifest_paths = {}
    for name, row_list in buckets.items():
        path = output_dir / f"manifest_{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(row_list)
        manifest_paths[name] = path
        print(f"wrote manifest_{name}.csv  n={len(row_list)}", flush=True)
    return manifest_paths


def main() -> int:
    """CLI entry point for writing RML2018.01A split manifests."""
    parser = argparse.ArgumentParser(
        description="Index RadioML 2018.01A HDF5 and write split manifests."
    )
    parser.add_argument("--hdf5",       type=Path, required=True,
                        help="Path to GOLD_XYZ_OSC.0001_1024.hdf5")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write manifest_*.csv files")
    args = parser.parse_args()
    write_manifests(args.hdf5.expanduser().resolve(),
                    args.output_dir.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
