#!/usr/bin/env python3
"""Run one first-stage hypothesis over the rml2018 dataset and write measurement artifacts."""
import argparse
from pathlib import Path

import torch

from common.utils import enable_fast_matmul
from rf.hypotheses import REGISTRY, DATA_FREE
from rf.hypotheses._artifacts import DEFAULT_RATIOS, noise_tag, write_records
from rf.data import read_manifest, select_indices, load_frames
from rf.signal_model import FRAME_LEN


def _parse_ratios(s):
    return [float(x) for x in s.split(",")] if s else DEFAULT_RATIOS


def _parse_snrs(s):
    return [int(x) for x in s.split(",")] if s else None


def main() -> int:
    enable_fast_matmul()
    p = argparse.ArgumentParser(description="Run a first-stage CoAug hypothesis on rml2018.")
    p.add_argument("--mechanism", type=str, default="",
                   help="one mechanism, or several comma-separated to share one process")
    p.add_argument("--mechanisms", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ratios", type=str, default="")
    p.add_argument("--hdf5", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-group", type=int, default=32)
    p.add_argument("--snrs", type=str, default="")
    p.add_argument("--dictionary", type=str, default="")
    p.add_argument("--draws", type=int, default=0)
    p.add_argument("--measurement-snr", type=float, default=None,
                   help="measurement SNR in dB for y = Phi x + w; omit for a noiseless pipeline")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)

    names = [m for m in (args.mechanisms or args.mechanism).split(",") if m]
    unknown = [m for m in names if m not in REGISTRY]
    if not names or unknown:
        raise SystemExit(f"expected mechanisms from {sorted(REGISTRY)}, got {unknown or 'none'}")

    # loading once amortizes the HDF5 read, the dictionary build and the optimal_kappa cache
    if all(m in DATA_FREE for m in names):
        frames = torch.zeros(2, FRAME_LEN, dtype=torch.complex64, device=device)
        meta = {}
    else:
        rows = read_manifest(args.manifest)
        picked = select_indices(rows, args.per_group, _parse_snrs(args.snrs), args.seed)
        frames = load_frames(args.hdf5, [r["frame_idx"] for r in picked], device)
        meta = {"mod": [r["mod"] for r in picked], "snr": [r["snr"] for r in picked]}

    kw = {}
    if args.dictionary:
        kw["dictionary"] = args.dictionary
    if args.draws:
        kw["draws"] = args.draws
    if args.measurement_snr is not None:
        kw["measurement_snr"] = args.measurement_snr

    for name in names:
        fn = REGISTRY[name]
        accepted = fn.__code__.co_varnames[:fn.__code__.co_argcount]
        # never filter silently: a dropped measurement_snr would write noiseless rows tagged as noisy
        unconsumed = sorted(set(kw) - set(accepted))
        if unconsumed:
            raise SystemExit(f"{name} does not accept {unconsumed}; drop the flag or add it")
        records = fn(frames, meta, _parse_ratios(args.ratios), args.seed, device, **kw)
        tag = noise_tag(args.measurement_snr) if 'measurement_snr' in accepted else ''
        path = write_records(args.out, name, args.seed, records, tag)
        print(f"wrote {path}  rows={len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
