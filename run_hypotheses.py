#!/usr/bin/env python3
"""Run one first-stage hypothesis over the rml2018 dataset and write measurement artifacts."""
import argparse
from pathlib import Path

import torch

from rf.hypotheses import REGISTRY, DATA_FREE
from rf.hypotheses._artifacts import DEFAULT_RATIOS, write_records
from rf.data import read_manifest, select_indices, load_frames
from rf.signal_model import FRAME_LEN


def _parse_ratios(s):
    return [float(x) for x in s.split(",")] if s else DEFAULT_RATIOS


def _parse_snrs(s):
    return [int(x) for x in s.split(",")] if s else None


def main() -> int:
    p = argparse.ArgumentParser(description="Run a first-stage CoAug hypothesis on rml2018.")
    p.add_argument("--mechanism", required=True, choices=list(REGISTRY))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ratios", type=str, default="")
    p.add_argument("--hdf5", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-group", type=int, default=32)
    p.add_argument("--snrs", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)

    if args.mechanism in DATA_FREE:
        frames = torch.zeros(2, FRAME_LEN, dtype=torch.complex64, device=device)
        meta = {}
    else:
        rows = read_manifest(args.manifest)
        picked = select_indices(rows, args.per_group, _parse_snrs(args.snrs), args.seed)
        frames = load_frames(args.hdf5, [r["frame_idx"] for r in picked], device)
        meta = {"mod": [r["mod"] for r in picked], "snr": [r["snr"] for r in picked]}

    records = REGISTRY[args.mechanism](frames, meta, _parse_ratios(args.ratios), args.seed, device)
    path = write_records(args.out, args.mechanism, args.seed, records)
    print(f"wrote {path}  rows={len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
