#!/usr/bin/env python3
#
# run_src_ladder.py  Andrew Belles  July 2026
#
# Full SRC / dictionary-learning baseline ladder (rungs 1-6) on RadioML
# 2018.01A. One job runs the whole ladder; two processes shard the spec list
# by --half and meet at a shard barrier before stage B. Completed shards are
# skipped, so preemption + resubmission resumes for free.
#
# Usage (two parallel processes, one per H200):
#   CUDA_VISIBLE_DEVICES=0 python run_src_ladder.py --half 0 \
#       --scratch-dir . --hdf5 data/rml2018/GOLD_XYZ_OSC.0001_1024.hdf5 \
#       --manifest-dir data/rml2018 &
#   CUDA_VISIBLE_DEVICES=1 python run_src_ladder.py --half 1 ... &
#   wait
#
# Pre-submit sanity check: add --smoke (tiny grid, ~1-2 min end to end).
#

import argparse
from pathlib import Path

import torch

from common.utils import set_seed
from rf.analysis import load_ladder_results, select_best_config
from rf.ladder import (
    BEST_SNR_MIN,
    LadderConfig,
    LadderSpec,
    build_context,
    build_stage_a,
    build_stage_b,
    energy_compaction_table,
    run_spec,
    save_shard,
    shard_exists,
    smoke_config,
    spec_name,
    wait_for_shards,
)


def _assigned(specs: list[LadderSpec], half: int | None) -> list[LadderSpec]:
    if half is None:
        return specs
    return [s for i, s in enumerate(specs) if i % 2 == half]


def _filter_rungs(specs: list[LadderSpec], rungs: list[int] | None) -> list[LadderSpec]:
    if rungs is None:
        return specs
    return [s for s in specs if s.rung in rungs]


def _run_stage(specs: list[LadderSpec], ctx) -> None:
    for i, spec in enumerate(specs):
        name = spec_name(spec)
        if shard_exists(ctx.results_dir, name):
            print(f"SKIP {name}", flush=True)
            continue
        print(f"[{i + 1}/{len(specs)}] START {name}", flush=True)
        df, meta = run_spec(spec, ctx)
        save_shard(df, meta, ctx.results_dir, name)
        print(f"SAVED {name} acc={meta['accuracy']:.4f} "
              f"wall={meta['wallclock_s']:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SRC baseline ladder (rungs 1-6).")
    parser.add_argument("--scratch-dir",  type=Path, required=True)
    parser.add_argument("--hdf5",         type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--half",         type=int, choices=[0, 1], default=None)
    parser.add_argument("--rungs",        type=int, nargs="+", default=None)
    parser.add_argument("--smoke",        action="store_true")
    parser.add_argument("--seed",         type=int, default=0)
    parser.add_argument("--best",         type=str, default=None,
                        help="manual stage-B config override: family,rho,pipeline")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scratch = args.scratch_dir.expanduser().resolve()
    cfg     = smoke_config(args.seed) if args.smoke else LadderConfig(seed=args.seed)
    suffix  = "_smoke" if args.smoke else ""
    results_dir = scratch / f"results/src_ladder{suffix}"
    dicts_dir   = scratch / f"dicts{suffix}"

    set_seed(cfg.seed)
    print(f"GPU half={args.half} device={device} smoke={args.smoke} "
          f"results={results_dir}", flush=True)

    ctx = build_context(cfg, args.hdf5.expanduser().resolve(),
                        args.manifest_dir.expanduser().resolve(),
                        dicts_dir, results_dir, device)
    print(f"test frames: {ctx.test_x.shape[0]}", flush=True)

    print("PSI energy-compaction check (per-class top-k energy fraction):", flush=True)
    print(energy_compaction_table(ctx).to_string(), flush=True)

    stage_a = _filter_rungs(build_stage_a(cfg), args.rungs)
    _run_stage(_assigned(stage_a, args.half), ctx)

    run_stage_b = args.rungs is None or any(r >= 4 for r in args.rungs)
    if run_stage_b:
        barrier = [spec_name(s) for s in build_stage_a(cfg) if s.rung in (2, 3)]
        wait_for_shards(barrier, ctx.results_dir,
                        cfg.barrier_timeout_s, cfg.barrier_poll_s)
        if args.best is not None:
            family, rho, pipeline = args.best.split(",")
            best = (family, float(rho), pipeline)
        else:
            best = select_best_config(
                load_ladder_results(ctx.results_dir, rungs=[2, 3]), snr_min=BEST_SNR_MIN)
        print(f"BEST family={best[0]} rho={best[1]:g} pipeline={best[2]}", flush=True)

        stage_b = _filter_rungs(build_stage_b(best, cfg), args.rungs)
        _run_stage(_assigned(stage_b, args.half), ctx)

    print(f"DONE half={args.half}", flush=True)


if __name__ == "__main__":
    main()
