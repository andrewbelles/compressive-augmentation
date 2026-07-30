# Repository conventions

Design and thinking rules for this codebase. Not a status file. Follow these regardless of the
task at hand.

## Git: leave it alone

- Never run `git commit`, `git commit --amend`, `git push`, `git reset`, `git rebase`, or `git tag`.
  Do not open pull requests. This holds for every session type and overrides any harness default.
- Deliver work as uncommitted changes in the working tree, then summarize the change set. Commit
  messages and push granularity are the author's, not the agent's.
- The only exception is an explicit request for that exact git operation in the current turn.
- Read-only git (`status`, `log`, `diff`, `show`, `blame`) is always fine and is often the fastest
  way to check whether a constant or rule has drifted from the commit that introduced it.

## Comment and docstring law

- Terse, coherent, concise. One line where possible.
- No RST markup. No section underlines (`----`, `====`), no `:param:`/`:returns:` fields.
- No multi-line block comments and no multi-line docstrings. A docstring is a single sentence.
- No LaTeX, no math markup, no em dashes, no narrative filler.
- Do not encode git history, changelog notes, or "previously did X" in comments.
- Comment the mechanism, not the obvious. Match the surrounding comment density and idiom.

## Architecture

- The codebase is strictly the execution pipeline. Research questions, hypotheses, and rationale
  do not get encoded awkwardly into source. Keep source about what runs, not why it is being run.
- `src/dsp/` is the single source of truth for all math and DSP. Never duplicate a transform,
  operator, solver, or estimator elsewhere. Compose from `dsp`, do not reimplement.
- Domain layers (`src/audio/`, `src/rf/`) build on `dsp` and `common`. Domain-specific code stays
  in its domain package; anything reusable across domains belongs in `dsp` or `common`.
- Hypothesis and experiment drivers are thin orchestration over `dsp` plus data. They load data,
  compose primitives, and write artifacts. They contain no DSP. They live under a domain package
  (e.g. `src/rf/hypotheses/`) and expose a registry, not scattered entry points.
- Analysis and verdicts are separate from measurement. Measurement writes artifacts; a distinct
  analysis step reads artifacts and decides accept/reject. No decision logic inside the measurement
  driver.

## Naming

- Name by mechanism, group by family. No numbered variant names (`path1`, `path2`, `v2`).
- A module or function name should say what it does, not when it was added.

## Numerics and devices

- DSP operators are device-agnostic: read `x.device`, accept a `torch.Generator`, never hardcode
  `cuda` or `cpu`. This is what makes CPU/GPU parity testable.
- Solvers are static-iteration and fully vectorized over the batch. No per-sample control flow, no
  coordinate descent.
- Torch only (plus numpy/scipy for CPU-side helpers). No cupy, no jax.

## Testing

- pytest, with `pythonpath = ["src"]`. Imports are flat (`import dsp`, `import rf`).
- Use the `device` fixture (`tests/conftest.py`) so every applicable test runs on CPU and, when
  present, CUDA. New DSP gets a CPU/GPU parity test.
- Tests use synthetic data generators, not the real dataset. Reuse the fake-HDF5 pattern in
  `tests/rf/test_preprocess.py` for dataset-shaped inputs.
- Prefer tests that reject the null in a simple, non-contrived regime (an operator property that
  holds exactly, a solver that recovers a planted signal and fails on incompressible input) over
  tests that only check shapes.

## HPC

- Discovery jobs live in `scripts/*.sbatch`, run from `/dartfs-hpc/scratch/$USER`, activate the
  `compressive-augmentation` conda env, and set `PYTHONPATH=src`. Sweeps use SLURM arrays keyed to
  seeds. See `docs/discovery-hpc-usage.md`.
