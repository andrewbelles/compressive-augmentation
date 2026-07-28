# Discovery HPC usage

End-to-end setup for running the CoAug Stage-1 hypothesis battery on Dartmouth Discovery.
Steps run in order. `$USER` is your Discovery netid.

## 1. Prerequisites

- A Discovery account with ssh access: `ssh $USER@discovery.dartmouth.edu`.
- Scratch space at `/dartfs-hpc/scratch/$USER` (large, fast, not backed up). All work lives here,
  not in `$HOME` (small quota, slow for large IO).
- A Kaggle API token for dataset download. On your laptop, create it at
  kaggle.com/settings > API > Create New Token, then copy it to Discovery:

```
scp kaggle.json $USER@discovery.dartmouth.edu:~/.kaggle/kaggle.json
ssh $USER@discovery.dartmouth.edu chmod 600 ~/.kaggle/kaggle.json
```

## 2. Repo setup

Clone into scratch (not home):

```
cd /dartfs-hpc/scratch/$USER
git clone <repo-url> compressive-augmentation
cd compressive-augmentation
```

The package imports flat from `src/` (`import dsp`, `import rf`). Scripts set `PYTHONPATH=src`, so
`pip install -e .` is optional; install it only if you want the package importable without the env
var.

## 3. Conda environment

```
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda create -y -n compressive-augmentation python=3.12
conda activate compressive-augmentation
pip install -r requirements.txt
pip install -e .          # optional, see above
```

Verify torch sees a GPU on a real GPU node (login nodes have none):

```
srun --account=free --partition=h200_preemptable --gres=gpu:h200:1 --time=00:05:00 --pty \
    python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 4. Data acquisition

Download and index RadioML 2018.01A in one job:

```
sbatch scripts/acquire_rml2018.sbatch
```

This downloads via Kaggle (needs compute-node internet egress, available on Discovery), or
decompresses a pre-staged `data/rml2018/rml2018.hdf5.zst` if you rsynced one, then writes split
manifests. If compute nodes lack egress, use the fallback: download and compress locally
(`bash scripts/ingest_rml2018.sh data/ --compress`), rsync the `.zst` to
`/dartfs-hpc/scratch/$USER/compressive-augmentation/data/rml2018/`, then submit the job.

Confirm the data landed:

```
ls -lh data/rml2018/GOLD_XYZ_OSC.0001_1024.hdf5 data/rml2018/manifest_all.csv
```

## 5. Run the hypothesis battery

Submit the array:

```
sbatch scripts/run_hypotheses.sbatch
```

The array index decomposes as `seed = id / 4`, `measurement_snr = (30, 20, 10, noiseless)[id % 4]`,
so 64 tasks cover 16 seeds at four noise levels. Artifacts are named
`{mechanism}_seed{NNN}_{tag}.parquet` so tasks sharing a seed do not clobber each other. Mechanisms
that ignore the noise axis run only on `id % 4 == 0`. Nine drivers write artifacts; three further
verdicts are scored from them, since one reconstruction serves several claims. Aggregate once the
array finishes:

```
sbatch --dependency=afterok:<ARRAY_JOB_ID> scripts/report_hypotheses.sbatch
```

Results:

```
cat analysis/rf_hypotheses/verdicts.csv          # accept/reject per hypothesis + GO/NO-GO
ls  analysis/rf_hypotheses/figures/              # tradeoff, calibration, compressibility, margin
```

Rows marked `informational` are controls (`operator_isometry`, `backprojection_law`) and theory
(`admissible_band`); they validate the implementation and do not vote. **GO requires both gates:**
`kernel_geometry` (the CS kernel is geometrically distinct from AWGN and from back-projection at
matched distortion) and `label_nuisance_tradeoff` (its retention curve beats AWGN at matched view
diversity). Decision figures are `figures/kernel_geometry.png` and
`figures/retention_vs_diversity.png`.

A cheap high-SNR pass, useful before committing to the full array:

```
PYTHONPATH=src python run_hypotheses.py --mechanism dictionary_compressibility --seed 0 \
    --hdf5 "$HDF5" --manifest "$MANIFEST" --out "$ARTIFACTS" \
    --snrs 20,22,24,26,28,30 --per-group 16
```

## 6. Lines a $USER must check before submitting

Edit these in `scripts/run_hypotheses.sbatch` (and `acquire_rml2018.sbatch`,
`report_hypotheses.sbatch`) to match your allocation:

- `#SBATCH --account=free` — set to your Slurm account if `free` is not available to you.
- `#SBATCH --partition=h200_preemptable` / `--gres=gpu:h200:1` — change if you use a different GPU
  partition, or drop the gres and use `--partition=standard` to run CPU-only (slower).
- `#SBATCH --array=0-63` — 16 seeds x 4 measurement-SNR levels. Narrow to `0-3` for one seed.
- `SCRATCH=/dartfs-hpc/scratch/$USER` and `REPO=$SCRATCH/compressive-augmentation` — change if your
  clone lives elsewhere.
- `CONDA_ENV=compressive-augmentation` — match your env name.
- `--per-group 64` — frames sampled per (mod, snr) group; lower for a fast smoke run.
- `--mechanisms a,b,c` — run several in one process, sharing the HDF5 read, the
  dictionary build and the state-evolution cache. `--mechanism` remains as an alias.
- `--draws 64` — operator draws for `operator_draw_variance`. The prediction is an 11-17% variance
  difference, so a small draw count cannot resolve it; lower this only for smoke runs.
- `--snrs` — restrict to an SNR subset (e.g. `20,22,24,26,28,30`). Artifacts stay stratified either
  way; this only limits which strata are computed.
- `--dictionary` — override the dictionary arm (`dft`, `windowed_dft`, `gabor_symbol`) for the
  mechanisms that take one. `dictionary_compressibility` and `operator_isometry` always sweep all
  three.
- `--measurement-snr` — dB level for `w` in `y = Phi x + w`; omit for a noiseless pipeline. This is
  the second knob: `rho` sets the structural floor, `sigma` sets the ceiling. Without it the
  recovery curve runs away as `rho -> 1` and `se_knee` cannot find an interior maximum. A driver
  that does not declare the flag now raises rather than silently running noiseless.

## 7. Monitoring and teardown

```
squeue -u $USER                                  # queued/running jobs
tail -f logs/coaug-hypotheses_<jobid>_<task>.out # live progress
scancel <jobid>                                  # cancel a job or array
```

Safe to delete from scratch when done: `logs/`, `analysis/rf_hypotheses/*.parquet` (regenerable by
re-running), and the `.triton_cache/` and `.inductor_cache/` compile caches if present. Keep
`data/rml2018/` to avoid re-downloading ~21 GB.
