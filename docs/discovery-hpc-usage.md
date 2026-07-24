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

Submit the array (index = seed):

```
sbatch scripts/run_hypotheses.sbatch
```

Each array task runs all seven mechanisms for one seed and writes seed-tagged parquet to
`analysis/rf_hypotheses/`. Aggregate into verdicts once the array finishes:

```
sbatch --dependency=afterok:<ARRAY_JOB_ID> scripts/report_hypotheses.sbatch
```

Results:

```
cat analysis/rf_hypotheses/verdicts.csv          # accept/reject per hypothesis + GO/NO-GO
ls  analysis/rf_hypotheses/figures/              # calibration and related plots
```

## 6. Lines a $USER must check before submitting

Edit these in `scripts/run_hypotheses.sbatch` (and `acquire_rml2018.sbatch`,
`report_hypotheses.sbatch`) to match your allocation:

- `#SBATCH --account=free` — set to your Slurm account if `free` is not available to you.
- `#SBATCH --partition=h200_preemptable` / `--gres=gpu:h200:1` — change if you use a different GPU
  partition, or drop the gres and use `--partition=standard` to run CPU-only (slower).
- `#SBATCH --array=0-15` — the seed range. Widen for more seeds, narrow to test.
- `SCRATCH=/dartfs-hpc/scratch/$USER` and `REPO=$SCRATCH/compressive-augmentation` — change if your
  clone lives elsewhere.
- `CONDA_ENV=compressive-augmentation` — match your env name.
- `--per-group 64` — frames sampled per (mod, snr) group; lower for a fast smoke run.

## 7. Monitoring and teardown

```
squeue -u $USER                                  # queued/running jobs
tail -f logs/coaug-hypotheses_<jobid>_<task>.out # live progress
scancel <jobid>                                  # cancel a job or array
```

Safe to delete from scratch when done: `logs/`, `analysis/rf_hypotheses/*.parquet` (regenerable by
re-running), and the `.triton_cache/` and `.inductor_cache/` compile caches if present. Keep
`data/rml2018/` to avoid re-downloading ~21 GB.
