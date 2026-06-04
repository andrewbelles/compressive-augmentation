# Compressive-Augmentation for View-Based Learning 

## Abstract 

We study whether classical compressive sensing methods can provide effective augmentations for learning semantically meaningful music-genre representations. We use the "FMA small" dataset, a collection of 8,000, 30-second clips from songs split evenly across 8 genres. Augmentation is applied directly to waveforms, then converted to mel-spectrogram for encoding and projection onto a Barlow Twins objective. Compressive style augmentation performs competitively on F1-macro against traditional augmentation and exhibits stronger between-view alignment. Furthermore we show against a supervised reference manifold that higher magnitude nuisance perturbation corresponds to higher F1-macro scores. 

## Repository 

This repository defines the training, analysis, and visualization that defines the experiments outlined in my ENGS109 (Compressive-Sensing) project. Instructions below define how to use the repository in a general sense, but the `.sbatch` scripts assume you have access to Dartmouth College's discovery cluster (specifically their H200s). 

# Usage

## Setup

Install Python dependencies and make sure `ffmpeg` is available on `PATH`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Data

Place FMA audio and metadata under `preprocess/data/`:

```text
preprocess/data/fma_small/
preprocess/data/tracks.csv
```

or:

```text
preprocess/data/fma_small/
preprocess/data/fma_metadata/tracks.csv
```

Predecode audio and generate mel manifests:

```bash
python -m preprocess.decode_audio -d preprocess/data/fma_small
python -m preprocess.mel -d preprocess/data/fma_small --sample-images
```

## Train

Run both halves for the full sweep:

```bash
python train.py --half 0 --scratch-dir . \
  --data-dir preprocess/data/fma_small_mel \
  --audio-root preprocess/data

python train.py --half 1 --scratch-dir . \
  --data-dir preprocess/data/fma_small_mel \
  --audio-root preprocess/data
```

On SLURM, use:

```bash
sbatch run_train.sbatch
```

To run only selected families, add `--kinds`, for example:

```bash
python train.py --half 0 --scratch-dir . --kinds supcon traditional
```

## Analyze

```bash
python analyze.py \
  --parquet data/wave_barlow_fma_small.parquet \
  --output-dir analysis \
  --checkpoint-dir checkpoints \
  --audio-root preprocess/data/fma_small_mel
```

On SLURM, use:

```bash
sbatch run_analyze.sbatch
```

## Plot

```bash
python plot.py --analysis-dir analysis --output-dir images
```

## Outputs

- `checkpoints/`: trained model checkpoints
- `data/wave_barlow_fma_small.parquet`: extracted embeddings
- `analysis/`: analysis CSVs
- `images/`: generated figures

## Citation

This dataset was made possible by the work of Defferrard et al. If you use FMA downstream, credit the original dataset authors:

> Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson.  
> **"FMA: A Dataset for Music Analysis"**  
> *18th International Society for Music Information Retrieval Conference (ISMIR), 2017.*  
> [Official FMA GitHub Repository](https://github.com/mdeff/fma)
