# Compressive Sensing on Mel-Spectrogram Representations for FMA Music

This repository studies compressive sensing (CS) directly on mel-spectrogram images and evaluates whether sparse codes recovered from highly compressed measurements retain enough structure to serve as inputs to an audio masked autoencoder (MAE). The current project direction is:

1. preprocess FMA-small audio into log-normalized mel-spectrogram tensors,
2. learn a fixed Audio Barlow Twins / SupCon contrastive anchor manifold (reference space),
3. treat each track's mel spectrogram as a vector `x ∈ ℝ^d` (`d = 64 × 128 = 8192`),
4. apply five CS sensing methods to produce sparse codes `α` such that `x ≈ Dα`, and
5. evaluate reconstruction success and downstream utility of the recovered sparse codes.

The central scientific question is whether structured sparsity in the mel time-frequency domain is sufficient for faithful reconstruction at low measurement ratios, and which basis (identity, DCT, learned patch dictionary) best captures the relevant signal.

## Repository Map

- `preprocess/`: FMA metadata/audio download helpers and log-mel tensor generation.
- `representation/`: Audio Barlow Twins training, embedding extraction, manifold selection, and UMAP visualization.
- `compression/`: CS mel sensing pipeline (`cs_mel.py`) and reconstruction evaluation (`cs_mel_eval.py`).
- `persistence/`: persistence diagrams, residual diagrams, and within/between-genre topology variation.
- `evaluation/`: linear probes over compressed representations.
- `configs/`: YAML configs for each stage. Runtime artifacts under `data/`, `images/`, and checkpoints are ignored.

## Data And Preprocessing

The expected dataset is FMA-small: 8 top-level genres with 1,000 tracks per genre. The preprocessing path downsamples audio to `22.05 kHz`, computes `64`-bin mel-spectrograms, applies log scaling, then min-max normalizes each track.

Download metadata:

```bash
bash -v preprocess/meta.sh
```

Download and extract FMA-small:

```bash
bash -v preprocess/small.sh
```

Generate mel tensors:

```bash
python -m preprocess.mel -d preprocess/data/fma_small
```

Add `--sample-images` to write a few preview spectrograms to `preprocess/images/`.

## Anchor Manifold

The anchor manifold is produced by `representation.manifold`. It trains the configured Audio Barlow Twins grid, optionally adds SupCon regularization, extracts track-level embeddings, evaluates candidate manifolds on validation probes, and writes the selected manifold as:

```text
representation/data/anchor_fma_small_mel_training.parquet
representation/data/anchor_fma_small_mel_validation.parquet
representation/data/anchor_fma_small_mel_test.parquet
```

Run:

```bash
python -m representation.manifold -d preprocess/data/fma_small_mel
```

Plot the selected anchor manifold:

```bash
python -m representation.umap -a representation/data
```

## Compressive Sensing on Mel Spectrograms

Each mel tensor `[64, T]` is truncated/padded to `[64, 128]` and flattened to `x ∈ ℝ^{8192}`. Five sensing methods are evaluated at measurement ratios `m/d ∈ {1%, 2%, 4%, 8%, 12%, 16%, 24%, 50%}`:

| Method | Sensing / Basis `D` | Recovery |
|---|---|---|
| `random_patch` | Randomly keep `m` indices | Identity basis; null baseline |
| `energy_patch_topk` | Keep top-`m` indices by `|x_i|` | Identity basis; simple structured sparsity |
| `perframe_mel_topk` | Keep top-`m` time-frequency bins by column energy | Identity basis; music/time-frequency sparsity |
| `dct_topk` | Keep top-`m` DCT coefficients | DCT basis; classical transform compression |
| `patch_dictionary` | OMP recovery in a learned patch dictionary | MiniBatchDictionaryLearning basis; learned sparse basis |

The recovered sparse code `α` (nonzero values + support indices) is what the downstream audio MAE uses. Only nonzero entries are stored for efficiency.

Run the full sensing sweep (writes one parquet per method):

```bash
python -m compression.cs_mel -c configs/cs_mel.yaml
```

Outputs:

```text
compression/data/cs_mel_random_patch_fma_small_mel.parquet
compression/data/cs_mel_energy_patch_topk_fma_small_mel.parquet
compression/data/cs_mel_perframe_mel_topk_fma_small_mel.parquet
compression/data/cs_mel_dct_topk_fma_small_mel.parquet
compression/data/cs_mel_patch_dictionary_fma_small_mel.parquet
```

Each parquet is keyed internally by `ratio_percent` with columns: `track_id`, `genre_top`, `split`, `method`, `ratio_percent`, `m_dim`, `d_dim`, `alpha_values` (list), `alpha_support` (list), `recon_mse`.

## Reconstruction Success Experiment

Evaluate reconstruction success rate (P(‖x − x̂‖₂ < 1e-6)) for 128 training samples across all methods and ratios:

```bash
python -m compression.cs_mel_eval -c configs/cs_mel.yaml
```

Saves a plot of all 5 methods on one axes to:

```text
compression/images/cs_mel_reconstruction_success.png
```

## Persistence Analysis

Compute per-genre persistence diagrams and residual diagrams for the selected anchor:

```bash
python -m persistence.diagrams --mel-dir preprocess/data/fma_small_mel
```

Measure topology variation within and across genres:

```bash
python -m persistence.variation -a representation/data
```

## Evaluation

Linear probe evaluation over compressed sparse codes:

```bash
python -m evaluation.linear -d compression/data -c configs/linear.yaml
```

## Citation

This dataset was made possible by the work of Defferrard et al. If you use FMA downstream, credit the original dataset authors:

> Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson.  
> **"FMA: A Dataset for Music Analysis"**  
> *18th International Society for Music Information Retrieval Conference (ISMIR), 2017.*  
> [Official FMA GitHub Repository](https://github.com/mdeff/fma)
