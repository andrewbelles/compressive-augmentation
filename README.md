# Compressive Sensing on Mel-Spectrogram Representations for FMA Music

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

## Citation

This dataset was made possible by the work of Defferrard et al. If you use FMA downstream, credit the original dataset authors:

> Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson.  
> **"FMA: A Dataset for Music Analysis"**  
> *18th International Society for Music Information Retrieval Conference (ISMIR), 2017.*  
> [Official FMA GitHub Repository](https://github.com/mdeff/fma)
