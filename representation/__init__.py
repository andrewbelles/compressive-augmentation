"""Waveform Barlow Twins representation learning."""

from pathlib import Path

from representation.audio import (
    WaveSTFTEncoder,
    WaveBarlowModel,
    WaveBarlowDataset,
    WaveABTDataset,
    HybridWaveDataset,
    ChainedHybridWaveDataset,
    barlow_twins_loss,
    load_manifest,
    _load_waveform,
    _dct_cs_view,
)


def run_wave_barlow(
    ratio: int | None = None,
    policy: str | None = None,
    method: str | None = None,
    embedding_dim: int = 256,
    uniform: bool = False,
    exclude_genres: list[str] | None = None,
    data_dir: str | Path = "preprocess/data/fma_small_mel",
    audio_root: str | Path = "preprocess/data",
    checkpoint_dir: str | Path = "representation/checkpoints",
    output_dir: str | Path = "representation/data",
    force_retrain: bool = True,
    **config_overrides,
) -> Path:
    from compression.train_utils import load_config, resolve_device, set_seed
    from representation.wave_barlow import (
        DEFAULT_CONFIG, DEFAULT_CONFIG_PATH,
        get_source_name, train_one, extract_embeddings,
    )

    if method is not None:
        mode = method
    elif ratio is not None and policy is None:
        mode = "cs"
    elif policy is not None and ratio is None:
        mode = "traditional"
    elif ratio is not None and policy is not None:
        mode = "hybrid"
    else:
        raise ValueError("provide ratio, policy, ratio+policy, or method")

    config = load_config(DEFAULT_CONFIG_PATH, DEFAULT_CONFIG)
    config.update(config_overrides)
    config["force_retrain"] = force_retrain

    device = resolve_device(str(config["device"]))
    set_seed(int(config["seed"]))

    data_dir = Path(data_dir).expanduser().resolve()
    audio_root = Path(audio_root).expanduser().resolve()
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    if exclude_genres is not None:
        config["exclude_genres"] = list(exclude_genres)
    exclude_genres = list(config.get("exclude_genres", []))
    dataset_name = str(config.get("dataset", "fma_small"))
    uniform = uniform if mode == "cs" else False

    if mode not in ("cs", "traditional", "hybrid", "hybrid_chain"):
        raise ValueError(f"unknown mode: {mode}")

    if method is not None and ratio is None and policy is None:
        if mode == "cs":
            grid = [(embedding_dim, int(r), None) for r in config.get("ratios", [20])]
        elif mode == "traditional":
            grid = [(embedding_dim, None, str(p)) for p in config.get("policies", ["w3"])]
        else:
            grid = [(embedding_dim, int(r), str(p))
                    for r in config.get("ratios", [20])
                    for p in config.get("policies", ["w3"])]
    else:
        grid = [(embedding_dim, ratio, policy)]

    parquet_path = None
    for emb_dim, r, p in grid:
        source = get_source_name(mode, emb_dim, r, p, exclude_genres, uniform)
        ckpt_path = checkpoint_dir / f"{source}_{dataset_name}.pt"
        ckpt_path = train_one(
            data_dir, audio_root, checkpoint_dir, mode, emb_dim, r, p, config, device, uniform
        )
        parquet_path = extract_embeddings(data_dir, audio_root, ckpt_path, output_dir, config, device)

    return parquet_path


__all__ = [
    "WaveSTFTEncoder",
    "WaveBarlowModel",
    "WaveBarlowDataset",
    "WaveABTDataset",
    "HybridWaveDataset",
    "ChainedHybridWaveDataset",
    "barlow_twins_loss",
    "load_manifest",
    "_load_waveform",
    "_dct_cs_view",
    "run_wave_barlow",
]
