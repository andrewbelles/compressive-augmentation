"""Waveform Barlow Twins representation learning."""

from pathlib import Path

from representation.audio import (
    WaveSTFTEncoder,
    WaveBarlowModel,
    WaveBarlowDataset,
    WaveABTDataset,
    barlow_twins_loss,
    load_manifest,
    _load_waveform,
    _dct_cs_view,
    _srht_cs_view,
    apply_wave_policy,
)


def run_supcon(
    seed: int = 0,
    exclude_genres: list[str] | None = None,
    data_dir: str | Path = "preprocess/data/fma_small_mel",
    audio_root: str | Path = "preprocess/data",
    checkpoint_dir: str | Path = "representation/checkpoints",
    output_dir: str | Path = "representation/data",
    temp: float = 0.07,
    proj_dim: int = 128,
    **config_overrides,
) -> Path:
    from representation.utils import load_config, resolve_device, set_seed
    from representation.wave_barlow import DEFAULT_CONFIG, DEFAULT_CONFIG_PATH
    from representation.supcon import train_supcon, extract_supcon_embeddings

    config = load_config(DEFAULT_CONFIG_PATH, DEFAULT_CONFIG)
    config.update(config_overrides)
    if exclude_genres is not None:
        config["exclude_genres"] = list(exclude_genres)

    device = resolve_device(str(config["device"]))
    set_seed(seed)

    data_dir       = Path(data_dir).expanduser().resolve()
    audio_root     = Path(audio_root).expanduser().resolve()
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    output_dir     = Path(output_dir).expanduser().resolve()

    ckpt = train_supcon(data_dir, audio_root, checkpoint_dir, config, device,
                        seed=seed, temp=temp, proj_dim=proj_dim)
    return extract_supcon_embeddings(data_dir, audio_root, ckpt, output_dir, config, device)


def run_wave_barlow(
    ratio: int | None = None,
    policy: str | None = None,
    embedding_dim: int = 256,
    uniform: bool = False,
    srht: bool = False,
    supervised: bool = False,
    seed: int | None = None,
    exclude_genres: list[str] | None = None,
    data_dir: str | Path = "preprocess/data/fma_small_mel",
    audio_root: str | Path = "preprocess/data",
    checkpoint_dir: str | Path = "representation/checkpoints",
    output_dir: str | Path = "representation/data",
    **config_overrides,
) -> Path:
    from representation.utils import load_config, resolve_device, set_seed
    from representation.wave_barlow import (
        DEFAULT_CONFIG, DEFAULT_CONFIG_PATH,
        get_source_name, train_one, extract_embeddings,
    )

    if ratio is None and policy is None:
        raise ValueError("provide ratio (cs mode) or policy (traditional mode)")

    mode = "cs" if ratio is not None else "traditional"

    config = load_config(DEFAULT_CONFIG_PATH, DEFAULT_CONFIG)
    config.update(config_overrides)
    if seed is not None:
        config["seed"] = seed
    if exclude_genres is not None:
        config["exclude_genres"] = list(exclude_genres)

    device = resolve_device(str(config["device"]))
    set_seed(int(config["seed"]))

    data_dir       = Path(data_dir).expanduser().resolve()
    audio_root     = Path(audio_root).expanduser().resolve()
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    output_dir     = Path(output_dir).expanduser().resolve()

    ckpt_path = train_one(
        data_dir, audio_root, checkpoint_dir, mode, embedding_dim,
        ratio, policy, config, device, uniform, srht, supervised,
    )
    return extract_embeddings(data_dir, audio_root, ckpt_path, output_dir, config, device)


__all__ = [
    "run_supcon",
    "WaveSTFTEncoder",
    "WaveBarlowModel",
    "WaveBarlowDataset",
    "WaveABTDataset",
    "barlow_twins_loss",
    "load_manifest",
    "_load_waveform",
    "_dct_cs_view",
    "_srht_cs_view",
    "apply_wave_policy",
    "run_wave_barlow",
]
