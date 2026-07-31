from pathlib import Path

import pandas as pd
import torch

# shared driver plumbing: rho grid, SNR strata, and per-mechanism parquet artifacts

# log-spaced below the sparsity ratio k_eff/n = 0.169 and linear above it, so the grid straddles
# the point where the measurement count stops covering the support and keeps rho_dt interior
DEFAULT_RATIOS = [0.02, 0.03, 0.05, 0.08, 0.12, 0.17, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
EPS = 1e-12


def normalize_power(x: torch.Tensor) -> torch.Tensor:
    """Scale each frame to unit mean power so absolute thresholds mean the same thing."""
    return x / x.abs().pow(2).mean(-1, keepdim=True).clamp_min(EPS).sqrt()


def snr_strata(meta: dict, n_rows: int) -> dict:
    """Group row indices by SNR label; pooling across SNR is the dominant confound."""
    labels = meta.get("snr", [0] * n_rows)
    groups: dict[int, list[int]] = {}
    for i, s in enumerate(labels):
        groups.setdefault(int(s), []).append(i)
    return dict(sorted(groups.items()))


def noise_sigma(snr_db: float) -> float:
    """Noise standard deviation of a unit-power frame at the given channel SNR."""
    return (1.0 / (1.0 + 10.0 ** (snr_db / 10.0))) ** 0.5


def mods_at(meta: dict, rows: list[int]) -> list[str]:
    """Modulation labels for the given row indices."""
    labels = meta.get("mod", [""] * (max(rows) + 1 if rows else 0))
    return [labels[i] for i in rows]


def noise_tag(measurement_snr) -> str:
    """Filename tag for the measurement-SNR axis; tasks sharing a seed must not clobber each other."""
    return "noiseless" if measurement_snr is None else f"{measurement_snr:g}dB"


def write_records(out_dir: Path, mechanism: str, seed: int, records: list[dict],
                  tag: str = "") -> Path:
    """Write one mechanism, seed and noise-level measurement table to parquet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(records)
    df.insert(0, "seed", seed)
    df.insert(0, "mechanism", mechanism)
    suffix = f"_{tag}" if tag else ""
    path = out_dir / f"{mechanism}_seed{seed:03d}{suffix}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_records(out_dir: Path, mechanism: str) -> pd.DataFrame:
    """Concatenate all seed artifacts for a mechanism, rejecting mixed schemas."""
    paths = sorted(Path(out_dir).glob(f"{mechanism}_seed*.parquet"))
    if not paths:
        return pd.DataFrame()
    frames = {p: pd.read_parquet(p) for p in paths}
    # the glob cannot tell a stale untagged artifact from a current tagged one, and pooling two
    # schemas silently averages columns a previous code version defined differently
    schemas = {p: frozenset(df.columns) for p, df in frames.items()}
    newest = max(schemas.values(), key=lambda s: sum(v == s for v in schemas.values()))
    stale = sorted(p.name for p, s in schemas.items() if s != newest)
    if stale:
        missing = sorted(newest - schemas[Path(out_dir) / stale[0]])
        raise ValueError(
            f"{mechanism}: {len(stale)} artifact(s) carry a different schema and would be pooled "
            f"with {len(paths) - len(stale)} current one(s); columns missing: {missing}. "
            f"Delete or re-tag: {stale}")
    return pd.concat(frames.values(), ignore_index=True)
