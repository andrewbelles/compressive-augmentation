from pathlib import Path

import pandas as pd

# shared driver plumbing: rho grid and per-mechanism parquet artifacts

DEFAULT_RATIOS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def write_records(out_dir: Path, mechanism: str, seed: int, records: list[dict]) -> Path:
    """Write one mechanism-and-seed measurement table to parquet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(records)
    df.insert(0, "seed", seed)
    df.insert(0, "mechanism", mechanism)
    path = out_dir / f"{mechanism}_seed{seed:03d}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_records(out_dir: Path, mechanism: str) -> pd.DataFrame:
    """Concatenate all seed artifacts for a mechanism."""
    paths = sorted(Path(out_dir).glob(f"{mechanism}_seed*.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
