import fcntl
from pathlib import Path

import pandas as pd


def write_frames_to_parquet(
    frames: list[pd.DataFrame],
    out_path: Path,
    dedup_keys: list[str],
) -> None:
    """Atomic append with dedup to a shared parquet file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out_path.with_suffix(".lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
        combined = pd.concat([existing, *frames], ignore_index=True)
        combined = combined.drop_duplicates(subset=dedup_keys, keep="last")
        combined.to_parquet(out_path, index=False)
    print(f"wrote path={out_path} total_rows={len(combined)}", flush=True)
