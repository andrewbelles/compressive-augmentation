#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def find_tracks_csv(data_dir: Path) -> Path:
    """Locate FMA tracks.csv relative to the audio directory."""
    for candidate in [data_dir.parent / "tracks.csv",
                      data_dir.parent / "fma_metadata" / "tracks.csv"]:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("could not find tracks.csv")


def load_track_metadata(tracks_csv: Path) -> dict[int, dict]:
    """Read FMA track metadata from the two-row-header CSV."""
    result = {}
    with tracks_csv.open("r", encoding="utf-8", newline="") as fh:
        reader  = csv.reader(fh)
        top     = next(reader)
        bot     = next(reader)
        idx     = {(t.strip(), b.strip()): i for i, (t, b) in enumerate(zip(top, bot))}
        split_i  = idx[("set",   "split")]
        subset_i = idx[("set",   "subset")]
        genre_i  = idx[("track", "genre_top")]
        dur_i    = idx[("track", "duration")]
        title_i  = idx[("track", "title")]
        for row in reader:
            if not row:
                continue
            try:
                tid = int(row[0])
            except ValueError:
                continue
            result[tid] = {
                "split":     row[split_i].strip(),
                "subset":    row[subset_i].strip(),
                "genre_top": row[genre_i].strip(),
                "duration":  row[dur_i].strip(),
                "title":     row[title_i].strip(),
            }
    return result


def write_manifests(data_dir: Path, output_dir: Path) -> dict[str, Path]:
    """Write split manifests linking track ids, audio paths, and mel paths."""
    tracks_csv = find_tracks_csv(data_dir)
    metadata   = load_track_metadata(tracks_csv)
    fields     = ["track_id", "split", "subset", "genre_top", "duration", "title", "audio_path", "mel_path"]
    rows: dict[str, list] = {"all": [], "training": [], "validation": [], "test": []}

    # scan .npy files to confirm decode_audio has run; use .pt paths for mel_path column
    for npy_path in sorted(output_dir.rglob("*.npy") if output_dir != data_dir else data_dir.rglob("*.npy")):
        try:
            tid = int(npy_path.stem)
        except ValueError:
            continue
        meta = metadata.get(tid)
        if meta is None:
            continue
        audio_rel = (data_dir / npy_path.relative_to(data_dir)).with_suffix(".mp3").relative_to(data_dir.parent)
        # mel_path points to where mel.py would write; may not exist yet
        mel_dir   = data_dir.parent / f"{data_dir.name}_mel"
        mel_rel   = (mel_dir / npy_path.relative_to(data_dir)).with_suffix(".pt").relative_to(data_dir.parent)
        split     = meta["split"] or "unknown"
        row = {
            "track_id":  str(tid),
            "split":     split,
            "subset":    meta["subset"],
            "genre_top": meta["genre_top"],
            "duration":  meta["duration"],
            "title":     meta["title"],
            "audio_path": audio_rel.as_posix(),
            "mel_path":   mel_rel.as_posix(),
        }
        rows["all"].append(row)
        if split in rows:
            rows[split].append(row)

    manifest_dir = data_dir.parent / f"{data_dir.name}_mel"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = {}
    for name, row_list in rows.items():
        path = manifest_dir / f"manifest_{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(row_list)
        manifest_paths[name] = path
        print(f"wrote manifest_{name}.csv  n={len(row_list)}", flush=True)
    return manifest_paths


def main() -> int:
    """CLI entry point for writing FMA split manifests from decoded .npy files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", type=Path, required=True)
    args     = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(f"directory not found: {data_dir}")
    write_manifests(data_dir, data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
