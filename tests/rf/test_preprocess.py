import csv
from pathlib import Path

import h5py
import numpy as np
import pytest

from rf.preprocess.manifests import (
    MOD_CLASSES,
    FIELDS,
    load_hdf5_index,
    write_manifests,
)

N_MODS  = len(MOD_CLASSES)
N_SNRS  = 5
N_FRAMES_PER_GROUP = 20
N_TOTAL = N_MODS * N_SNRS * N_FRAMES_PER_GROUP
SNR_VALUES = list(range(-20, -20 + N_SNRS * 2, 2))


def _make_fake_hdf5(path: Path) -> Path:
    """Write a minimal HDF5 file with the same structure as the real dataset."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((N_TOTAL, 1024, 2)).astype(np.float32)
    Y = np.zeros((N_TOTAL, N_MODS), dtype=np.float32)
    Z = np.zeros(N_TOTAL, dtype=np.float64)
    idx = 0
    for mod_i in range(N_MODS):
        for snr in SNR_VALUES:
            for _ in range(N_FRAMES_PER_GROUP):
                Y[idx, mod_i] = 1.0
                Z[idx] = float(snr)
                idx += 1
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("Y", data=Y)
        f.create_dataset("Z", data=Z)
    return path


@pytest.fixture
def fake_hdf5(tmp_path):
    return _make_fake_hdf5(tmp_path / "fake_rml2018.hdf5")


@pytest.fixture
def manifests(fake_hdf5, tmp_path):
    out_dir = tmp_path / "manifests"
    write_manifests(fake_hdf5, out_dir)
    return out_dir


class TestLoadHDF5Index:
    def test_returns_correct_columns(self, fake_hdf5):
        rows = load_hdf5_index(fake_hdf5)
        assert isinstance(rows, list)
        assert len(rows) > 0
        assert set(rows[0].keys()) >= {"frame_idx", "mod", "snr"}

    def test_frame_count_matches_hdf5(self, fake_hdf5):
        rows = load_hdf5_index(fake_hdf5)
        assert len(rows) == N_TOTAL

    def test_mod_strings_valid(self, fake_hdf5):
        rows = load_hdf5_index(fake_hdf5)
        for row in rows:
            assert row["mod"] in MOD_CLASSES

    def test_snr_values_present(self, fake_hdf5):
        rows = load_hdf5_index(fake_hdf5)
        found_snrs = {row["snr"] for row in rows}
        assert set(SNR_VALUES).issubset(found_snrs)

    def test_frame_idx_contiguous(self, fake_hdf5):
        rows = load_hdf5_index(fake_hdf5)
        indices = sorted(row["frame_idx"] for row in rows)
        assert indices == list(range(N_TOTAL))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_hdf5_index(tmp_path / "nonexistent.hdf5")


class TestWriteManifests:
    def test_all_four_csvs_written(self, manifests):
        for name in ("all", "training", "validation", "test"):
            assert (manifests / f"manifest_{name}.csv").is_file()

    def test_required_fields(self, manifests):
        path = manifests / "manifest_all.csv"
        with path.open() as fh:
            reader = csv.DictReader(fh)
            assert set(reader.fieldnames) == set(FIELDS)
            row = next(reader)
            for field in FIELDS:
                assert field in row

    def test_all_is_union_of_splits(self, manifests):
        def _count(name):
            with (manifests / f"manifest_{name}.csv").open() as fh:
                return sum(1 for _ in csv.DictReader(fh))
        total = _count("all")
        parts = _count("training") + _count("validation") + _count("test")
        assert total == parts == N_TOTAL

    def test_no_frame_in_multiple_splits(self, manifests):
        seen = set()
        for name in ("training", "validation", "test"):
            with (manifests / f"manifest_{name}.csv").open() as fh:
                for row in csv.DictReader(fh):
                    idx = int(row["frame_idx"])
                    assert idx not in seen, f"frame {idx} appears in multiple splits"
                    seen.add(idx)
        assert len(seen) == N_TOTAL

    def test_stratified_per_mod_snr(self, manifests):
        group_splits: dict[tuple, set] = {}
        for name in ("training", "validation", "test"):
            with (manifests / f"manifest_{name}.csv").open() as fh:
                for row in csv.DictReader(fh):
                    key = (row["mod"], int(row["snr"]))
                    group_splits.setdefault(key, set()).add(name)
        for key, splits in group_splits.items():
            assert len(splits) == 3, (
                f"group {key} only appears in splits {splits}, expected all three"
            )

    def test_split_ratio_approx(self, manifests):
        def _count(name):
            with (manifests / f"manifest_{name}.csv").open() as fh:
                return sum(1 for _ in csv.DictReader(fh))
        n_train = _count("training")
        n_val   = _count("validation")
        n_test  = _count("test")
        total   = n_train + n_val + n_test
        assert abs(n_train / total - 0.7) < 0.05
        assert abs(n_val   / total - 0.2) < 0.05
        assert abs(n_test  / total - 0.1) < 0.05

    def test_idempotent(self, fake_hdf5, tmp_path):
        out = tmp_path / "manifests_idem"
        write_manifests(fake_hdf5, out)
        content_first = {}
        for name in ("all", "training", "validation", "test"):
            content_first[name] = (out / f"manifest_{name}.csv").read_text()
        write_manifests(fake_hdf5, out)
        for name in ("all", "training", "validation", "test"):
            assert (out / f"manifest_{name}.csv").read_text() == content_first[name]
