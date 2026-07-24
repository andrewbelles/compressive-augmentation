import numpy as np
import h5py
import pytest
import torch

from rf.data import read_manifest, select_indices, load_frames, FRAME_LEN
from rf.preprocess.manifests import write_manifests, MOD_CLASSES

N_MODS = 4
N_SNR = 3
PER = 12
N_TOTAL = N_MODS * N_SNR * PER


def _make_hdf5(path):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N_TOTAL, FRAME_LEN, 2)).astype(np.float32)
    Y = np.zeros((N_TOTAL, len(MOD_CLASSES)), dtype=np.float32)
    Z = np.zeros(N_TOTAL, dtype=np.float64)
    i = 0
    for mi in range(N_MODS):
        for snr in range(N_SNR):
            for _ in range(PER):
                Y[i, mi] = 1.0
                Z[i] = float(snr * 2)
                i += 1
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("Y", data=Y)
        f.create_dataset("Z", data=Z)
    return X


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "fake.hdf5"
    X = _make_hdf5(path)
    write_manifests(path, tmp_path)
    return path, X, tmp_path / "manifest_all.csv"


class TestLoadFrames:
    def test_returns_complex_frames(self, dataset):
        path, X, _ = dataset
        frames = load_frames(path, [0, 5, 10])
        assert frames.shape == (3, FRAME_LEN)
        assert frames.is_complex()

    def test_values_match_iq(self, dataset):
        path, X, _ = dataset
        frames = load_frames(path, [7])
        assert np.allclose(frames[0].real.numpy(), X[7, :, 0], atol=1e-5)
        assert np.allclose(frames[0].imag.numpy(), X[7, :, 1], atol=1e-5)

    def test_unsorted_indices_preserved(self, dataset):
        path, X, _ = dataset
        idx = [10, 2, 30]
        frames = load_frames(path, idx)
        for row, i in enumerate(idx):
            assert np.allclose(frames[row].real.numpy(), X[i, :, 0], atol=1e-5)


class TestSelection:
    def test_balanced_per_group(self, dataset):
        _, _, manifest = dataset
        rows = read_manifest(manifest)
        picked = select_indices(rows, per_group=3, seed=0)
        keys = {}
        for r in picked:
            keys[(r["mod"], r["snr"])] = keys.get((r["mod"], r["snr"]), 0) + 1
        assert all(v <= 3 for v in keys.values())

    def test_snr_filter(self, dataset):
        _, _, manifest = dataset
        rows = read_manifest(manifest)
        picked = select_indices(rows, per_group=5, snrs=[0], seed=0)
        assert {r["snr"] for r in picked} == {0}
