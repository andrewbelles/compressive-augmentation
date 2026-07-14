import h5py
import numpy as np
import pytest
import torch

from rf.frames import load_complex_frames, load_manifest, select_frames
from rf.preprocess.manifests import MOD_CLASSES, write_manifests

N_MODS = len(MOD_CLASSES)
N_SNRS = 3
N_FRAMES_PER_GROUP = 20
N_TOTAL = N_MODS * N_SNRS * N_FRAMES_PER_GROUP
SNR_VALUES = [-10, 0, 10]


def _make_fake_hdf5(path):
    """Write a minimal HDF5 file with the same structure as the real dataset.

    Frames are sparse in frequency (16 random DFT tones + small noise) so that
    CS reconstruction behaves like it does on real RF frames, unlike white noise
    which is incompressible in every basis.
    """
    rng = np.random.default_rng(42)
    spectra = np.zeros((N_TOTAL, 1024), dtype=np.complex64)
    for i in range(N_TOTAL):
        bins = rng.choice(1024, size=16, replace=False)
        spectra[i, bins] = rng.standard_normal(16) + 1j * rng.standard_normal(16)
    signals = np.fft.ifft(spectra, axis=1) * np.sqrt(1024)
    signals += 0.001 * (rng.standard_normal(signals.shape)
                        + 1j * rng.standard_normal(signals.shape))
    X = np.stack([signals.real, signals.imag], axis=-1).astype(np.float32)
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


@pytest.fixture(scope="module")
def fake_hdf5(tmp_path_factory):
    return _make_fake_hdf5(tmp_path_factory.mktemp("frames") / "fake_rml2018.hdf5")


@pytest.fixture(scope="module")
def manifest_dir(fake_hdf5, tmp_path_factory):
    out = tmp_path_factory.mktemp("manifests")
    write_manifests(fake_hdf5, out)
    return out


class TestLoadManifest:
    def test_columns_and_dtypes(self, manifest_dir):
        df = load_manifest(manifest_dir, "training")
        assert {"frame_idx", "mod", "snr", "split"} <= set(df.columns)
        assert df["frame_idx"].dtype == np.int64
        assert df["snr"].dtype == np.int64

    def test_missing_split_raises(self, manifest_dir):
        with pytest.raises(FileNotFoundError):
            load_manifest(manifest_dir, "bogus")


class TestSelectFrames:
    def test_per_cell_cap(self, manifest_dir):
        df = load_manifest(manifest_dir, "training")
        sel = select_frames(df, per_cell=5, seed=0)
        counts = sel.groupby(["mod", "snr"]).size()
        assert (counts == 5).all()
        assert len(counts) == N_MODS * N_SNRS

    def test_deterministic(self, manifest_dir):
        df = load_manifest(manifest_dir, "training")
        a = select_frames(df, per_cell=5, seed=3)
        b = select_frames(df, per_cell=5, seed=3)
        assert a["frame_idx"].tolist() == b["frame_idx"].tolist()

    def test_seed_changes_selection(self, manifest_dir):
        df = load_manifest(manifest_dir, "training")
        a = select_frames(df, per_cell=5, seed=3)
        b = select_frames(df, per_cell=5, seed=4)
        assert a["frame_idx"].tolist() != b["frame_idx"].tolist()

    def test_filters(self, manifest_dir):
        df = load_manifest(manifest_dir, "training")
        sel = select_frames(df, per_cell=5, seed=0, snr_min=0, mods=["BPSK", "FM"])
        assert set(sel["mod"]) == {"BPSK", "FM"}
        assert (sel["snr"] >= 0).all()
        sel = select_frames(df, per_cell=5, seed=0, snr_values=[10])
        assert set(sel["snr"]) == {10}

    def test_no_match_raises(self, manifest_dir):
        df = load_manifest(manifest_dir, "training")
        with pytest.raises(ValueError):
            select_frames(df, per_cell=5, seed=0, snr_min=99)


class TestLoadComplexFrames:
    def test_exact_iq_values(self, fake_hdf5, device):
        idx = np.array([0, 5, 17])
        x = load_complex_frames(fake_hdf5, idx, device, normalize=False)
        with h5py.File(fake_hdf5, "r") as f:
            raw = f["X"][np.sort(idx)]
        expected = torch.from_numpy(raw[..., 0] + 1j * raw[..., 1]).to(torch.complex64).to(device)
        assert x.shape == (3, 1024)
        assert x.dtype == torch.complex64
        assert torch.allclose(x, expected)

    def test_unsorted_order_preserved(self, fake_hdf5, device):
        idx = np.array([40, 3, 99, 7])
        x = load_complex_frames(fake_hdf5, idx, device, normalize=False)
        for row, i in enumerate(idx):
            single = load_complex_frames(fake_hdf5, np.array([i]), device, normalize=False)
            assert torch.allclose(x[row], single[0])

    def test_unit_normalization(self, fake_hdf5, device):
        x = load_complex_frames(fake_hdf5, np.arange(8), device, normalize=True)
        assert torch.allclose(x.norm(dim=1), torch.ones(8, device=device), atol=1e-5)
