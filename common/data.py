import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.fft import dct, idct
from torch.utils.data import Dataset

from common.ops import apply_wave_policy, _get_dct_probs


def load_manifest(data_dir: Path, split: str) -> pd.DataFrame:
    """Load the CSV manifest for one dataset split."""
    manifest_path = data_dir / f"manifest_{split}.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    return pd.read_csv(manifest_path)


def load_waveform(
    audio_path: Path,
    sr: int,
    offset_sec: float,
    duration_sec: float
) -> np.ndarray:
    """Load a normalized mono waveform segment from cached numpy data or ffmpeg."""
    npy_path = audio_path.with_suffix(".npy")
    if npy_path.exists():
        y       = np.load(npy_path, mmap_mode="r")
        start   = int(offset_sec * sr)
        n       = int(duration_sec * sr)
        segment = np.array(y[start : start + n], dtype=np.float32)
        if len(segment) < n:
            segment = np.pad(segment, (0, n - len(segment)))
        return segment
    cmd    = ["ffmpeg", "-y", "-i", str(audio_path), "-ar", str(sr), "-ac", "1",
              "-ss", str(offset_sec), "-t", str(duration_sec), "-f", "f32le", "-"]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or len(result.stdout) == 0:
        raise RuntimeError(f"ffmpeg failed for {audio_path}")
    y = np.frombuffer(result.stdout, dtype=np.float32)
    n = int(sr * duration_sec)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    y    = y[:n]
    peak = np.abs(y).max()
    if peak > 1e-8:
        y = y / peak
    return y.astype(np.float32)


def srht_cs_view(y: np.ndarray, ratio: float, rng: np.random.Generator) -> torch.Tensor:
    """Apply SRHT compressive sensing and return the reconstructed view."""
    n  = len(y)
    m  = max(1, int(round(n * ratio / 100.0)))
    p2 = 1 << math.ceil(math.log2(max(n, 2)))
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=p2)
    yp    = np.zeros(p2, dtype=np.float32)
    yp[:n] = y * signs[:n]
    h = 1
    while h < p2:
        yp = yp.reshape(-1, h * 2)
        u, v = yp[:, :h].copy(), yp[:, h:].copy()
        yp[:, :h] = u + v
        yp[:, h:] = u - v
        yp = yp.ravel()
        h *= 2
    yp /= math.sqrt(p2)
    support = np.sort(rng.choice(p2, m, replace=False))
    z       = np.zeros(p2, dtype=np.float32)
    z[support] = yp[support] * math.sqrt(p2 / m)
    h = 1
    while h < p2:
        z = z.reshape(-1, h * 2)
        u, v = z[:, :h].copy(), z[:, h:].copy()
        z[:, :h] = u + v
        z[:, h:] = u - v
        z = z.ravel()
        h *= 2
    z /= math.sqrt(p2)
    return torch.from_numpy((z[:n] * signs[:n]).astype(np.float32))


def dct_cs_view(
    y: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
    uniform: bool = False
) -> torch.Tensor:
    """Apply DCT-masked compressive sensing and return the reconstructed view."""
    n      = len(y)
    m      = max(1, int(round(n * ratio / 100.0)))
    coeffs = dct(y, norm="ortho", workers=1)
    idx    = (rng.choice(n, m, replace=False)
              if uniform else rng.choice(n, m, replace=False, p=_get_dct_probs(n)))
    z      = np.zeros(n, dtype=np.float32)
    z[idx] = coeffs[idx] * math.sqrt(n / m)
    return torch.from_numpy(idct(z, norm="ortho", workers=1).astype(np.float32))


class BaseBarlowDataset(Dataset):
    """Abstract base for Barlow Twins datasets.

    Subclasses implement load_sample and make_views. Both views returned by
    make_views must be independent draws from the same augmentation family and
    strength; the Barlow cross-correlation target is only valid under this
    exchangeability constraint.
    """

    _raw_only: bool = False

    def load_sample(self, index: int) -> np.ndarray:
        """Return the raw sample array for index without augmentation."""
        raise NotImplementedError

    def make_views(
        self,
        index: int,
        rng1: np.random.Generator,
        rng2: np.random.Generator,
    ) -> tuple:
        """Return (view1, view2) tensors from independent draws of the same augmentation."""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple:
        epoch_seed = int(torch.initial_seed()) % (2 ** 31) if getattr(self, "is_train", False) else 0
        if self._raw_only:
            y = self.load_sample(index)
            return (torch.from_numpy(np.asarray(y, dtype=np.float32)),)
        rng1 = np.random.default_rng([getattr(self, "seed", 0), index, epoch_seed, 1])
        rng2 = np.random.default_rng([getattr(self, "seed", 0), index, epoch_seed, 2])
        return self.make_views(index, rng1, rng2)


class WaveBarlowDataset(BaseBarlowDataset):
    """Barlow Twins dataset yielding paired DCT/SRHT compressive views of audio waveforms."""

    def __init__(
        self,
        data_dir: Path,
        split: str,
        ratio: float,
        segment_seconds: float,
        sample_rate: int,
        audio_root: Path,
        seed: int = 0,
        exclude_genres: list[str] | None = None,
        uniform: bool = False,
        srht: bool = False,
        preload: bool = False,
    ) -> None:
        manifest = load_manifest(data_dir.resolve(), split)
        if exclude_genres:
            manifest = manifest[~manifest["genre_top"].isin(exclude_genres)]
        self.rows            = manifest.to_dict("records")
        self.ratio           = float(ratio)
        self.segment_seconds = float(segment_seconds)
        self.sample_rate     = int(sample_rate)
        self.audio_root      = audio_root.resolve()
        self.seed            = seed
        self.uniform         = uniform
        self.srht            = srht
        self.is_train        = (split == "training")
        self._raw_only       = False
        self._wav_cache: list[np.ndarray] | None = None
        if preload:
            self._wav_cache = [
                np.load((self.audio_root / Path(r["audio_path"])).with_suffix(".npy"))
                for r in self.rows
            ]

    def __len__(self) -> int:
        return len(self.rows)

    def load_sample(self, index: int) -> np.ndarray:
        """Return a random crop of the audio track at index."""
        epoch_seed = int(torch.initial_seed()) % (2 ** 31) if self.is_train else 0
        rng    = np.random.default_rng([self.seed, index, epoch_seed])
        offset = float(rng.uniform(10.0, 25.0))
        n      = int(self.segment_seconds * self.sample_rate)
        if self._wav_cache is not None:
            y_full = self._wav_cache[index]
            start  = int(offset * self.sample_rate)
            seg    = y_full[start : start + n].astype(np.float32)
            if len(seg) < n:
                seg = np.pad(seg, (0, n - len(seg)))
            return seg
        return load_waveform(
            self.audio_root / Path(self.rows[index]["audio_path"]),
            self.sample_rate, offset, self.segment_seconds,
        )

    def make_views(
        self,
        index: int,
        rng1: np.random.Generator,
        rng2: np.random.Generator,
    ) -> tuple:
        y = self.load_sample(index)
        if self.srht:
            v1 = srht_cs_view(y, self.ratio, rng1).unsqueeze(0)
            v2 = srht_cs_view(y, self.ratio, rng2).unsqueeze(0)
        else:
            v1 = dct_cs_view(y, self.ratio, rng1, self.uniform).unsqueeze(0)
            v2 = dct_cs_view(y, self.ratio, rng2, self.uniform).unsqueeze(0)
        return v1, v2


class WaveABTDataset(BaseBarlowDataset):
    """Barlow Twins dataset yielding paired traditional waveform augmentation views."""

    def __init__(
        self,
        data_dir: Path,
        split: str,
        policy: str,
        segment_seconds: float,
        sample_rate: int,
        audio_root: Path,
        augment_config: dict,
        seed: int = 0,
        exclude_genres: list[str] | None = None,
        preload: bool = False,
    ) -> None:
        manifest = load_manifest(data_dir.resolve(), split)
        if exclude_genres:
            manifest = manifest[~manifest["genre_top"].isin(exclude_genres)]
        self.rows            = manifest.to_dict("records")
        self.policy          = str(policy)
        self.segment_seconds = float(segment_seconds)
        self.sample_rate     = int(sample_rate)
        self.audio_root      = audio_root.resolve()
        self.augment_config  = augment_config
        self.seed            = seed
        self.is_train        = (split == "training")
        self._raw_only       = False
        self._wav_cache: list[np.ndarray] | None = None
        if preload:
            self._wav_cache = [
                np.load((self.audio_root / Path(r["audio_path"])).with_suffix(".npy"))
                for r in self.rows
            ]

    def __len__(self) -> int:
        return len(self.rows)

    def load_sample(self, index: int) -> np.ndarray:
        """Return a random crop of the audio track at index."""
        epoch_seed = int(torch.initial_seed()) % (2 ** 31) if self.is_train else 0
        rng    = np.random.default_rng([self.seed, index, epoch_seed])
        offset = float(rng.uniform(10.0, 25.0))
        n      = int(self.segment_seconds * self.sample_rate)
        if self._wav_cache is not None:
            y_full = self._wav_cache[index]
            start  = int(offset * self.sample_rate)
            seg    = y_full[start : start + n].astype(np.float32)
            if len(seg) < n:
                seg = np.pad(seg, (0, n - len(seg)))
            return seg
        return load_waveform(
            self.audio_root / Path(self.rows[index]["audio_path"]),
            self.sample_rate, offset, self.segment_seconds,
        )

    def make_views(
        self,
        index: int,
        rng1: np.random.Generator,
        rng2: np.random.Generator,
    ) -> tuple:
        y  = self.load_sample(index)
        v1 = torch.from_numpy(
            apply_wave_policy(y, self.policy, self.augment_config, rng1)
        ).unsqueeze(0)
        v2 = torch.from_numpy(
            apply_wave_policy(y, self.policy, self.augment_config, rng2)
        ).unsqueeze(0)
        return v1, v2


class SupConDataset(BaseBarlowDataset):
    """Dataset yielding paired waveform augmentations and genre labels for supervised contrastive training."""

    def __init__(
        self,
        data_dir: Path,
        split: str,
        segment_seconds: float,
        sample_rate: int,
        audio_root: Path,
        augment_config: dict,
        seed: int = 0,
        exclude_genres: list[str] | None = None,
        preload: bool = False,
    ) -> None:
        manifest = load_manifest(data_dir.resolve(), split)
        if exclude_genres:
            manifest = manifest[~manifest["genre_top"].isin(exclude_genres)]
        manifest = manifest.dropna(subset=["genre_top"])
        self.rows            = manifest.to_dict("records")
        self.segment_seconds = float(segment_seconds)
        self.sample_rate     = int(sample_rate)
        self.audio_root      = audio_root.resolve()
        self.augment_config  = augment_config
        self.seed            = seed
        self.is_train        = (split == "training")
        genres               = sorted({r["genre_top"] for r in self.rows})
        self.genre_to_idx    = {g: i for i, g in enumerate(genres)}
        self._raw_only       = False
        self._wav_cache: list[np.ndarray] | None = None
        if preload:
            self._wav_cache = [
                np.load((self.audio_root / Path(r["audio_path"])).with_suffix(".npy"))
                for r in self.rows
            ]

    def __len__(self) -> int:
        return len(self.rows)

    def load_sample(self, index: int) -> np.ndarray:
        """Return a random crop of the audio track at index."""
        epoch_seed = int(torch.initial_seed()) % (2 ** 31) if self.is_train else 0
        rng    = np.random.default_rng([self.seed, index, epoch_seed])
        offset = float(rng.uniform(10.0, 25.0))
        n      = int(self.segment_seconds * self.sample_rate)
        if self._wav_cache is not None:
            y_full = self._wav_cache[index]
            start  = int(offset * self.sample_rate)
            seg    = y_full[start : start + n].astype(np.float32)
            if len(seg) < n:
                seg = np.pad(seg, (0, n - len(seg)))
            return seg
        return load_waveform(
            self.audio_root / Path(self.rows[index]["audio_path"]),
            self.sample_rate, offset, self.segment_seconds,
        )

    def make_views(
        self,
        index: int,
        rng1: np.random.Generator,
        rng2: np.random.Generator,
    ) -> tuple:
        # SupConDataset returns a triple; make_views is unused but satisfies the base contract
        y  = self.load_sample(index)
        v1 = torch.from_numpy(
            apply_wave_policy(y, "w3", self.augment_config, rng1)
        ).unsqueeze(0)
        v2 = torch.from_numpy(
            apply_wave_policy(y, "w3", self.augment_config, rng2)
        ).unsqueeze(0)
        return v1, v2

    def __getitem__(self, index: int) -> tuple:
        row        = self.rows[index]
        epoch_seed = int(torch.initial_seed()) % (2 ** 31) if self.is_train else 0
        lbl        = self.genre_to_idx[row["genre_top"]]
        if self._raw_only:
            return torch.from_numpy(self.load_sample(index)), lbl
        rng1 = np.random.default_rng([self.seed, index, epoch_seed, 1])
        rng2 = np.random.default_rng([self.seed, index, epoch_seed, 2])
        v1, v2 = self.make_views(index, rng1, rng2)
        return v1, v2, lbl
