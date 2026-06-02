#!/usr/bin/env python3
#
# audio.py  Andrew Belles  May 8th, 2026
#
# Waveform Barlow Twins: datasets, encoder, DCT CS view generation, and loss.
#

import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def load_manifest(data_dir: Path, split: str) -> pd.DataFrame:
    manifest_path = data_dir / f"manifest_{split}.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    return pd.read_csv(manifest_path)


def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    n, m = matrix.shape
    if n != m:
        raise ValueError("expected square matrix")
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_twins_loss(left: torch.Tensor, right: torch.Tensor, lambd: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = left.size(0)
    left  = (left  - left.mean(dim=0))  / left.std(dim=0).clamp_min(1e-6)
    right = (right - right.mean(dim=0)) / right.std(dim=0).clamp_min(1e-6)
    correlation = left.T @ right / batch_size
    on_diag  = torch.diagonal(correlation).add_(-1.0).pow_(2).sum()
    off_diag = off_diagonal(correlation).pow_(2).sum()
    return on_diag + float(lambd) * off_diag, on_diag, off_diag


def _load_waveform(audio_path: Path, sr: int, offset_sec: float, duration_sec: float) -> np.ndarray:
    npy_path = audio_path.with_suffix(".npy")
    if npy_path.exists():
        y = np.load(npy_path, mmap_mode="r")
        start = int(offset_sec * sr)
        n     = int(duration_sec * sr)
        segment = np.array(y[start : start + n], dtype=np.float32)
        if len(segment) < n:
            segment = np.pad(segment, (0, n - len(segment)))
        return segment
    cmd = ["ffmpeg", "-y", "-i", str(audio_path), "-ar", str(sr), "-ac", "1",
           "-ss", str(offset_sec), "-t", str(duration_sec), "-f", "f32le", "-"]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or len(result.stdout) == 0:
        raise RuntimeError(f"ffmpeg failed for {audio_path}")
    y = np.frombuffer(result.stdout, dtype=np.float32)
    n = int(sr * duration_sec)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    y = y[:n]
    peak = np.abs(y).max()
    if peak > 1e-8:
        y = y / peak
    return y.astype(np.float32)


_DCT_PROBS_CACHE: dict[int, np.ndarray] = {}


def _get_dct_probs(n: int) -> np.ndarray:
    if n not in _DCT_PROBS_CACHE:
        probs = 1.0 / np.sqrt(np.arange(1, n + 1, dtype=np.float32))
        probs /= probs.sum()
        _DCT_PROBS_CACHE[n] = probs
    return _DCT_PROBS_CACHE[n]


def _srht_cs_view(y: np.ndarray, ratio: float, rng: np.random.Generator, energy_rescale: bool = True) -> torch.Tensor:
    n  = len(y)
    m  = max(1, int(round(n * ratio / 100.0)))
    p2 = 1 << math.ceil(math.log2(max(n, 2)))
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=p2)
    yp = np.zeros(p2, dtype=np.float32)
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
    z = np.zeros(p2, dtype=np.float32)
    z[support] = yp[support] * (math.sqrt(p2 / m) if energy_rescale else 1.0)
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


def _dct_cs_view(y: np.ndarray, ratio: float, rng: np.random.Generator, uniform: bool = False, energy_rescale: bool = True) -> torch.Tensor:
    from scipy.fft import dct, idct
    n      = len(y)
    m      = max(1, int(round(n * ratio / 100.0)))
    coeffs = dct(y, norm="ortho", workers=1)
    idx    = rng.choice(n, m, replace=False) if uniform else rng.choice(n, m, replace=False, p=_get_dct_probs(n))
    z      = np.zeros(n, dtype=np.float32)
    z[idx] = coeffs[idx] * (math.sqrt(n / m) if energy_rescale else 1.0)
    return torch.from_numpy(idct(z, norm="ortho", workers=1).astype(np.float32))


def random_gain(y: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    return (y * float(rng.uniform(1.0 - strength, 1.0 + strength))).astype(np.float32)


def random_time_stretch(y: np.ndarray, scale_range: tuple, rng: np.random.Generator) -> np.ndarray:
    from scipy.signal import resample
    scale      = float(rng.uniform(float(scale_range[0]), float(scale_range[1])))
    n          = len(y)
    n_res      = max(1, int(round(n * scale)))
    y_stretched = resample(y.astype(np.float64), n_res).astype(np.float32)
    if n_res >= n:
        start = int(rng.integers(0, n_res - n + 1))
        return y_stretched[start : start + n]
    pad = n - n_res
    pad_left = int(rng.integers(0, pad + 1))
    return np.pad(y_stretched, (pad_left, pad - pad_left))


def random_waveform_mask(y: np.ndarray, n_masks: int, max_width: int, rng: np.random.Generator) -> np.ndarray:
    y = y.copy()
    n = len(y)
    for _ in range(n_masks):
        width = int(rng.integers(1, max_width + 1))
        start = int(rng.integers(0, max(1, n - width)))
        y[start : start + width] = 0.0
    return y


def random_waveform_noise(y: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    return (y + rng.standard_normal(len(y)).astype(np.float32) * std).astype(np.float32)


def apply_wave_policy(y: np.ndarray, policy: str, config: dict, rng: np.random.Generator) -> np.ndarray:
    y = random_time_stretch(y, tuple(config["wave_stretch_scale"]), rng)
    if policy in {"w2", "w3", "w4"}:
        y = random_gain(y, float(config["wave_gain_strength"]), rng)
    if policy == "w3":
        y = random_waveform_mask(y, int(config["wave_n_masks"]), int(config["wave_mask_width"]), rng)
        y = random_waveform_noise(y, float(config["wave_noise_std"]), rng)
    if policy == "w4":
        y = random_waveform_mask(y, int(config["wave_n_masks"]), int(config["wave_mask_width"]), rng)
    return y


class WaveBarlowDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        ratio: int,
        segment_seconds: float,
        sample_rate: int,
        audio_root: Path,
        seed: int = 0,
        exclude_genres: list[str] | None = None,
        uniform: bool = False,
        srht: bool = False,
        supervised: bool = False,
    ) -> None:
        manifest = load_manifest(data_dir.resolve(), split)
        if exclude_genres:
            manifest = manifest[~manifest["genre_top"].isin(exclude_genres)]
        self.rows            = manifest.to_dict("records")
        self.ratio           = int(ratio)
        self.segment_seconds = float(segment_seconds)
        self.sample_rate     = int(sample_rate)
        self.audio_root      = audio_root.resolve()
        self.seed            = seed
        self.uniform         = uniform
        self.srht            = srht
        self.supervised      = supervised
        self.is_train        = (split == "training")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row        = self.rows[index]
        audio_path = self.audio_root / Path(row["audio_path"])
        epoch_seed = int(torch.initial_seed()) % (2 ** 31) if self.is_train else 0
        rng        = np.random.default_rng([self.seed, index, epoch_seed])

        if self.supervised:
            off1 = float(rng.uniform(10.0, 25.0))
            off2 = float(rng.uniform(10.0, 25.0))
            y1 = _load_waveform(audio_path, self.sample_rate, off1, self.segment_seconds)
            y2 = _load_waveform(audio_path, self.sample_rate, off2, self.segment_seconds)
            return torch.from_numpy(y1).unsqueeze(0), torch.from_numpy(y2).unsqueeze(0)

        offset = float(rng.uniform(10.0, 25.0))
        y      = _load_waveform(audio_path, self.sample_rate, offset, self.segment_seconds)
        rng1   = np.random.default_rng([self.seed, index, epoch_seed, 1])
        rng2   = np.random.default_rng([self.seed, index, epoch_seed, 2])
        if self.srht:
            v1 = _srht_cs_view(y, self.ratio, rng1).unsqueeze(0)
            v2 = _srht_cs_view(y, self.ratio, rng2).unsqueeze(0)
        else:
            v1 = _dct_cs_view(y, self.ratio, rng1, self.uniform).unsqueeze(0)
            v2 = _dct_cs_view(y, self.ratio, rng2, self.uniform).unsqueeze(0)
        return v1, v2


class WaveABTDataset(Dataset):
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

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row        = self.rows[index]
        audio_path = self.audio_root / Path(row["audio_path"])
        epoch_seed = int(torch.initial_seed()) % (2 ** 31) if self.is_train else 0
        rng        = np.random.default_rng([self.seed, index, epoch_seed])
        offset     = float(rng.uniform(10.0, 25.0))
        y          = _load_waveform(audio_path, self.sample_rate, offset, self.segment_seconds)
        rng1       = np.random.default_rng([self.seed, index, epoch_seed, 1])
        rng2       = np.random.default_rng([self.seed, index, epoch_seed, 2])
        v1 = torch.from_numpy(apply_wave_policy(y, self.policy, self.augment_config, rng1)).unsqueeze(0)
        v2 = torch.from_numpy(apply_wave_policy(y, self.policy, self.augment_config, rng2)).unsqueeze(0)
        return v1, v2


class WaveSTFTEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        base_channels: int = 16,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_blocks: int = 3,
        n_mels: int = 128,
        sample_rate: int = 22050,
    ) -> None:
        super().__init__()
        self.n_fft      = int(n_fft)
        self.hop_length = int(hop_length)
        self.register_buffer("window", torch.hann_window(n_fft))
        import torchaudio.functional as AF
        fb = AF.melscale_fbanks(
            n_freqs=n_fft // 2 + 1, f_min=80.0, f_max=float(sample_rate) / 2.0,
            n_mels=int(n_mels), sample_rate=int(sample_rate), norm="slaney", mel_scale="htk",
        )
        self.register_buffer("mel_fb", fb)
        channels = [base_channels * (2 ** i) for i in range(int(n_blocks))]
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2),
            ])
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        feat_dim = channels[-1] * 2
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2, bias=False),
            nn.BatchNorm1d(feat_dim * 2), nn.ReLU(inplace=True),
            nn.Linear(feat_dim * 2, embedding_dim),
        )

    def _to_mel(self, x: torch.Tensor) -> torch.Tensor:
        y    = x.squeeze(1)
        spec = torch.stft(y, n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.n_fft, window=self.window, return_complex=True)
        mel  = torch.einsum("bft,fm->bmt", spec.abs(), self.mel_fb)
        mel  = torch.log1p(mel).unsqueeze(1)
        mean = mel.mean(dim=(2, 3), keepdim=True)
        std  = mel.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (mel - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat   = self.features(self._to_mel(x))
        pooled = torch.cat([feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))], dim=1)
        return self.head(pooled)


class WaveBarlowModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        base_channels: int,
        projection_hidden_dim: int,
        projection_dim: int,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_blocks: int = 3,
        n_mels: int = 128,
        sample_rate: int = 22050,
    ) -> None:
        super().__init__()
        self.encoder   = WaveSTFTEncoder(embedding_dim, base_channels, n_fft, hop_length, n_blocks, n_mels, sample_rate)
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, projection_hidden_dim, bias=False),
            nn.BatchNorm1d(projection_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(projection_hidden_dim, projection_dim, bias=False),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        return h1, h2, self.projector(h1), self.projector(h2)
