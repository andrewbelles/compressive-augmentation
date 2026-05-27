#!/usr/bin/env python3
#
# audio.py  Andrew Belles  May 8th, 2026
#
# Audio Barlow Twins + CS-VICReg models and mel crop/sensing utilities.
#

import math
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


AUGMENTATION_POLICIES = ("a0", "a1", "a2", "a3", "a4", "a5")


def resolve_relative_data_path(base_dir: Path, manifest_path: str, use_low_rank: bool = False) -> Path:
    relative_path = Path(str(manifest_path))
    if relative_path.parts and relative_path.parts[0] == base_dir.name:
        relative_path = Path(*relative_path.parts[1:])
    full_path = base_dir / relative_path
    if use_low_rank:
        lr_path = full_path.with_suffix(".lr.pt")
        if not lr_path.exists():
            raise FileNotFoundError(
                f"low-rank file not found: {lr_path}  "
                f"(run: python -m preprocess.rpca -d {base_dir})"
            )
        return lr_path
    return full_path


def load_manifest(data_dir: Path, split: str) -> pd.DataFrame:
    manifest_path = data_dir / f"manifest_{split}.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    return pd.read_csv(manifest_path)


def crop_or_pad(mel: torch.Tensor, frames: int, random_crop: bool) -> torch.Tensor:
    if mel.size(1) < frames:
        mel = F.pad(mel, (0, frames - mel.size(1)))
    if mel.size(1) == frames:
        return mel
    if random_crop:
        start = random.randint(0, mel.size(1) - frames)
    else:
        start = (mel.size(1) - frames) // 2
    return mel[:, start : start + frames]


def resize_time(mel: torch.Tensor, frames: int) -> torch.Tensor:
    resized = F.interpolate(
        mel.unsqueeze(0).unsqueeze(0),
        size=(mel.size(0), frames),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0)


def random_resize_crop(mel: torch.Tensor, frames: int, scale: tuple[float, float]) -> torch.Tensor:
    low, high = float(scale[0]), float(scale[1])
    crop_frames = max(4, int(round(frames * random.uniform(low, high))))
    crop = crop_or_pad(mel, crop_frames, random_crop=True)
    return resize_time(crop, frames)


def random_linear_fader(mel: torch.Tensor, strength: float) -> torch.Tensor:
    if strength <= 0.0:
        return mel
    start = 1.0 + random.uniform(-strength, strength)
    stop = 1.0 + random.uniform(-strength, strength)
    fade = torch.linspace(start, stop, mel.size(1), dtype=mel.dtype, device=mel.device).unsqueeze(0)
    return mel * fade


def random_additive_noise(mel: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0.0:
        return mel
    return (mel + torch.randn_like(mel) * std).clamp_(0.0, 1.0)


def time_frequency_mask(mel: torch.Tensor, time_width: int, freq_width: int) -> torch.Tensor:
    output = mel.clone()
    if time_width > 0 and output.size(1) > 1:
        width = random.randint(1, min(time_width, output.size(1)))
        start = random.randint(0, output.size(1) - width)
        output[:, start : start + width] = 0.0
    if freq_width > 0 and output.size(0) > 1:
        width = random.randint(1, min(freq_width, output.size(0)))
        start = random.randint(0, output.size(0) - width)
        output[start : start + width, :] = 0.0
    return output


def apply_policy(mel: torch.Tensor, policy: str, config: dict) -> torch.Tensor:
    frames = int(config["crop_frames"])
    if policy == "a0":
        return crop_or_pad(mel, frames, random_crop=False)
    if policy in {"a1", "a2", "a3", "a4", "a5"}:
        output = random_resize_crop(mel, frames, tuple(config["resize_scale"]))
    else:
        raise ValueError(f"unsupported augmentation policy: {policy}")

    if policy in {"a2", "a3", "a4", "a5"}:
        output = random_linear_fader(output, float(config["linear_fader_strength"]))
    if policy in {"a3", "a4", "a5"}:
        output = time_frequency_mask(output, int(config["time_mask_width"]), int(config["freq_mask_width"]))
    if policy in {"a4", "a5"}:
        output = random_additive_noise(output, float(config.get("noise_std", 0.0)))
    if policy == "a5":
        output = time_frequency_mask(output, int(config["time_mask_width"]), int(config["freq_mask_width"]))
    return output


def mixup_batch(left: torch.Tensor, right: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0.0 or left.size(0) < 2:
        return left, right
    beta = torch.distributions.Beta(alpha, alpha)
    lam = beta.sample((left.size(0),)).to(left.device).view(-1, 1, 1, 1)
    permutation = torch.randperm(left.size(0), device=left.device)
    return lam * left + (1.0 - lam) * left[permutation], lam * right + (1.0 - lam) * right[permutation]


class BarlowCropDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, policy: str, augment_config: dict, paired: bool, use_low_rank: bool = False):
        self.data_dir = data_dir.resolve()
        frame = load_manifest(self.data_dir, split)
        self.mel_paths = [
            resolve_relative_data_path(self.data_dir, str(r), use_low_rank) for r in frame["mel_path"]
        ]
        self.policy = str(policy)
        self.augment_config = augment_config
        self.paired = bool(paired)

    def __len__(self) -> int:
        return len(self.mel_paths)

    def __getitem__(self, index: int):
        mel_path = self.mel_paths[index]
        mel = torch.load(mel_path, map_location="cpu", weights_only=True).float()
        if mel.ndim != 2:
            raise ValueError(f"expected 2D mel tensor at {mel_path}, got {tuple(mel.shape)}")

        if self.paired:
            left = apply_policy(mel, self.policy, self.augment_config)
            right = left.clone() if self.policy == "a0" else apply_policy(mel, self.policy, self.augment_config)
            return left.unsqueeze(0), right.unsqueeze(0)

        crop = crop_or_pad(mel, int(self.augment_config["crop_frames"]), random_crop=False)
        return crop.unsqueeze(0), {}


def collate_embedding_batch(batch):
    inputs = torch.stack([item[0] for item in batch], dim=0)
    keys = batch[0][1].keys()
    metadata = {key: [item[1][key] for item in batch] for key in keys}
    return inputs, metadata


class AudioCNNEncoder(nn.Module):
    def __init__(self, embedding_dim: int, base_channels: int, dropout: float):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.head = nn.Linear(channels[-1] * 2, embedding_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        mean_pool = features.mean(dim=(2, 3))
        max_pool = features.amax(dim=(2, 3))
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        return self.head(self.dropout(pooled))


class BarlowTwinsModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        base_channels: int,
        dropout: float,
        projector_hidden_dim: int,
        projector_dim: int,
    ):
        super().__init__()
        self.encoder = AudioCNNEncoder(embedding_dim, base_channels, dropout)
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, projector_hidden_dim, bias=False),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden_dim, projector_dim, bias=False),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(inputs)
        projection = self.projector(embedding)
        return embedding, projection


def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    n, m = matrix.shape
    if n != m:
        raise ValueError("expected square matrix")
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_twins_loss(left: torch.Tensor, right: torch.Tensor, lambd: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = left.size(0)
    left = (left - left.mean(dim=0)) / left.std(dim=0).clamp_min(1e-6)
    right = (right - right.mean(dim=0)) / right.std(dim=0).clamp_min(1e-6)
    correlation = left.T @ right / batch_size
    on_diag = torch.diagonal(correlation).add_(-1.0).pow_(2).sum()
    off_diag = off_diagonal(correlation).pow_(2).sum()
    return on_diag + float(lambd) * off_diag, on_diag, off_diag



class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class CSEncoder(nn.Module):
    def __init__(self, embedding_dim: int, base_channels: int, dropout: float) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                _ResBlock(out_ch),
                nn.AvgPool2d(kernel_size=2),
            ])
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.head = nn.Linear(channels[-1] * 2, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = torch.cat([feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))], dim=1)
        return self.head(self.dropout(pooled))



def _dct_ortho(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[0]
    v = torch.cat([x, x.flip(0)]).double()
    V = torch.fft.rfft(v, n=2 * d)
    k = torch.arange(d, dtype=torch.float64, device=x.device)
    phase = torch.exp(-1j * math.pi * k / (2.0 * d))
    coeffs = (V[:d] * phase).real
    coeffs[0] /= math.sqrt(4.0 * d)
    coeffs[1:] /= math.sqrt(2.0 * d)
    return coeffs.float()


def _idct_ortho(c: torch.Tensor) -> torch.Tensor:
    d = c.shape[0]
    c2 = c.double().clone()
    c2[0] *= math.sqrt(4.0 * d)
    c2[1:] *= math.sqrt(2.0 * d)
    k = torch.arange(d, dtype=torch.float64, device=c.device)
    phase = torch.exp(1j * math.pi * k / (2.0 * d))
    V_half = (c2 * phase).to(torch.complex128)
    V_full = torch.cat([V_half, torch.zeros(1, dtype=torch.complex128, device=c.device), V_half[1:].flip(0).conj()])
    return torch.fft.ifft(V_full).real[:d].float()


def dct_backproject(x_flat: torch.Tensor, m: int) -> torch.Tensor:
    d = x_flat.shape[0]
    coeffs = _dct_ortho(x_flat)
    idx = torch.randperm(d, device=x_flat.device)[:min(m, d)]
    z = torch.zeros(d, dtype=torch.float32, device=x_flat.device)
    z[idx] = coeffs[idx]
    return _idct_ortho(z)


def _dct_ortho_batch(x: torch.Tensor) -> torch.Tensor:
    n, d = x.shape
    v = torch.cat([x, x.flip(1)], dim=1).double()
    V = torch.fft.rfft(v, n=2 * d, dim=1)
    k = torch.arange(d, dtype=torch.float64, device=x.device)
    phase = torch.exp(-1j * math.pi * k / (2.0 * d))
    coeffs = (V[:, :d] * phase).real
    coeffs[:, 0] /= math.sqrt(4.0 * d)
    coeffs[:, 1:] /= math.sqrt(2.0 * d)
    return coeffs.float()


def _idct_ortho_batch(c: torch.Tensor) -> torch.Tensor:
    n, d = c.shape
    c2 = c.double().clone()
    c2[:, 0] *= math.sqrt(4.0 * d)
    c2[:, 1:] *= math.sqrt(2.0 * d)
    k = torch.arange(d, dtype=torch.float64, device=c.device)
    phase = torch.exp(1j * math.pi * k / (2.0 * d))
    V_half = c2 * phase
    V_full = torch.cat([V_half, torch.zeros(n, 1, dtype=torch.complex128, device=c.device), V_half[:, 1:].flip(1).conj()], dim=1)
    return torch.fft.ifft(V_full, dim=1).real[:, :d].float()


def dct_backproject_patch(
    x: torch.Tensor, m_per_patch: int, patch_f: int = 16, patch_t: int = 16
) -> torch.Tensor:
    f, t = x.shape
    nf, nt = f // patch_f, t // patch_t
    d = patch_f * patch_t
    m = max(1, min(d, m_per_patch))

    patches = x[:nf*patch_f, :nt*patch_t].reshape(nf, patch_f, nt, patch_t)
    patches = patches.permute(0, 2, 1, 3).reshape(nf * nt, d)

    coeffs = _dct_ortho_batch(patches)

    noise = torch.rand(nf * nt, d, device=x.device)
    topk_idx = noise.topk(m, dim=1).indices
    mask = torch.zeros(nf * nt, d, device=x.device)
    mask.scatter_(1, topk_idx, 1.0)
    coeffs = coeffs * mask

    recon = _idct_ortho_batch(coeffs)
    recon = recon.reshape(nf, nt, patch_f, patch_t).permute(0, 2, 1, 3).reshape(nf * patch_f, nt * patch_t)

    out = torch.zeros_like(x)
    out[:nf*patch_f, :nt*patch_t] = recon
    return out


def gaussian_backproject(x_flat: torch.Tensor, m: int) -> torch.Tensor:
    d = x_flat.shape[0]
    m = min(m, d)
    Phi = torch.randn(m, d, device=x_flat.device, dtype=x_flat.dtype) / math.sqrt(m)
    y = Phi @ x_flat
    return Phi.t() @ y


def parse_sensing_pair(sensing_pair: str) -> tuple[str, str]:
    known = {"dct", "gaussian", "patch_dct"}
    parts = sensing_pair.split("_", 1)
    left_name = parts[0] if parts[0] in known else sensing_pair
    right_name = parts[1] if len(parts) > 1 and parts[1] in known else left_name
    if sensing_pair.startswith("patch_dct"):
        left_name = right_name = "patch_dct"
    if left_name not in known or right_name not in known:
        raise ValueError(f"unsupported sensing_pair: {sensing_pair!r}")
    return left_name, right_name


def cs_view_pair(
    mel: torch.Tensor,
    left_name: str,
    right_name: str,
    ratio: int,
    patch_f: int = 16,
    patch_t: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    c, f, t = mel.shape
    x_flat = mel.reshape(-1)
    d = x_flat.shape[0]
    m = max(1, int(round(d * ratio / 100.0)))

    def apply(name: str) -> torch.Tensor:
        if name == "dct":
            return dct_backproject(x_flat, m).reshape(c, f, t)
        if name == "gaussian":
            return gaussian_backproject(x_flat, m).reshape(c, f, t)
        if name == "patch_dct":
            m_per_patch = max(1, int(round(patch_f * patch_t * ratio / 100.0)))
            return dct_backproject_patch(mel.squeeze(0), m_per_patch, patch_f, patch_t).unsqueeze(0)
        raise ValueError(f"unknown sensing method: {name}")

    v1 = apply(left_name)
    v2 = apply(right_name)
    return v1, v2


_POLICY_LADDER = ["a0", "a1", "a2", "a3", "a4", "a5"]


class HybridBarlowDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        policy_strong: str,
        augment_config: dict,
        sensing_pair: str,
        ratio: int,
        cs_prob: float = 1.0,
        symmetric: bool = False,
        use_low_rank: bool = False,
    ) -> None:
        self.data_dir = data_dir.resolve()
        frame = load_manifest(self.data_dir, split)
        self.mel_paths = [
            resolve_relative_data_path(self.data_dir, str(r), use_low_rank) for r in frame["mel_path"]
        ]
        self.policy_strong = str(policy_strong)
        idx = _POLICY_LADDER.index(self.policy_strong)
        self.policy_weak = _POLICY_LADDER[max(0, idx - 1)]
        self.augment_config = augment_config
        self.left_name, self.right_name = parse_sensing_pair(sensing_pair)
        self.ratio = ratio
        self.patch_f = int(augment_config.get("patch_f", 16))
        self.patch_t = int(augment_config.get("patch_t", 16))
        self.cs_prob = float(cs_prob)
        self.symmetric = bool(symmetric)

    def __len__(self) -> int:
        return len(self.mel_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        mel_path = self.mel_paths[index]
        mel = torch.load(mel_path, map_location="cpu", weights_only=True).float()
        if mel.ndim != 2:
            raise ValueError(f"expected 2D mel tensor at {mel_path}")

        if self.symmetric:
            aug1 = apply_policy(mel, self.policy_strong, self.augment_config).unsqueeze(0)
            aug2 = apply_policy(mel, self.policy_strong, self.augment_config).unsqueeze(0)
            v1, _ = cs_view_pair(aug1, self.left_name, self.right_name, self.ratio, self.patch_f, self.patch_t)
            v2, _ = cs_view_pair(aug2, self.left_name, self.right_name, self.ratio, self.patch_f, self.patch_t)
            return v1, v2

        v2 = apply_policy(mel, self.policy_weak, self.augment_config).unsqueeze(0)
        aug = apply_policy(mel, self.policy_strong, self.augment_config).unsqueeze(0)
        if self.cs_prob >= 1.0 or random.random() < self.cs_prob:
            v1, _ = cs_view_pair(aug, self.left_name, self.right_name, self.ratio, self.patch_f, self.patch_t)
        else:
            v1 = aug
        return v1, v2


class FactoredHybridDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        policy_strong: str,
        augment_config: dict,
        r_L: int,
    ) -> None:
        self.data_dir = data_dir.resolve()
        frame = load_manifest(self.data_dir, split)
        self.mel_paths = [
            resolve_relative_data_path(self.data_dir, str(r), use_low_rank=False) for r in frame["mel_path"]
        ]
        self.lr_paths = [
            resolve_relative_data_path(self.data_dir, str(r), use_low_rank=True) for r in frame["mel_path"]
        ]
        self.policy_strong = str(policy_strong)
        idx = _POLICY_LADDER.index(self.policy_strong)
        self.policy_weak = _POLICY_LADDER[max(0, idx - 1)]
        self.augment_config = augment_config
        self.r_L = int(r_L)

    def __len__(self) -> int:
        return len(self.mel_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        X = torch.load(self.mel_paths[index], map_location="cpu", weights_only=True).float()
        L = torch.load(self.lr_paths[index],  map_location="cpu", weights_only=True).float()
        if X.ndim != 2 or L.ndim != 2:
            raise ValueError(f"expected 2D tensors at index {index}")
        S = (X - L).clamp_(0.0, 1.0)

        frames = int(self.augment_config["crop_frames"])
        t = X.shape[1]
        start = random.randint(0, max(t - frames, 0))
        L_crop = crop_or_pad(L[:, start:], frames, random_crop=False)
        S_crop = crop_or_pad(S[:, start:], frames, random_crop=False)

        X_crop = crop_or_pad(X[:, start:], frames, random_crop=False)
        v1 = apply_policy(X_crop, self.policy_strong, self.augment_config).unsqueeze(0)

        d = L_crop.numel()
        m = max(1, min(d, int(round(d * self.r_L / 100.0))))
        bp_L = dct_backproject(L_crop.reshape(-1), m).reshape(L_crop.shape)
        v2 = (bp_L + S_crop).clamp_(0.0, 1.0).unsqueeze(0)

        return v1, v2


class CSVICRegDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        sensing_pair: str,
        ratio: int,
        augment_config: dict,
        ref_coords: dict[str, np.ndarray] | None = None,
        use_low_rank: bool = False,
    ) -> None:
        self.data_dir = data_dir.resolve()
        frame = load_manifest(self.data_dir, split)
        self.mel_paths = [
            resolve_relative_data_path(self.data_dir, str(r), use_low_rank) for r in frame["mel_path"]
        ]
        self.left_name, self.right_name = parse_sensing_pair(sensing_pair)
        self.ratio = ratio
        self.augment_config = augment_config
        self.crop_frames = int(augment_config["crop_frames"])
        self.time_w = int(augment_config.get("time_mask_width", 0))
        self.freq_w = int(augment_config.get("freq_mask_width", 0))
        self.ref_coords = ref_coords

    def __len__(self) -> int:
        return len(self.mel_paths)

    def __getitem__(self, index: int):
        mel_path = self.mel_paths[index]
        mel = torch.load(mel_path, map_location="cpu", weights_only=True).float()
        if mel.ndim != 2:
            raise ValueError(f"expected 2D mel tensor at {mel_path}")

        mel = crop_or_pad(mel, self.crop_frames, random_crop=True)
        if self.time_w > 0 or self.freq_w > 0:
            mel = time_frequency_mask(mel, self.time_w, self.freq_w)

        mel_t = mel.unsqueeze(0)
        v1, v2 = cs_view_pair(mel_t, self.left_name, self.right_name, self.ratio)
        return v1, v2


class CSBarlowModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        base_channels: int,
        dropout: float,
        projection_hidden_dim: int,
        projection_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = CSEncoder(embedding_dim, base_channels, dropout)
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, projection_hidden_dim, bias=False),
            nn.BatchNorm1d(projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden_dim, projection_dim, bias=False),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        return h1, h2, z1, z2


def _load_waveform(audio_path: Path, sr: int, offset_sec: float, duration_sec: float) -> np.ndarray:
    npy_path = audio_path.with_suffix(".npy")
    if npy_path.exists():
        y = np.load(npy_path, mmap_mode="r")
        start = int(offset_sec * sr)
        n = int(duration_sec * sr)
        segment = y[start:start + n]
        if len(segment) < n:
            segment = np.pad(segment, (0, n - len(segment)))
        return np.array(segment, dtype=np.float32)
    cmd = [
        "ffmpeg", "-y", "-i", str(audio_path),
        "-ar", str(sr), "-ac", "1",
        "-ss", str(offset_sec), "-t", str(duration_sec),
        "-f", "f32le", "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
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


def _dct_cs_view(y: np.ndarray, ratio: float, rng: np.random.Generator) -> torch.Tensor:
    from scipy.fft import dct, idct
    n = len(y)
    m = max(1, int(round(n * ratio / 100.0)))
    coeffs = dct(y, norm="ortho", workers=1)
    probs = _get_dct_probs(n)
    idx = rng.choice(n, m, replace=False, p=probs)
    z = np.zeros(n, dtype=np.float32)
    z[idx] = coeffs[idx]
    recon = idct(z, norm="ortho", workers=1).astype(np.float32)
    return torch.from_numpy(recon)


WAVE_AUGMENTATION_POLICIES = ("w1", "w2", "w3", "w4")


def random_gain(y: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    gain = float(rng.uniform(1.0 - strength, 1.0 + strength))
    return (y * gain).astype(np.float32)


def random_time_stretch(y: np.ndarray, scale_range: tuple, rng: np.random.Generator) -> np.ndarray:
    from scipy.signal import resample
    scale = float(rng.uniform(float(scale_range[0]), float(scale_range[1])))
    n = len(y)
    n_resampled = max(1, int(round(n * scale)))
    y_stretched = resample(y.astype(np.float64), n_resampled).astype(np.float32)
    if n_resampled >= n:
        start = int(rng.integers(0, n_resampled - n + 1))
        return y_stretched[start : start + n]
    pad = n - n_resampled
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
    if policy not in WAVE_AUGMENTATION_POLICIES:
        raise ValueError(f"unsupported wave policy: {policy}")
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
    ) -> None:
        self.data_dir = data_dir.resolve()
        manifest = load_manifest(self.data_dir, split)
        self.rows = manifest.to_dict("records")
        self.ratio = int(ratio)
        self.segment_samples = int(sample_rate * segment_seconds)
        self.segment_seconds = float(segment_seconds)
        self.sample_rate = int(sample_rate)
        self.audio_root = audio_root.resolve()
        self.seed = seed
        self.epoch: int = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        audio_path = self.audio_root / Path(row["audio_path"])
        rng = np.random.default_rng([self.seed, self.epoch, index])
        offset = float(rng.uniform(10.0, 25.0))
        y = _load_waveform(audio_path, self.sample_rate, offset, self.segment_seconds)
        rng1 = np.random.default_rng([self.seed, self.epoch, index, 1])
        rng2 = np.random.default_rng([self.seed, self.epoch, index, 2])
        v1 = _dct_cs_view(y, self.ratio, rng1).unsqueeze(0)
        v2 = _dct_cs_view(y, self.ratio, rng2).unsqueeze(0)
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
    ) -> None:
        self.data_dir = data_dir.resolve()
        manifest = load_manifest(self.data_dir, split)
        self.rows = manifest.to_dict("records")
        self.policy = str(policy)
        self.segment_samples = int(sample_rate * segment_seconds)
        self.segment_seconds = float(segment_seconds)
        self.sample_rate = int(sample_rate)
        self.audio_root = audio_root.resolve()
        self.augment_config = augment_config
        self.seed = seed
        self.epoch: int = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        audio_path = self.audio_root / Path(row["audio_path"])
        rng = np.random.default_rng([self.seed, self.epoch, index])
        offset = float(rng.uniform(10.0, 25.0))
        y = _load_waveform(audio_path, self.sample_rate, offset, self.segment_seconds)
        rng1 = np.random.default_rng([self.seed, self.epoch, index, 1])
        rng2 = np.random.default_rng([self.seed, self.epoch, index, 2])
        v1 = torch.from_numpy(apply_wave_policy(y, self.policy, self.augment_config, rng1)).unsqueeze(0)
        v2 = torch.from_numpy(apply_wave_policy(y, self.policy, self.augment_config, rng2)).unsqueeze(0)
        return v1, v2


class HybridWaveDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        ratio: int,
        policy: str,
        segment_seconds: float,
        sample_rate: int,
        audio_root: Path,
        augment_config: dict,
        seed: int = 0,
    ) -> None:
        self.data_dir = data_dir.resolve()
        manifest = load_manifest(self.data_dir, split)
        self.rows = manifest.to_dict("records")
        self.ratio = int(ratio)
        self.policy = str(policy)
        self.segment_samples = int(sample_rate * segment_seconds)
        self.segment_seconds = float(segment_seconds)
        self.sample_rate = int(sample_rate)
        self.audio_root = audio_root.resolve()
        self.augment_config = augment_config
        self.seed = seed
        self.epoch: int = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        audio_path = self.audio_root / Path(row["audio_path"])
        rng = np.random.default_rng([self.seed, self.epoch, index])
        offset = float(rng.uniform(10.0, 25.0))
        y = _load_waveform(audio_path, self.sample_rate, offset, self.segment_seconds)
        rng1 = np.random.default_rng([self.seed, self.epoch, index, 1])
        rng2 = np.random.default_rng([self.seed, self.epoch, index, 2])
        v1 = _dct_cs_view(y, self.ratio, rng1).unsqueeze(0)
        v2 = torch.from_numpy(apply_wave_policy(y, self.policy, self.augment_config, rng2)).unsqueeze(0)
        return v1, v2


class _WaveResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
        ) if (stride > 1 or in_channels != out_channels) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x), inplace=True)


class WaveEncoder1D(nn.Module):
    def __init__(self, embedding_dim: int, base_channels: int = 128) -> None:
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv1d(1, c, kernel_size=15, stride=3, padding=7, bias=False),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.Sequential(
            _WaveResBlock(c,     c,     stride=3),
            _WaveResBlock(c,     c,     stride=3),
            _WaveResBlock(c,     c * 2, stride=3),
            _WaveResBlock(c * 2, c * 2, stride=3),
            _WaveResBlock(c * 2, c * 4, stride=3),
            _WaveResBlock(c * 4, c * 4, stride=3),
        )
        self.head = nn.Linear(c * 4 * 2, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layers(x)
        avg = x.mean(dim=2)
        mx = x.amax(dim=2)
        return self.head(torch.cat([avg, mx], dim=1))


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
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.register_buffer("window", torch.hann_window(n_fft))
        import torchaudio.functional as AF
        fb = AF.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0.0,
            f_max=float(sample_rate) / 2.0,
            n_mels=int(n_mels),
            sample_rate=int(sample_rate),
            norm="slaney",
            mel_scale="htk",
        )
        self.register_buffer("mel_fb", fb)
        channels = [base_channels * (2 ** i) for i in range(int(n_blocks))]
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            ])
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-1] * 2, embedding_dim)

    def _to_mel(self, x: torch.Tensor) -> torch.Tensor:
        y = x.squeeze(1)
        spec = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
        )
        mag = spec.abs()
        mel = torch.einsum("bft,fm->bmt", mag, self.mel_fb)
        return torch.log1p(mel).unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = self._to_mel(x)
        feat = self.features(spec)
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
        self.encoder = WaveSTFTEncoder(embedding_dim, base_channels, n_fft, hop_length, n_blocks, n_mels, sample_rate)
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, projection_hidden_dim, bias=False),
            nn.BatchNorm1d(projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden_dim, projection_dim, bias=False),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        return h1, h2, z1, z2
