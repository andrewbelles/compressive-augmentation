#!/usr/bin/env python3
#
# audio.py  Andrew Belles  May 8th, 2026
#
# Audio Barlow Twins + CS-VICReg models and mel crop/sensing utilities.
#

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


AUGMENTATION_POLICIES = ("a0", "a1", "a2", "a3", "a4")


def resolve_relative_data_path(base_dir: Path, manifest_path: str) -> Path:
    relative_path = Path(str(manifest_path))
    if relative_path.parts and relative_path.parts[0] == base_dir.name:
        relative_path = Path(*relative_path.parts[1:])
    return base_dir / relative_path


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
    if policy in {"a1", "a2", "a3", "a4"}:
        output = random_resize_crop(mel, frames, tuple(config["resize_scale"]))
    else:
        raise ValueError(f"unsupported augmentation policy: {policy}")

    if policy in {"a3", "a4"}:
        output = random_linear_fader(output, float(config["linear_fader_strength"]))
    if policy == "a4":
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
    def __init__(self, data_dir: Path, split: str, policy: str, augment_config: dict, paired: bool, return_labels: bool = False):
        self.data_dir = data_dir.resolve()
        self.root_dir = self.data_dir.parent
        self.frame = load_manifest(self.data_dir, split)
        self.policy = str(policy)
        self.augment_config = augment_config
        self.paired = bool(paired)
        self.return_labels = bool(return_labels)
        genres = sorted(str(value) for value in self.frame["genre_top"].dropna().unique())
        self.genre_to_index = {genre: index for index, genre in enumerate(genres)}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index].to_dict()
        mel_path = resolve_relative_data_path(self.data_dir, str(row["mel_path"]))
        mel = torch.load(mel_path, map_location="cpu", weights_only=True).float()
        if mel.ndim != 2:
            raise ValueError(f"expected 2D mel tensor at {mel_path}, got {tuple(mel.shape)}")

        if self.paired:
            left = apply_policy(mel, self.policy, self.augment_config)
            right = left.clone() if self.policy == "a0" else apply_policy(mel, self.policy, self.augment_config)
            if self.return_labels:
                label = self.genre_to_index[str(row["genre_top"])]
                return left.unsqueeze(0).contiguous(), right.unsqueeze(0).contiguous(), label
            return left.unsqueeze(0).contiguous(), right.unsqueeze(0).contiguous()

        crop = crop_or_pad(mel, int(self.augment_config["crop_frames"]), random_crop=False)
        return crop.unsqueeze(0).contiguous(), row


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


def barlow_twins_loss(left: torch.Tensor, right: torch.Tensor, lambd: float) -> torch.Tensor:
    batch_size = left.size(0)
    left = (left - left.mean(dim=0)) / left.std(dim=0).clamp_min(1e-6)
    right = (right - right.mean(dim=0)) / right.std(dim=0).clamp_min(1e-6)
    correlation = left.T @ right / batch_size
    on_diag = torch.diagonal(correlation).add_(-1.0).pow_(2).sum()
    off_diag = off_diagonal(correlation).pow_(2).sum()
    return on_diag + float(lambd) * off_diag


def supervised_contrastive_loss(left: torch.Tensor, right: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    if left.size(0) != right.size(0) or left.size(0) != labels.size(0):
        raise ValueError("left, right, and labels must have matching batch sizes")
    features = F.normalize(torch.cat([left, right], dim=0), dim=1)
    repeated_labels = labels.reshape(-1, 1).repeat(2, 1)
    positive_mask = torch.eq(repeated_labels, repeated_labels.T).float().to(features.device)
    logits = features @ features.T / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(logits.size(0), dtype=torch.float32, device=features.device)
    positive_mask = positive_mask * (1.0 - self_mask)
    exp_logits = torch.exp(logits) * (1.0 - self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positives_per_row = positive_mask.sum(dim=1).clamp_min(1.0)
    return -((positive_mask * log_prob).sum(dim=1) / positives_per_row).mean()


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


def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _fwht_inplace(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    h = 1
    while h < n:
        x = x.reshape(-1, h * 2)
        u = x[:, :h].clone()
        v = x[:, h:].clone()
        x[:, :h] = u + v
        x[:, h:] = u - v
        x = x.reshape(-1)
        h *= 2
    return x


def srht_backproject(x_flat: torch.Tensor, m: int) -> torch.Tensor:
    d = x_flat.shape[0]
    d2 = _next_power_of_two(d)
    xp = F.pad(x_flat, (0, d2 - d))
    signs = torch.randint(0, 2, (d2,), device=x_flat.device, dtype=x_flat.dtype) * 2 - 1
    xp = xp * signs
    xp = _fwht_inplace(xp) / math.sqrt(d2)
    m = min(m, d2)
    rows = torch.randperm(d2, device=x_flat.device)[:m]
    y = xp[rows]
    z = torch.zeros(d2, device=x_flat.device, dtype=x_flat.dtype)
    z[rows] = y
    x_hat = _fwht_inplace(z) / math.sqrt(d2) * signs
    return x_hat[:d] * (d2 / m)


def dct_backproject(x_flat: torch.Tensor, m: int) -> torch.Tensor:
    d = x_flat.shape[0]
    x_np = x_flat.cpu().numpy().astype(np.float64)
    from scipy.fft import dct, idct
    coeffs = dct(x_np, norm="ortho")
    idx = np.random.choice(d, size=min(m, d), replace=False)
    z = np.zeros(d, dtype=np.float64)
    z[idx] = coeffs[idx]
    x_hat = idct(z, norm="ortho").astype(np.float32)
    return torch.from_numpy(x_hat).to(x_flat.device)


def gaussian_backproject(x_flat: torch.Tensor, m: int) -> torch.Tensor:
    d = x_flat.shape[0]
    m = min(m, d)
    Phi = torch.randn(m, d, device=x_flat.device, dtype=x_flat.dtype) / math.sqrt(m)
    y = Phi @ x_flat
    return Phi.t() @ y


def cs_view_pair(
    mel: torch.Tensor,
    sensing_pair: str,
    ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    c, f, t = mel.shape
    x_flat = mel.reshape(-1)
    d = x_flat.shape[0]
    m = max(1, int(round(d * ratio / 100.0)))

    known = {"srht", "dct", "gaussian"}
    parts = sensing_pair.split("_")
    left_name = parts[0]
    right_name = "_".join(parts[1:]) if len(parts) > 1 else parts[0]
    if left_name not in known or right_name not in known:
        raise ValueError(f"unsupported sensing_pair: {sensing_pair!r}")

    def apply(name: str) -> torch.Tensor:
        if name == "srht":
            return srht_backproject(x_flat, m)
        if name == "dct":
            return dct_backproject(x_flat, m)
        if name == "gaussian":
            return gaussian_backproject(x_flat, m)
        raise ValueError(f"unknown sensing method: {name}")

    v1 = apply(left_name).reshape(c, f, t)
    v2 = apply(right_name).reshape(c, f, t)
    return v1, v2


class CSVICRegDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        sensing_pair: str,
        ratio: int,
        augment_config: dict,
    ) -> None:
        self.data_dir = data_dir.resolve()
        self.frame = load_manifest(self.data_dir, split)
        self.sensing_pair = sensing_pair
        self.ratio = ratio
        self.augment_config = augment_config
        self.crop_frames = int(augment_config["crop_frames"])

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index].to_dict()
        mel_path = resolve_relative_data_path(self.data_dir, str(row["mel_path"]))
        mel = torch.load(mel_path, map_location="cpu", weights_only=True).float()
        if mel.ndim != 2:
            raise ValueError(f"expected 2D mel tensor at {mel_path}")

        mel = crop_or_pad(mel, self.crop_frames, random_crop=True)
        time_w = int(self.augment_config.get("time_mask_width", 0))
        freq_w = int(self.augment_config.get("freq_mask_width", 0))
        if time_w > 0 or freq_w > 0:
            mel = time_frequency_mask(mel, time_w, freq_w)

        mel_t = mel.unsqueeze(0).contiguous()
        v1, v2 = cs_view_pair(mel_t, self.sensing_pair, self.ratio)
        return v1.contiguous(), v2.contiguous()


class CSVICRegModel(nn.Module):
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
            nn.Linear(projection_hidden_dim, projection_hidden_dim, bias=False),
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


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    inv_w: float,
    var_w: float,
    cov_w: float,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inv = F.mse_loss(z1, z2)

    def variance_term(z: torch.Tensor) -> torch.Tensor:
        std = z.std(dim=0, correction=1).clamp_min(1e-4)
        return F.relu(gamma - std).mean()

    var = 0.5 * (variance_term(z1) + variance_term(z2))

    def covariance_term(z: torch.Tensor) -> torch.Tensor:
        n, d = z.shape
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (n - 1)
        return off_diagonal(cov).pow_(2).sum() / d

    cov = 0.5 * (covariance_term(z1) + covariance_term(z2))

    total = inv_w * inv + var_w * var + cov_w * cov
    return total, inv, var, cov
