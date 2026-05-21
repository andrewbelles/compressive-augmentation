#!/usr/bin/env python3
#
# cs_mel.py  Andrew Belles  May 13th, 2026
#
# Compressive sensing experiments directly on mel-spectrogram images.
# Each track is treated as a vector x in R^d where d = mel_bins * mel_frames (default 64*128=8192).
#
# All methods use m = ratio_to_m(d, r) measurements, making ratios directly comparable.
# Baselines (classical CS measurement operators):
#
#   gaussian_random    -- Gaussian random projection: y = Phi x, Phi ~ N(0,1/m), rows sampled iid.
#   sfrt               -- SFRT: y = R * F * D * x  (random sign flip, FFT, random row sample).
#   shrt               -- SHRT: y = R * H * D * x  (random sign flip, fast WHT, random row sample).
#   dct_random         -- Partial DCT: y = DCT(x)[support], assumes x frequency-sparse.
#
# Hypothesis (learned sparse coding):
#   patch_dictionary   -- learned basis.  LARS path over D^T; pick step with k_target active atoms.
#
# cs_mel.py stores raw measurements y = Phi x only -- no reconstruction.
# CoSaMP reconstruction lives in cs_mel_eval.py and runs on demand for diagnostic purposes.
# One parquet file is written per method, keyed internally by ratio_percent.
#

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.fft import dct
from scipy.signal import get_window
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import lars_path

from compression.train_utils import load_config


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "cs_mel.yaml"
DEFAULT_CONFIG: dict = {
    "dataset": "fma_small_mel",
    "mel_dir": "preprocess/data/fma_small_mel",
    "output_dir": "compression/data",
    "mel_bins": 64,
    "mel_frames": 128,
    "ratios": [1, 2, 4, 8, 12, 16, 24, 50],
    "methods": [
        "gaussian_random",
        "sfrt",
        "shrt",
        "dct_random",
        "patch_dictionary",
        "shrt_cqt",
    ],
    "n_eval_samples": 128,
    "seed": 0,
    "dictionary": {
        "n_components": 128,
        "max_iter": 500,
    },
    "cqt": {
        "sample_rate": 22050,
        "fmin": 32.7,
        "n_bins": 84,
        "bins_per_octave": 12,
        "hop_length": 512,
        "n_frames": 128,
    },
}

_DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def report(message: str) -> None:
    print(message, flush=True)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CS mel-spectrogram sensing experiments.")
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def load_manifest(mel_dir: Path, split: str) -> list[dict[str, str]]:
    path = mel_dir / f"manifest_{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing manifest: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_mel_tensor(mel_dir: Path, row: dict[str, str], mel_bins: int, mel_frames: int) -> np.ndarray | None:
    rel = row.get("mel_path", "")
    if not rel:
        return None
    path = mel_dir.parent / rel
    if not path.is_file():
        return None
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    if tensor.dim() != 2 or tensor.shape[0] != mel_bins:
        return None
    t = tensor.shape[1]
    if t >= mel_frames:
        sliced = tensor[:, :mel_frames]
    else:
        sliced = torch.nn.functional.pad(tensor, (0, mel_frames - t))
    return sliced.numpy().astype(np.float32).ravel()


def load_audio_ffmpeg(audio_path: Path, sample_rate: int) -> np.ndarray | None:
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(audio_path),
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sample_rate), "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True)
    except Exception:
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def _build_cqt_kernels(
    sr: int,
    fmin: float,
    n_bins: int,
    bins_per_octave: int,
) -> tuple[list[np.ndarray], list[int], int]:
    Q = 1.0 / (2 ** (1.0 / bins_per_octave) - 1)
    freqs = fmin * 2 ** (np.arange(n_bins) / bins_per_octave)
    kernels: list[np.ndarray] = []
    Nks: list[int] = []
    for fc in freqs:
        N_k = int(np.round(Q * sr / fc))
        win = get_window("hann", N_k, fftbins=True).astype(np.float32)
        t = np.arange(N_k, dtype=np.float32) / sr
        kernel = (win * np.exp(-2j * np.pi * fc * t) / N_k).astype(np.complex64)
        kernels.append(kernel)
        Nks.append(N_k)
    max_Nk = max(Nks)
    return kernels, Nks, max_Nk


def compute_cqt(
    audio: np.ndarray,
    kernels: list[np.ndarray],
    Nks: list[int],
    max_Nk: int,
    hop_length: int,
    n_frames: int,
    device: torch.device,
) -> np.ndarray:
    audio_pad = np.pad(audio.astype(np.float32), (max_Nk // 2, max_Nk // 2 + n_frames * hop_length))
    fft_len = len(audio_pad)
    Audio = torch.fft.rfft(torch.from_numpy(audio_pad).to(device), n=fft_len)
    n_bins = len(kernels)
    rfft_bins = fft_len // 2 + 1
    k_batch = np.zeros((n_bins, fft_len), dtype=np.complex64)
    for b, (kernel, N_k) in enumerate(zip(kernels, Nks)):
        k_batch[b, :N_k] = kernel[::-1]
    K_full = torch.fft.fft(torch.from_numpy(k_batch).to(device))
    K_mat = K_full[:, :rfft_bins]
    Conv = torch.fft.irfft(K_mat * Audio.unsqueeze(0), n=fft_len)
    frames = Conv[:, ::hop_length][:, :n_frames]
    cqt = frames.abs()
    return cqt.cpu().numpy()


def _log_normalize_cqt(C: np.ndarray) -> np.ndarray:
    C_db = 20.0 * np.log10(np.maximum(C, 1e-10))
    c_min, c_max = C_db.min(), C_db.max()
    return ((C_db - c_min) / max(c_max - c_min, 1e-6)).astype(np.float32)


def load_cqt_tensor(
    mel_dir: Path,
    row: dict[str, str],
    kernels: list[np.ndarray],
    Nks: list[int],
    max_Nk: int,
    hop_length: int,
    n_frames: int,
    sample_rate: int,
    device: torch.device,
) -> np.ndarray | None:
    mel_rel = row.get("mel_path", "")
    if not mel_rel:
        return None
    mel_path = mel_dir.parent / mel_rel
    cqt_path = mel_path.with_suffix(".cqt.pt")
    if cqt_path.is_file():
        try:
            t = torch.load(cqt_path, map_location="cpu", weights_only=True)
            return t.numpy().astype(np.float32).ravel()
        except Exception:
            pass
    audio_rel = row.get("audio_path", "")
    if not audio_rel:
        return None
    audio_path = mel_dir.parent / audio_rel
    if not audio_path.is_file():
        return None
    audio = load_audio_ffmpeg(audio_path, sample_rate)
    if audio is None or len(audio) == 0:
        return None
    C = compute_cqt(audio, kernels, Nks, max_Nk, hop_length, n_frames, device)
    C_norm = _log_normalize_cqt(C)
    try:
        torch.save(torch.from_numpy(C_norm), cqt_path)
    except Exception:
        pass
    return C_norm.ravel()


def ratio_to_m(d: int, ratio_percent: int) -> int:
    return max(1, min(d, int(round(d * ratio_percent / 100.0))))


def fwht(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    n = len(y)
    h = 1
    while h < n:
        y = y.reshape(-1, h * 2)
        u, v = y[:, :h].copy(), y[:, h:].copy()
        y[:, :h] = u + v
        y[:, h:] = u - v
        y = y.ravel()
        h *= 2
    return y


def fwht_torch(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    h = 1
    while h < d:
        x = x.reshape(*x.shape[:-1], -1, h * 2)
        u = x[..., :h].clone()
        v = x[..., h:].clone()
        x = torch.cat([u + v, u - v], dim=-1)
        x = x.reshape(*x.shape[:-2], -1)
        h *= 2
    return x


def sensing_gaussian_random_batch(
    X_np: np.ndarray,
    m: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    d = X_np.shape[1]
    Phi_np = rng.standard_normal((m, d)).astype(np.float32) / math.sqrt(m)
    Phi = torch.from_numpy(Phi_np).to(_DEVICE)
    X_gpu = torch.from_numpy(X_np).to(_DEVICE)
    Y = X_gpu @ Phi.T
    if _DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return Y.cpu().numpy(), Phi_np


def sensing_sfrt_batch(
    X_np: np.ndarray,
    m: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, d = X_np.shape
    n_rfft = d // 2 + 1
    m_clamped = min(m, n_rfft)
    signs_np = (rng.integers(0, 2, size=d).astype(np.float32) * 2 - 1)
    row_idx_np = np.sort(rng.choice(n_rfft, size=m_clamped, replace=False))
    signs = torch.from_numpy(signs_np).to(_DEVICE)
    row_idx = torch.from_numpy(row_idx_np).to(_DEVICE)
    X_gpu = torch.from_numpy(X_np).to(_DEVICE)
    Y = torch.fft.rfft(X_gpu * signs.unsqueeze(0), dim=-1)[:, row_idx] / math.sqrt(d)
    if _DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return Y.cpu().numpy(), signs_np, row_idx_np


def sensing_shrt_batch(
    X_np: np.ndarray,
    m: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, d = X_np.shape
    signs_np = (rng.integers(0, 2, size=d).astype(np.float32) * 2 - 1)
    row_idx_np = np.sort(rng.choice(d, size=min(m, d), replace=False))
    signs = torch.from_numpy(signs_np).to(_DEVICE)
    row_idx = torch.from_numpy(row_idx_np).to(_DEVICE)
    X_gpu = torch.from_numpy(X_np).to(_DEVICE)
    Y = fwht_torch(X_gpu * signs.unsqueeze(0))[:, row_idx] / math.sqrt(d)
    if _DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return Y.cpu().numpy(), signs_np, row_idx_np


def sensing_dct_random_batch(
    X_np: np.ndarray,
    m: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    d = X_np.shape[1]
    row_idx_np = np.sort(rng.choice(d, size=min(m, d), replace=False))
    coeffs = dct(X_np.astype(np.float64), norm="ortho", axis=-1)
    Y = coeffs[:, row_idx_np].astype(np.float32)
    return Y, row_idx_np


def sensing_shrt_cqt_batch(
    X_np: np.ndarray,
    m: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, d = X_np.shape
    d_pad = 1 << math.ceil(math.log2(max(d, 2)))
    signs_np = (rng.integers(0, 2, size=d_pad).astype(np.float32) * 2 - 1)
    row_idx_np = np.sort(rng.choice(d_pad, size=min(m, d_pad), replace=False))
    signs = torch.from_numpy(signs_np).to(_DEVICE)
    row_idx = torch.from_numpy(row_idx_np).to(_DEVICE)
    X_pad = torch.zeros(N, d_pad, device=_DEVICE, dtype=torch.float32)
    X_pad[:, :d] = torch.from_numpy(X_np).to(_DEVICE)
    Y = fwht_torch(X_pad * signs.unsqueeze(0))[:, row_idx] / math.sqrt(d_pad)
    if _DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return Y.cpu().numpy(), signs_np, row_idx_np


def fit_patch_dictionary(
    X_train: np.ndarray,
    n_components: int,
    max_iter: int,
    seed: int,
) -> np.ndarray:
    learner = MiniBatchDictionaryLearning(
        n_components=n_components,
        max_iter=max_iter,
        random_state=seed,
        fit_algorithm="cd",
        n_jobs=1,
    )
    learner.fit(X_train)
    return learner.components_.astype(np.float32)


def sensing_patch_dictionary(
    x: np.ndarray,
    dictionary: np.ndarray,
    k_target: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, coef_path = lars_path(
        dictionary.T,
        x.astype(np.float64),
        method="lasso",
        max_iter=dictionary.shape[0],
    )
    nnz = np.count_nonzero(coef_path, axis=0)
    step = int(np.searchsorted(nnz, k_target, side="right")) - 1
    step = max(0, min(step, coef_path.shape[1] - 1))
    codes = coef_path[:, step].astype(np.float32)
    support = np.where(codes != 0.0)[0]
    alpha_vals = codes[support]
    x_hat = (dictionary[support].T @ alpha_vals.reshape(-1, 1)).ravel().astype(np.float32)
    return alpha_vals, support, x_hat


def process_split(
    method: str,
    split: str,
    rows: list[dict[str, str]],
    mel_dir: Path,
    mel_bins: int,
    mel_frames: int,
    ratios: list[int],
    seed: int,
    dataset: str,
    dictionary: np.ndarray | None = None,
) -> list[dict]:
    d = mel_bins * mel_frames
    records: list[dict] = []
    n_rows = len(rows)
    n_skipped = 0
    log_every = max(1, n_rows // 10)
    report(f"  [{method}|{split}] processing {n_rows} manifest rows...")

    if method == "patch_dictionary":
        n_loaded = 0
        for i, row in enumerate(rows):
            x = load_mel_tensor(mel_dir, row, mel_bins, mel_frames)
            if x is None:
                n_skipped += 1
                continue
            n_loaded += 1
            track_id = int(row["track_id"])
            genre = row.get("genre_top", "")
            for ratio in ratios:
                assert dictionary is not None
                n_components = dictionary.shape[0]
                k_target = max(1, min(n_components, round(ratio / 100.0 * n_components)))
                alpha_vals, support, _ = sensing_patch_dictionary(x, dictionary, k_target)
                records.append({
                    "track_id": track_id, "genre_top": genre, "split": split,
                    "method": method, "dataset": dataset, "ratio_percent": ratio,
                    "m_dim": ratio_to_m(d, ratio), "d_dim": d, "seed": seed,
                    "alpha_values": alpha_vals.tolist(), "alpha_support": support.tolist(),
                })
            if (i + 1) % log_every == 0 or (i + 1) == n_rows:
                report(f"  [{method}|{split}] {i+1}/{n_rows} loaded={n_loaded} skipped={n_skipped} records={len(records)}")
        report(f"  [{method}|{split}] done: loaded={n_loaded} skipped={n_skipped} records={len(records)}")
        return records

    vectors: list[np.ndarray] = []
    meta: list[tuple[int, str]] = []
    for i, row in enumerate(rows):
        x = load_mel_tensor(mel_dir, row, mel_bins, mel_frames)
        if x is None:
            n_skipped += 1
        else:
            vectors.append(x)
            meta.append((int(row["track_id"]), row.get("genre_top", "")))
        if (i + 1) % log_every == 0 or (i + 1) == n_rows:
            report(f"  [{method}|{split}] loading {i+1}/{n_rows} loaded={len(vectors)} skipped={n_skipped}")

    n_loaded = len(vectors)
    if n_loaded == 0:
        report(f"  [{method}|{split}] done: loaded=0 skipped={n_skipped} records=0")
        return records

    X_np = np.stack(vectors, axis=0).astype(np.float32)

    for ratio in ratios:
        m = ratio_to_m(d, ratio)
        report(f"  [{method}|{split}] ratio={ratio}% m={m} ...")
        method_rng = np.random.default_rng(seed + hash(method + split + str(ratio)) % (2**31))

        if method == "gaussian_random":
            Y_np, Phi_np = sensing_gaussian_random_batch(X_np, m, method_rng)
            for sample_idx, (track_id, genre) in enumerate(meta):
                records.append({
                    "track_id": track_id, "genre_top": genre, "split": split,
                    "method": method, "dataset": dataset, "ratio_percent": ratio,
                    "m_dim": m, "d_dim": d, "seed": seed,
                    "alpha_values": Y_np[sample_idx].tolist(), "alpha_support": [],
                })

        elif method == "sfrt":
            Y_np, signs_np, row_idx = sensing_sfrt_batch(X_np, m, method_rng)
            for sample_idx, (track_id, genre) in enumerate(meta):
                records.append({
                    "track_id": track_id, "genre_top": genre, "split": split,
                    "method": method, "dataset": dataset, "ratio_percent": ratio,
                    "m_dim": m, "d_dim": d, "seed": seed,
                    "alpha_values": Y_np[sample_idx].tolist(), "alpha_support": row_idx.tolist(),
                })

        elif method == "shrt":
            Y_np, signs_np, row_idx = sensing_shrt_batch(X_np, m, method_rng)
            for sample_idx, (track_id, genre) in enumerate(meta):
                records.append({
                    "track_id": track_id, "genre_top": genre, "split": split,
                    "method": method, "dataset": dataset, "ratio_percent": ratio,
                    "m_dim": m, "d_dim": d, "seed": seed,
                    "alpha_values": Y_np[sample_idx].tolist(), "alpha_support": row_idx.tolist(),
                })

        elif method == "dct_random":
            Y_np, row_idx = sensing_dct_random_batch(X_np, m, method_rng)
            for sample_idx, (track_id, genre) in enumerate(meta):
                records.append({
                    "track_id": track_id, "genre_top": genre, "split": split,
                    "method": method, "dataset": dataset, "ratio_percent": ratio,
                    "m_dim": m, "d_dim": d, "seed": seed,
                    "alpha_values": Y_np[sample_idx].tolist(), "alpha_support": row_idx.tolist(),
                })

        report(f"  [{method}|{split}] ratio={ratio}% done records_so_far={len(records)}")

    report(f"  [{method}|{split}] done: loaded={n_loaded} skipped={n_skipped} records={len(records)}")
    return records


def process_split_cqt(
    split: str,
    rows: list[dict[str, str]],
    mel_dir: Path,
    ratios: list[int],
    seed: int,
    dataset: str,
    cqt_cfg: dict,
) -> list[dict]:
    sr = int(cqt_cfg.get("sample_rate", 22050))
    fmin = float(cqt_cfg.get("fmin", 32.7))
    n_bins = int(cqt_cfg.get("n_bins", 84))
    bins_per_octave = int(cqt_cfg.get("bins_per_octave", 12))
    hop_length = int(cqt_cfg.get("hop_length", 512))
    n_frames = int(cqt_cfg.get("n_frames", 128))
    d_cqt = n_bins * n_frames
    method = "shrt_cqt"

    kernels, Nks, max_Nk = _build_cqt_kernels(sr, fmin, n_bins, bins_per_octave)
    records: list[dict] = []
    n_rows = len(rows)
    n_skipped = 0
    log_every = max(1, n_rows // 10)
    report(f"  [{method}|{split}] processing {n_rows} manifest rows (d_cqt={d_cqt})...")

    vectors: list[np.ndarray] = []
    meta: list[tuple[int, str]] = []
    for i, row in enumerate(rows):
        x = load_cqt_tensor(mel_dir, row, kernels, Nks, max_Nk, hop_length, n_frames, sr, _DEVICE)
        if x is None:
            n_skipped += 1
        else:
            vectors.append(x)
            meta.append((int(row["track_id"]), row.get("genre_top", "")))
        if (i + 1) % log_every == 0 or (i + 1) == n_rows:
            report(f"  [{method}|{split}] loading {i+1}/{n_rows} loaded={len(vectors)} skipped={n_skipped}")

    n_loaded = len(vectors)
    if n_loaded == 0:
        report(f"  [{method}|{split}] done: loaded=0 skipped={n_skipped} records=0")
        return records

    X_np = np.stack(vectors, axis=0).astype(np.float32)

    for ratio in ratios:
        m = ratio_to_m(d_cqt, ratio)
        report(f"  [{method}|{split}] ratio={ratio}% m={m} ...")
        method_rng = np.random.default_rng(seed + hash(method + split + str(ratio)) % (2**31))
        Y_np, signs_np, row_idx = sensing_shrt_cqt_batch(X_np, m, method_rng)
        for sample_idx, (track_id, genre) in enumerate(meta):
            records.append({
                "track_id": track_id, "genre_top": genre, "split": split,
                "method": method, "dataset": dataset, "ratio_percent": ratio,
                "m_dim": m, "d_dim": d_cqt, "seed": seed,
                "alpha_values": Y_np[sample_idx].tolist(), "alpha_support": row_idx.tolist(),
            })
        report(f"  [{method}|{split}] ratio={ratio}% done records_so_far={len(records)}")

    report(f"  [{method}|{split}] done: loaded={n_loaded} skipped={n_skipped} records={len(records)}")
    return records


def run(config: dict, output_dir: Path) -> list[Path]:
    mel_dir = Path(str(config["mel_dir"])).expanduser().resolve()
    mel_bins = int(config["mel_bins"])
    mel_frames = int(config["mel_frames"])
    ratios = [int(r) for r in config["ratios"]]
    methods = [str(m) for m in config["methods"]]
    seed = int(config["seed"])
    dataset = str(config["dataset"])
    dict_cfg = config.get("dictionary", {})
    n_components = int(dict_cfg.get("n_components", 128))
    max_iter = int(dict_cfg.get("max_iter", 500))

    report(f"device={_DEVICE}")

    rng = np.random.default_rng(seed)

    splits_rows: dict[str, list[dict[str, str]]] = {}
    for split in ("training", "validation", "test"):
        splits_rows[split] = load_manifest(mel_dir, split)

    dictionary: np.ndarray | None = None
    if "patch_dictionary" in methods:
        train_rows = splits_rows["training"]
        report(f"fitting patch dictionary on training set ({len(train_rows)} rows)...")
        X_train_list: list[np.ndarray] = []
        for i, row in enumerate(train_rows):
            x = load_mel_tensor(mel_dir, row, mel_bins, mel_frames)
            if x is not None:
                X_train_list.append(x)
            if (i + 1) % max(1, len(train_rows) // 10) == 0 or (i + 1) == len(train_rows):
                report(f"  [dictionary] loaded {len(X_train_list)}/{i+1} tensors")
        if not X_train_list:
            raise RuntimeError("no training mel tensors found for dictionary learning")
        X_train = np.stack(X_train_list, axis=0)
        report(f"  [dictionary] fitting MiniBatchDictionaryLearning n_components={n_components} max_iter={max_iter} X={X_train.shape}...")
        dictionary = fit_patch_dictionary(X_train, n_components, max_iter, seed)
        report(f"  [dictionary] done shape={dictionary.shape}")

    cqt_cfg: dict = config.get("cqt", DEFAULT_CONFIG["cqt"])

    written: list[Path] = []
    for method_idx, method in enumerate(methods):
        report(f"[{method_idx+1}/{len(methods)}] method={method}")
        all_records: list[dict] = []
        for split, rows in splits_rows.items():
            if method == "shrt_cqt":
                records = process_split_cqt(
                    split, rows, mel_dir, ratios, seed, dataset, cqt_cfg,
                )
            else:
                records = process_split(
                    method, split, rows, mel_dir, mel_bins, mel_frames,
                    ratios, seed, dataset, dictionary,
                )
            all_records.extend(records)
            report(f"  [{method}|{split}] total records={len(records)}")

        if not all_records:
            log(f"no records for method={method}, skipping")
            continue

        frame = pd.DataFrame.from_records(all_records)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"cs_mel_{method}_{dataset}.parquet"
        frame.to_parquet(out_path, index=False)
        written.append(out_path)
        report(f"wrote {out_path} rows={len(frame)}")

    return written


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_CONFIG)
    output_dir = Path(str(config["output_dir"])).expanduser().resolve()
    report(
        f"START module=compression.cs_mel methods={config['methods']} "
        f"ratios={config['ratios']} d={int(config['mel_bins'])*int(config['mel_frames'])}"
    )
    written = run(config, output_dir)
    report(f"DONE module=compression.cs_mel files={len(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
