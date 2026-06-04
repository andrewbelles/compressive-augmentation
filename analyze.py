#!/usr/bin/env python3
#
# analyze.py  Andrew Belles  June 2026
#
# Post-training analysis for the compressive SSL project.
# Reads the consolidated parquet and produces:
#
#   1. linear      -- linear probe (logistic regression, C grid, bootstrap CI, t-CI across seeds)
#   2. comparison  -- paired bootstrap delta vs best traditional baseline
#   3. perturbation -- semantic/nuisance decomposition vs SupCon reference manifold
#   4. alignment   -- between-views alignment and uniformity vs ratio
#
# All results saved as CSVs under --output-dir.
#
# Usage:
#   python analyze.py \
#       --parquet data/wave_barlow_fma_small.parquet \
#       --output-dir analysis/ \
#       --checkpoint-dir /path/to/checkpoints \
#       --audio-root preprocess/data/fma_small_mel
#

import argparse
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score as sklearn_f1
from sklearn.preprocessing import LabelEncoder, StandardScaler

from common.data import load_manifest
from common.model import WaveSTFTEncoder
from common.ops import gpu_dct_cs_view_batch, gpu_srht_batch, gpu_wave_policy_batch


SPLITS         = ("training", "validation", "test")
C_GRID         = list(np.logspace(-4, 2, 15).tolist())
N_BOOT         = 2000
BOOT_SEED      = 42
MEL_PCA_DIM    = 256
EPS            = 1e-12
TRAD_FAMILIES  = {"w2", "w3", "w4"}


def emb_cols(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("embedding_") and df[c].notna().all())


def parse_method(method: str) -> tuple[str, int | None]:
    base = re.sub(r"_s\d+$", "", method)
    m    = re.search(r"_r0?(\d+)", base)
    ratio = int(m.group(1)) if m else None
    if "srht"    in base: return "srht",        ratio
    if "uniform" in base: return "dct_uniform", ratio
    if "cs"      in base: return "dct_biased",  ratio
    if "supcon"  in base: return "supcon",       ratio
    m2 = re.search(r"_w(\d)", base)
    if m2: return f"w{m2.group(1)}", ratio
    return "other", ratio


def method_base(method: str) -> str:
    return re.sub(r"_s\d+$", "", str(method))


def method_label(method: str) -> str:
    base   = method_base(method)
    simple = {
        "raw_mel_pca256":                    "Mel PCA-256",
        "wave_barlow_abt_w2_d256_nopop":     "W2",
        "wave_barlow_abt_w3_d256_nopop":     "W3",
        "wave_barlow_abt_w4_d256_nopop":     "W4",
    }
    if base in simple:
        return simple[base]
    if base.startswith("supcon"):
        return "SupCon-W3"
    _, ratio = parse_method(base)
    r = f"r{ratio}" if ratio is not None else "?"
    if "srht"    in base: return f"SRHT {r}"
    if "uniform" in base: return f"DCT-U {r}"
    if "cs"      in base: return f"DCT-B {r}"
    return base


def split_xy(df: pd.DataFrame, cols: list[str], exclude_genres=None):
    if exclude_genres:
        df = df[~df["genre_top"].isin(exclude_genres)]
    df = df.dropna(subset=["genre_top"])
    le = LabelEncoder().fit(df["genre_top"])
    out = {}
    for sp in SPLITS:
        sub = df[df["split"] == sp]
        if sub.empty:
            out[sp] = (np.zeros((0, len(cols)), dtype=np.float32), np.array([], dtype=int))
        else:
            out[sp] = (sub[cols].to_numpy(dtype=np.float32), le.transform(sub["genre_top"]))
    return out, le


def gpu_macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int) -> torch.Tensor:
    f1s = torch.zeros(n_classes, device=y_true.device)
    for c in range(n_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().float()
        fp = ((y_pred == c) & (y_true != c)).sum().float()
        fn = ((y_pred != c) & (y_true == c)).sum().float()
        denom = 2 * tp + fp + fn
        f1s[c] = (2 * tp / denom) if denom > 0 else torch.tensor(0.0)
    return f1s.mean()


def gpu_boot_f1(
    y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int,
    n_boot: int = N_BOOT, seed: int = BOOT_SEED,
) -> torch.Tensor:
    n   = len(y_true)
    gen = torch.Generator(device=y_true.device).manual_seed(seed)
    idx = torch.randint(0, n, (n_boot, n), device=y_true.device, generator=gen)
    yt  = y_true[idx]
    yp  = y_pred[idx]
    f1s = torch.zeros(n_boot, device=y_true.device)
    for c in range(n_classes):
        tp    = ((yp == c) & (yt == c)).sum(dim=1).float()
        fp    = ((yp == c) & (yt != c)).sum(dim=1).float()
        fn    = ((yp != c) & (yt == c)).sum(dim=1).float()
        denom = 2 * tp + fp + fn
        f1s  += torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(tp))
    return f1s / n_classes


def probe_one(df: pd.DataFrame, device: torch.device, exclude_genres=None) -> dict:
    cols = emb_cols(df)
    splits, le = split_xy(df, cols, exclude_genres)
    n_classes  = len(le.classes_)
    x_tr, y_tr = splits["training"]
    x_va, y_va = splits["validation"]
    x_te, y_te = splits["test"]

    scaler  = StandardScaler().fit(x_tr)
    x_tr_s  = scaler.transform(x_tr)
    x_va_s  = scaler.transform(x_va)
    x_te_s  = scaler.transform(x_te)

    best_C, best_vf1, best_clf = C_GRID[0], -1.0, None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            clf.fit(x_tr_s, y_tr)
        vf1 = float(sklearn_f1(y_va, clf.predict(x_va_s), average="macro"))
        if vf1 > best_vf1:
            best_vf1, best_C, best_clf = vf1, C, clf

    if best_clf is None:
        raise RuntimeError("C-grid search produced no valid classifier")

    val_pred  = best_clf.predict(x_va_s)
    test_pred = best_clf.predict(x_te_s)

    y_te_t    = torch.from_numpy(y_te).long().to(device)
    y_va_t    = torch.from_numpy(y_va).long().to(device)
    te_pred_t = torch.from_numpy(test_pred).long().to(device)
    va_pred_t = torch.from_numpy(val_pred).long().to(device)

    val_f1  = float(gpu_macro_f1(va_pred_t, y_va_t, n_classes).item())
    test_f1 = float(gpu_macro_f1(te_pred_t, y_te_t, n_classes).item())

    te_boot    = gpu_boot_f1(y_te_t, te_pred_t, n_classes)
    va_boot    = gpu_boot_f1(y_va_t, va_pred_t, n_classes)
    te_boot_np = te_boot.cpu().numpy()
    va_boot_np = va_boot.cpu().numpy()

    return {
        "val_f1":     val_f1,
        "val_ci_lo":  float(np.percentile(va_boot_np, 2.5)),
        "val_ci_hi":  float(np.percentile(va_boot_np, 97.5)),
        "test_f1":    test_f1,
        "test_ci_lo": float(np.percentile(te_boot_np, 2.5)),
        "test_ci_hi": float(np.percentile(te_boot_np, 97.5)),
        "best_C":     best_C,
        "_te_boot":   te_boot_np,
        "_va_boot":   va_boot_np,
        "_y_te":      y_te,
        "_y_va":      y_va,
        "_test_pred": test_pred,
    }


def run_linear_analysis(
    df: pd.DataFrame,
    device: torch.device,
    exclude_genres=None,
    mel_data_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, dict[str, list[dict]]]:
    print("\nLinear probe analysis", flush=True)

    groups: dict[str, dict] = {}
    for method, group in df.groupby("method"):
        base = method_base(method)
        groups.setdefault(base, {})[method] = group

    if mel_data_dir is not None:
        print(f"  Loading raw mel PCA-{MEL_PCA_DIM} baseline from {mel_data_dir}", flush=True)
        mel_df = load_mel_pca_df(mel_data_dir)
        if not mel_df.empty:
            mel_base = mel_df["method"].iloc[0]
            groups[mel_base] = {mel_base: mel_df}
        else:
            print("  WARNING: mel baseline empty, skipping", flush=True)

    rows: list[dict] = []
    seed_results_map: dict[str, list[dict]] = {}

    for base, method_groups in sorted(groups.items()):
        seed_results = []
        for method, group in method_groups.items():
            r = probe_one(group, device, exclude_genres)
            seed_results.append(r)
        seed_results_map[base] = seed_results

        test_arr = np.array([r["test_f1"] for r in seed_results])
        val_arr  = np.array([r["val_f1"]  for r in seed_results])
        n        = len(test_arr)

        if n > 1:
            t = stats.t.ppf(0.975, df=n - 1)
            test_seed_ci = float(t * test_arr.std(ddof=1) / np.sqrt(n))
            val_seed_ci  = float(t * val_arr.std(ddof=1)  / np.sqrt(n))
        else:
            test_seed_ci = val_seed_ci = 0.0

        ref = seed_results[0]
        fam, ratio = parse_method(base)
        row = {
            "method":          base,
            "label":           method_label(base),
            "family":          fam,
            "ratio":           ratio,
            "n_seeds":         n,
            "val_f1_mean":     float(val_arr.mean()),
            "val_f1_seed_ci":  val_seed_ci,
            "val_ci_lo":       float(val_arr.mean()) - val_seed_ci,
            "val_ci_hi":       float(val_arr.mean()) + val_seed_ci,
            "val_boot_lo":     ref["val_ci_lo"],
            "val_boot_hi":     ref["val_ci_hi"],
            "test_f1_mean":    float(test_arr.mean()),
            "test_f1_seed_ci": test_seed_ci,
            "test_ci_lo":      float(test_arr.mean()) - test_seed_ci,
            "test_ci_hi":      float(test_arr.mean()) + test_seed_ci,
            "test_boot_lo":    ref["test_ci_lo"],
            "test_boot_hi":    ref["test_ci_hi"],
            "best_C_mean":     float(np.mean([r["best_C"] for r in seed_results])),
        }
        rows.append(row)
        print(
            f"  {row['label']:30s} n={n}  "
            f"val={row['val_f1_mean']:.4f} [{ref['val_ci_lo']:.4f},{ref['val_ci_hi']:.4f}]  "
            f"test={row['test_f1_mean']:.4f} [{ref['test_ci_lo']:.4f},{ref['test_ci_hi']:.4f}]  "
            f"+/-seed={test_seed_ci:.4f}",
            flush=True,
        )

    return pd.DataFrame(rows), seed_results_map


def load_mel_pca_df(data_dir: Path, pca_dim: int = MEL_PCA_DIM) -> pd.DataFrame:
    frames: dict[str, list[dict]] = {sp: [] for sp in SPLITS}
    for split in SPLITS:
        manifest_path = data_dir / f"manifest_{split}.csv"
        if not manifest_path.exists():
            continue
        manifest = load_manifest(data_dir, split)
        for _, row in manifest.iterrows():
            mel_path = data_dir.parent / str(row["mel_path"])
            if not mel_path.exists():
                continue
            mel = torch.load(mel_path, map_location="cpu", weights_only=True)
            if mel.dim() == 3:
                mel = mel.squeeze(0)
            mean_pool = mel.mean(dim=1).numpy().astype(np.float32)
            std_pool  = mel.std(dim=1).numpy().astype(np.float32)
            feat = np.concatenate([mean_pool, std_pool])
            frames[split].append({
                "track_id":  int(row["track_id"]),
                "genre_top": str(row["genre_top"]),
                "split":     split,
                "_feat":     feat,
            })

    all_rows = [r for sp in SPLITS for r in frames[sp]]
    if not all_rows:
        return pd.DataFrame()

    X_all   = np.stack([r["_feat"] for r in all_rows])
    tr_mask = np.array([r["split"] == "training" for r in all_rows])
    actual_dim = min(pca_dim, X_all.shape[1])
    pca = PCA(n_components=actual_dim, random_state=0)
    pca.fit(X_all[tr_mask])
    X_pca = pca.transform(X_all)

    result_rows = []
    for i, r in enumerate(all_rows):
        result_rows.append({
            "track_id":  r["track_id"],
            "genre_top": r["genre_top"],
            "split":     r["split"],
            "method":    f"raw_mel_pca{actual_dim}",
            **{f"embedding_{j:04d}": float(X_pca[i, j]) for j in range(actual_dim)},
        })
    return pd.DataFrame(result_rows)


def run_comparison_ci(
    seed_results_map: dict[str, list[dict]],
    linear_df: pd.DataFrame,
    ref_families: set[str] = TRAD_FAMILIES,
) -> pd.DataFrame:
    print("\n Comparison CIs vs best traditional baseline ", flush=True)

    trad_rows = linear_df[linear_df["family"].isin(ref_families)]
    if trad_rows.empty:
        print("  No traditional baseline found; skipping comparison CIs", flush=True)
        return pd.DataFrame()

    best_trad      = trad_rows.sort_values("test_f1_mean", ascending=False).iloc[0]
    best_trad_base = best_trad["method"]
    best_trad_f1   = float(best_trad["test_f1_mean"])
    print(f"  Reference baseline: {best_trad_base}  (test_f1={best_trad_f1:.4f})", flush=True)

    ref_results = seed_results_map.get(best_trad_base, [])
    if not ref_results:
        print(f"  WARNING: no seed results for {best_trad_base}; skipping", flush=True)
        return pd.DataFrame()

    ref_boot = np.mean(np.stack([r["_te_boot"] for r in ref_results]), axis=0)
    ref_y    = ref_results[0]["_y_te"]

    rows = []
    for base, seed_results in sorted(seed_results_map.items()):
        fam, ratio = parse_method(base)
        if base == best_trad_base:
            continue

        delta_per_seed: list[np.ndarray] = []
        for r in seed_results:
            if len(r["_y_te"]) != len(ref_y):
                continue
            delta_per_seed.append(r["_te_boot"] - ref_boot)

        if not delta_per_seed:
            continue

        delta_boot  = np.mean(np.stack(delta_per_seed), axis=0)
        delta_point = float(np.mean([r["test_f1"] for r in seed_results])) - best_trad_f1
        delta_lo    = float(np.percentile(delta_boot, 2.5))
        delta_hi    = float(np.percentile(delta_boot, 97.5))
        sig         = not (delta_lo <= 0 <= delta_hi)

        row = {
            "method":      base,
            "label":       method_label(base),
            "family":      fam,
            "ratio":       ratio,
            "ref_method":  best_trad_base,
            "delta_mean":  delta_point,
            "delta_ci_lo": delta_lo,
            "delta_ci_hi": delta_hi,
            "significant": sig,
        }
        rows.append(row)
        print(
            f"  {method_label(base):30s}  delta={delta_point:+.4f}  "
            f"95%CI=[{delta_lo:+.4f},{delta_hi:+.4f}] {'*' if sig else ' '}",
            flush=True,
        )

    return pd.DataFrame(rows)


def load_embeddings(df: pd.DataFrame, method_name: str, sp: str) -> tuple[np.ndarray, np.ndarray]:
    sub  = df[(df["method"] == method_name) & (df["split"] == sp)].dropna(subset=["genre_top"])
    cols = emb_cols(sub)
    return sub[cols].to_numpy(dtype=np.float32), sub["genre_top"].to_numpy()


def load_embeddings_any_seed(df: pd.DataFrame, base_name: str, sp: str) -> tuple[np.ndarray, np.ndarray]:
    sub = df[(df["method"].str.startswith(base_name)) & (df["split"] == sp)].dropna(subset=["genre_top"])
    seed_methods = sub["method"].unique()
    if len(seed_methods) == 0:
        return np.zeros((0, 0), dtype=np.float32), np.array([])
    first = sub[sub["method"] == seed_methods[0]]
    cols  = emb_cols(first)
    return first[cols].to_numpy(dtype=np.float32), first["genre_top"].to_numpy()


def run_perturbation_analysis(df: pd.DataFrame, ref_method: str, split: str = "test") -> pd.DataFrame:
    print(f"\nPerturbation analysis (ref={ref_method}, split={split})", flush=True)

    X_ref_tr, y_ref_tr = load_embeddings_any_seed(df, ref_method, "training")
    X_ref_te, _        = load_embeddings_any_seed(df, ref_method, split)
    if len(X_ref_tr) == 0:
        print(f"  WARNING: reference method {ref_method} not found in parquet", flush=True)
        return pd.DataFrame()

    scaler     = StandardScaler().fit(X_ref_tr)
    X_ref_te_s = scaler.transform(X_ref_te)

    n_classes = len(np.unique(y_ref_tr))
    lda       = LinearDiscriminantAnalysis(n_components=n_classes - 1)
    lda.fit(scaler.transform(X_ref_tr), LabelEncoder().fit_transform(y_ref_tr))
    sem_basis = lda.scalings_[:, :n_classes - 1]
    sem_basis = sem_basis / np.linalg.norm(sem_basis, axis=0, keepdims=True)

    d_full = X_ref_tr.shape[1]
    d_sem  = sem_basis.shape[1]
    d_nuis = d_full - d_sem

    cs_groups: dict[str, list[str]] = {}
    for m in df["method"].unique():
        g, _ = parse_method(m)
        if g in ("dct_biased", "dct_uniform", "srht"):
            cs_groups.setdefault(method_base(m), []).append(m)

    rows = []
    for base, seed_methods in sorted(cs_groups.items()):
        g, ratio = parse_method(base)
        per_seed_rows = []
        for method in seed_methods:
            X, _ = load_embeddings(df, method, split)
            if len(X) == 0:
                continue
            Xs         = scaler.transform(X)
            delta      = Xs - X_ref_te_s
            delta_nuis = delta - delta @ sem_basis @ sem_basis.T
            dir_mags   = np.abs((delta @ sem_basis).mean(axis=0))
            row_dict   = {"nuis_norm": float(np.linalg.norm(delta_nuis, axis=1).mean()) / np.sqrt(d_nuis)}
            for k in range(d_sem):
                row_dict[f"sem_dir_{k}"] = float(dir_mags[k])
            per_seed_rows.append(row_dict)

        if not per_seed_rows:
            continue

        keys  = per_seed_rows[0].keys()
        means = {k: float(np.mean([r[k] for r in per_seed_rows])) for k in keys}
        stds  = {k: float(np.std( [r[k] for r in per_seed_rows], ddof=0)) for k in keys}

        row = {
            "method": base, "label": method_label(base), "family": g, "ratio": ratio,
            "n_seeds": len(per_seed_rows),
            **{k:          means[k] for k in keys},
            **{f"{k}_std": stds[k]  for k in keys},
        }
        rows.append(row)
        dir_str = "  ".join(f"d{k}={means[f'sem_dir_{k}']:.4f}" for k in range(d_sem))
        print(f"  {method_label(base):25s} r={ratio}  nuis={means['nuis_norm']:.4f}  {dir_str}", flush=True)

    return pd.DataFrame(rows).sort_values(["family", "ratio"])


def uniformity_gpu(Z: torch.Tensor, t: float = 2.0, max_n: int = 2048) -> float:
    n = Z.shape[0]
    if n > max_n:
        idx = torch.randperm(n, device=Z.device)[:max_n]
        Z   = Z[idx]
    sq   = torch.cdist(Z, Z).pow(2)
    mask = torch.triu(torch.ones(len(Z), len(Z), device=Z.device, dtype=torch.bool), diagonal=1)
    return float(torch.log(torch.exp(-t * sq[mask]).mean() + EPS).item())


def load_encoder(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc   = ckpt.get("model", {})
    enc  = WaveSTFTEncoder(
        embedding_dim = int(ckpt.get("embedding_dim", 256)),
        base_channels = int(mc.get("base_channels",   16)),
        n_fft         = int(mc.get("n_fft",           1024)),
        hop_length    = int(mc.get("hop_length",      256)),
        n_blocks      = int(mc.get("n_blocks",        3)),
        n_mels        = int(mc.get("n_mels",          128)),
        sample_rate   = int(mc.get("sample_rate",     22050)),
    )
    sd = ckpt.get("state_dict") or ckpt.get("encoder_state_dict") or {}
    if any(k.startswith("encoder.") for k in sd):
        sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    enc.load_state_dict(sd, strict=True)
    enc.eval().to(device)
    return enc


WAVE_AUG_CONFIG = {
    "wave_stretch_scale": [0.8, 1.2], "wave_gain_strength": 0.25,
    "wave_n_masks": 2, "wave_mask_width": 4410, "wave_noise_std": 0.005,
}


def augment_for_method(
    raw: torch.Tensor, base: str, epoch_seed: int, view_idx: int, device: torch.device,
) -> torch.Tensor:
    gen      = torch.Generator(device=device).manual_seed(epoch_seed * 1000 + view_idx)
    g, ratio = parse_method(base)
    ratio_f  = float(ratio) if ratio is not None else 80.0
    if g == "srht":
        return gpu_srht_batch(raw, ratio_f, gen)
    if g in ("dct_biased", "dct_uniform"):
        return gpu_dct_cs_view_batch(raw, ratio_f, gen, uniform=(g == "dct_uniform"))
    policy = "w3"
    if "_w2_" in base or base.endswith("_w2"):
        policy = "w2"
    elif "_w4_" in base or base.endswith("_w4"):
        policy = "w4"
    return gpu_wave_policy_batch(raw, policy, WAVE_AUG_CONFIG, gen)


def run_alignment_analysis(
    df: pd.DataFrame,
    checkpoint_dir: Optional[Path],
    audio_root: Optional[Path],
    dataset_name: str = "fma_small",
    split: str = "test",
    device: Optional[torch.device] = None,
    n_aug_epochs: int = 4,
) -> pd.DataFrame:
    print(f"\nAlignment analysis (split={split})", flush=True)

    has_gpu = (checkpoint_dir is not None and audio_root is not None and checkpoint_dir.is_dir())
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if has_gpu:
        SAMPLE_RATE = 22050
        SEGMENT_SEC = 5.0
        SEG_LEN     = int(SAMPLE_RATE * SEGMENT_SEC)

        manifest  = load_manifest(audio_root, split)
        rng_load  = np.random.default_rng(0)
        waveforms = []
        for _, row in manifest.iterrows():
            npy = (audio_root.parent / Path(row["audio_path"])).with_suffix(".npy")
            if npy.exists():
                y   = np.load(npy)
                off = int(rng_load.uniform(10.0, 25.0) * SAMPLE_RATE)
                seg = y[off : off + SEG_LEN].astype(np.float32)
                if len(seg) < SEG_LEN:
                    seg = np.pad(seg, (0, SEG_LEN - len(seg)))
            else:
                seg = np.zeros(SEG_LEN, dtype=np.float32)
            waveforms.append(seg)
        raw_gpu = torch.from_numpy(np.stack(waveforms)).to(device)
        print(f"  Loaded {raw_gpu.shape[0]} waveforms to {device}", flush=True)
    else:
        raw_gpu = None
        print("  checkpoint_dir / audio_root not provided; encoder-based metrics skipped", flush=True)

    groups: dict[str, list[str]] = {}
    for m in df["method"].unique():
        groups.setdefault(method_base(m), []).append(m)

    rows = []
    for base, methods in sorted(groups.items()):
        g, ratio = parse_method(base)

        unifs_parquet: list[float] = []
        for method in methods:
            sub = df[(df["method"] == method) & (df["split"] == split)].dropna(subset=["genre_top"])
            if sub.empty:
                continue
            cols = emb_cols(sub)
            Z_t  = F.normalize(torch.from_numpy(sub[cols].to_numpy(dtype=np.float32)).to(device), dim=-1)
            unifs_parquet.append(uniformity_gpu(Z_t))

        between_views_list: list[float] = []
        unifs_encoder:      list[float] = []

        if has_gpu and raw_gpu is not None:
            ckpt_paths = (
                sorted(checkpoint_dir.glob(f"{base}_{dataset_name}.pt")) +
                sorted(checkpoint_dir.glob(f"{base}_s*_{dataset_name}.pt"))
            )
            if not ckpt_paths:
                ckpt_paths = [checkpoint_dir / f"{m}_{dataset_name}.pt" for m in methods
                              if (checkpoint_dir / f"{m}_{dataset_name}.pt").exists()]

            for ckpt_path in ckpt_paths:
                if not ckpt_path.exists():
                    continue
                try:
                    encoder = load_encoder(ckpt_path, device)
                except Exception as e:
                    print(f"  WARN: could not load {ckpt_path.name}: {e}", flush=True)
                    continue

                with torch.no_grad():
                    z_clean = F.normalize(encoder(raw_gpu.unsqueeze(1)).float(), dim=-1)

                bv_per_epoch: list[float] = []
                for ep in range(n_aug_epochs):
                    with torch.no_grad():
                        aug1 = augment_for_method(raw_gpu, base, epoch_seed=ep, view_idx=0, device=device)
                        aug2 = augment_for_method(raw_gpu, base, epoch_seed=ep, view_idx=1, device=device)
                        z1   = F.normalize(encoder(aug1.unsqueeze(1)).float(), dim=-1)
                        z2   = F.normalize(encoder(aug2.unsqueeze(1)).float(), dim=-1)
                    bv_per_epoch.append(float((z1 - z2).pow(2).sum(-1).mean().item()))

                between_views_list.append(float(np.mean(bv_per_epoch)))
                unifs_encoder.append(uniformity_gpu(z_clean))

        unif_src  = unifs_encoder if unifs_encoder else unifs_parquet
        unif_mean = float(np.mean(unif_src))          if unif_src          else float("nan")
        unif_std  = float(np.std(unif_src, ddof=0))   if unif_src          else float("nan")
        bv_mean   = float(np.mean(between_views_list)) if between_views_list else float("nan")
        bv_std    = float(np.std(between_views_list, ddof=0)) if between_views_list else float("nan")

        row = {
            "method":             base,
            "label":              method_label(base),
            "family":             g,
            "ratio":              ratio,
            "n_seeds":            len(methods),
            "between_views_mean": bv_mean,
            "between_views_std":  bv_std,
            "uniformity_mean":    unif_mean,
            "uniformity_std":     unif_std,
            "uniformity_source":  "encoder" if unifs_encoder else "parquet",
        }
        rows.append(row)
        print(f"  {method_label(base):25s}  bv={bv_mean:.4f}  unif={unif_mean:.3f}", flush=True)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet",        type=Path, default=Path("data/wave_barlow_fma_small.parquet"))
    parser.add_argument("--output-dir",     type=Path, default=Path("analysis"))
    parser.add_argument("--ref-method",     type=str,  default="supcon_w3_d256_nopop")
    parser.add_argument("--split",          type=str,  default="test", choices=("test", "validation"))
    parser.add_argument("--exclude-genres", nargs="*", default=["Pop"])
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--audio-root",     type=Path, default=None)
    parser.add_argument("--dataset-name",   type=str,  default="fma_small")
    parser.add_argument("--n-aug-epochs",   type=int,  default=4)
    parser.add_argument("--device",         type=str,  default=None)
    parser.add_argument("--analyses",       nargs="+",
                        choices=["linear", "comparison", "perturbation", "alignment"],
                        default=["linear", "comparison", "perturbation", "alignment"])
    args = parser.parse_args()
    run  = set(args.analyses)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print(f"Loading parquet: {args.parquet}", flush=True)
    df = pd.read_parquet(args.parquet)
    print(f"  rows={len(df)}  methods={df['method'].nunique()}  splits={sorted(df['split'].unique())}", flush=True)

    linear_df, seed_results_map = pd.DataFrame(), {}
    if "linear" in run or "comparison" in run:
        linear_df, seed_results_map = run_linear_analysis(
            df,
            device=device,
            exclude_genres=args.exclude_genres,
            mel_data_dir=args.audio_root,
        )
        if "linear" in run:
            out = args.output_dir / "linear_results.csv"
            linear_df.to_csv(out, index=False)
            print(f"Saved: {out}", flush=True)

    if "comparison" in run:
        cmp_df = run_comparison_ci(seed_results_map, linear_df)
        if not cmp_df.empty:
            out = args.output_dir / "comparison_ci.csv"
            cmp_df.to_csv(out, index=False)
            print(f"Saved: {out}", flush=True)

    if "perturbation" in run:
        perturb_df = run_perturbation_analysis(df, ref_method=args.ref_method, split=args.split)
        if not perturb_df.empty:
            out = args.output_dir / "perturbation_results.csv"
            perturb_df.to_csv(out, index=False)
            print(f"Saved: {out}", flush=True)

    if "alignment" in run:
        align_df = run_alignment_analysis(
            df,
            checkpoint_dir = args.checkpoint_dir,
            audio_root     = args.audio_root,
            dataset_name   = args.dataset_name,
            split          = args.split,
            device         = device,
            n_aug_epochs   = args.n_aug_epochs,
        )
        out = args.output_dir / "alignment_analysis.csv"
        align_df.to_csv(out, index=False)
        print(f"Saved: {out}", flush=True)

    print("\nAll analysis complete.", flush=True)


if __name__ == "__main__":
    main()
