#!/usr/bin/env python3
#
# rpca.py  Andrew Belles  May 2026
#
# Robust PCA decomposition of mel spectrograms: X = L + S
# Writes {track_id}.lr.pt (low-rank component) alongside each {track_id}.pt.
# Uses PyTorch SVD so the ALM inner loop runs on GPU when available.
#

import argparse
import sys
from pathlib import Path

import torch


DEFAULT_MEL_DIR = Path("preprocess/data/fma_small_mel")
DEFAULT_LAM_MULT = 0.10


def report(message: str) -> None:
    print(message, flush=True)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose mel spectrograms into low-rank + sparse components via RPCA."
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        default=DEFAULT_MEL_DIR,
        help=f"Mel spectrogram directory. Defaults to {DEFAULT_MEL_DIR}.",
    )
    parser.add_argument(
        "--lam-mult",
        type=float,
        default=DEFAULT_LAM_MULT,
        help=f"Lambda multiplier applied to 1/sqrt(max(rows,cols)). Default {DEFAULT_LAM_MULT}.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-7,
        help="Convergence tolerance for ALM. Default 1e-7.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Max ALM iterations. Default 500.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for SVD computation: auto, cpu, cuda. Default auto.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip tracks that already have a .lr.pt file (default: True).",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Recompute even if .lr.pt already exists.",
    )
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


@torch.no_grad()
def rpca_alm_torch(
    M: torch.Tensor,
    lam_mult: float = DEFAULT_LAM_MULT,
    tol: float = 1e-7,
    max_iter: int = 500,
) -> torch.Tensor:
    m, n = M.shape
    lam = lam_mult / (max(m, n) ** 0.5)

    norm_two = max(torch.linalg.matrix_norm(M, ord=2).item(), 1e-10)
    norm_inf = (M.abs().max() / lam).item()
    dual_norm = max(norm_two, norm_inf, 1e-10)

    Y = M / dual_norm
    mu = 1.25 / norm_two
    mu_bar = mu * 1e7
    rho = 1.5
    L = torch.zeros_like(M)
    S = torch.zeros_like(M)
    frob_M = max(torch.linalg.matrix_norm(M, ord="fro").item(), 1e-10)

    for _ in range(max_iter):
        U, sv, Vh = torch.linalg.svd(M - S + Y / mu, full_matrices=False)
        sv = (sv - 1.0 / mu).clamp_min_(0.0)
        L = (U * sv) @ Vh

        tmp = M - L + Y / mu
        S = tmp.sign() * (tmp.abs() - lam / mu).clamp_min_(0.0)

        residual = M - L - S
        Y = Y + mu * residual
        mu = min(mu * rho, mu_bar)

        if torch.linalg.matrix_norm(residual, ord="fro").item() / frob_M < tol:
            break

    return L.clamp_(0.0, 1.0).float()


def process_directory(
    data_dir: Path,
    device: torch.device,
    lam_mult: float,
    tol: float,
    max_iter: int,
    skip_existing: bool,
) -> tuple[int, int, int]:
    mel_paths = sorted(p for p in data_dir.rglob("*.pt") if p.stem.isdigit())
    total = len(mel_paths)
    processed = skipped = errors = 0

    report(f"START data_dir={data_dir} total={total} device={device} lam_mult={lam_mult}")

    for i, mel_path in enumerate(mel_paths, 1):
        lr_path = mel_path.with_suffix(".lr.pt")
        if skip_existing and lr_path.exists():
            skipped += 1
            continue
        try:
            M = torch.load(mel_path, map_location="cpu", weights_only=True).float().to(device)
            L = rpca_alm_torch(M, lam_mult=lam_mult, tol=tol, max_iter=max_iter)
            torch.save(L.cpu(), lr_path)
            processed += 1
        except Exception as exc:
            log(f"[rpca] error {mel_path}: {exc}")
            errors += 1

        if i % 200 == 0 or i == total:
            log(f"[rpca] {i}/{total}  processed={processed} skipped={skipped} errors={errors}")

    return processed, skipped, errors


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(f"directory not found: {data_dir}")

    device = resolve_device(args.device)

    processed, skipped, errors = process_directory(
        data_dir,
        device=device,
        lam_mult=args.lam_mult,
        tol=args.tol,
        max_iter=args.max_iter,
        skip_existing=args.skip_existing,
    )
    report(f"DONE processed={processed} skipped={skipped} errors={errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
