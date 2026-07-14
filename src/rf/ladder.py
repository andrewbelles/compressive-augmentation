import json
import math
import subprocess
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from csmath.operators import build_sensing_matrix, mutual_coherence
from csmath.solvers import debias_batch, fista_batch, operator_norm_sq
from rf.corruption import inject_cci, inject_impulsive
from rf.dictionary import Dictionary, get_or_build_dictionary
from rf.frames import load_complex_frames, load_manifest, select_frames
from rf.preprocess.manifests import MOD_CLASSES
from rf.src_classify import src_classify

# Run model for the 6-rung SRC/dictionary ladder. One LadderSpec = one parquet
# shard; the driver is resumable because completed shards are skipped and the
# stage-A -> stage-B barrier is a pure function of shard files on disk.

N              = 1024
RHO_GRID       = [0.0625, 0.125, 0.25, 0.375, 0.5, 0.75]
FAMILIES       = ["gaussian", "demod", "fourier"]
PER_CELL_TEST  = 100
ATOMS_PER_CLASS = 64
DICT_SNR_MIN   = 10
KSVD_SPARSITY  = 8
KSVD_ITERS     = 30
LC_ALPHA       = 1.0
FISTA_ITERS    = 200
LAMBDA_REL     = 0.05      # lam = LAMBDA_REL * ||A^H y||_inf per frame
RECOVERY_THRESH = 0.1      # relative recon error for the success bool (-20 dB NMSE)
DEBIAS_FRAC    = 0.25      # LS-debias support cap as a fraction of m (reconstruct pipeline)
FRAME_BATCH    = 8192
ORBIT_COPIES   = 4
CFO_MAX_EPS    = 0.5
SHIFT_MAX      = 32
SNR_BANDS      = {"high": (10, 30), "mid": (0, 8), "low": (-20, -2)}
BEST_SNR_MIN   = 10

IMP_BURSTS = 2
IMP_LEN    = 16
IMP_AMP    = 6.0
CCI_TONES  = 2
CCI_SIR_DB = 0.0

BPDN_EPS_SLACK = 1.1
BPDN_BISECT    = 6


@dataclass(frozen=True)
class LadderSpec:
    rung: int
    operator_family: str
    rho: float
    pipeline: str               # "reconstruct" (a) | "smashed" (b)
    dict_variant: str           # "v0".."v3"
    error_mode: str = "none"    # none | bpdn | sparse_time | sparse_freq
    corruption: str = "none"    # none | impulsive | cci
    seed: int = 0


@dataclass
class LadderConfig:
    """Runtime knobs; the module constants are the full-scale defaults."""
    per_cell_test:   int = PER_CELL_TEST
    atoms_per_class: int = ATOMS_PER_CLASS
    dict_snr_min:    int = DICT_SNR_MIN
    ksvd_sparsity:   int = KSVD_SPARSITY
    ksvd_iters:      int = KSVD_ITERS
    lc_alpha:        float = LC_ALPHA
    fista_iters:     int = FISTA_ITERS
    lambda_rel:      float = LAMBDA_REL
    frame_batch:     int = FRAME_BATCH
    rho_grid:        list[float] = field(default_factory=lambda: list(RHO_GRID))
    families:        list[str] = field(default_factory=lambda: list(FAMILIES))
    snr_values:      list[int] | None = None
    barrier_timeout_s: float = 4 * 3600.0
    barrier_poll_s:  float = 60.0
    seed:            int = 0


def smoke_config(seed: int = 0) -> LadderConfig:
    """Tiny end-to-end configuration: every rung, barrier, and selection in ~minutes."""
    return LadderConfig(
        per_cell_test=5, atoms_per_class=8, ksvd_iters=3, fista_iters=20,
        rho_grid=[0.25], families=["gaussian"], snr_values=[-10, 0, 10],
        barrier_timeout_s=600.0, barrier_poll_s=2.0, seed=seed,
    )


def _rho_tag(rho: float) -> str:
    return f"{rho:g}".replace(".", "p")


def spec_name(spec: LadderSpec) -> str:
    """Stable shard name, e.g. r3_fourier_rho0p375_smashed_v0_enone_cnone_s0."""
    return (f"r{spec.rung}_{spec.operator_family}_rho{_rho_tag(spec.rho)}_"
            f"{spec.pipeline}_{spec.dict_variant}_e{spec.error_mode}_"
            f"c{spec.corruption}_s{spec.seed}")


def seed_from_name(name: str) -> int:
    """Deterministic (non-randomized) integer seed derived from a spec name."""
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


def build_stage_a(cfg: LadderConfig) -> list[LadderSpec]:
    """Rung 1 (identity control) + rungs 2/3 over the family x rho grid."""
    specs = [LadderSpec(1, "identity", 1.0, "smashed", "v0", seed=cfg.seed)]
    for pipeline, rung in (("reconstruct", 2), ("smashed", 3)):
        for family in cfg.families:
            for rho in cfg.rho_grid:
                specs.append(LadderSpec(rung, family, rho, pipeline, "v0", seed=cfg.seed))
    return specs


def build_stage_b(best: tuple[str, float, str], cfg: LadderConfig) -> list[LadderSpec]:
    """Rungs 4-6 at the best (family, rho, pipeline) from stage A."""
    family, rho, pipeline = best
    specs = [LadderSpec(4, family, rho, pipeline, "v1", seed=cfg.seed)]
    for corruption, modes in (("impulsive", ["none", "sparse_time", "bpdn"]),
                              ("cci",       ["none", "sparse_freq", "bpdn"])):
        for mode in modes:
            specs.append(LadderSpec(5, family, rho, pipeline, "v0",
                                    error_mode=mode, corruption=corruption, seed=cfg.seed))
    specs.append(LadderSpec(6, family, rho, pipeline, "v2", seed=cfg.seed))
    specs.append(LadderSpec(6, family, rho, pipeline, "v3", seed=cfg.seed))
    return specs


def inverse_dft_matrix(n: int, device: torch.device) -> torch.Tensor:
    """Unitary inverse-DFT synthesis matrix Psi (x = Psi s, s the frequency code)."""
    k = torch.arange(n, device=device, dtype=torch.float32)
    return (torch.exp((2j * math.pi / n) * torch.outer(k, k).to(torch.complex64))
            / math.sqrt(n)).to(torch.complex64)


def dct_matrix(n: int, device: torch.device) -> torch.Tensor:
    """Orthonormal DCT-II analysis matrix as a real matrix (applied to complex vectors
    it is still a single C-linear map)."""
    i = torch.arange(n, device=device, dtype=torch.float32)
    C = torch.cos(math.pi / n * torch.outer(i, i + 0.5)) * math.sqrt(2.0 / n)
    C[0] /= math.sqrt(2.0)
    return C


@dataclass
class LadderContext:
    """Everything a run_spec call needs: test frames on device, dictionaries, paths."""
    cfg:          LadderConfig
    device:       torch.device
    test_x:       torch.Tensor      # [B, N] complex64 unit-norm
    test_mods:    np.ndarray        # [B] str
    test_snrs:    np.ndarray        # [B] int
    test_frame_idx: np.ndarray      # [B] int
    psi:          torch.Tensor      # [N, N] unitary inverse DFT
    hdf5_path:    Path
    manifest_dir: Path
    dicts_dir:    Path
    results_dir:  Path
    dicts:        dict[str, Dictionary] = field(default_factory=dict)

    def get_dict(self, variant: str) -> Dictionary:
        if variant not in self.dicts:
            self.dicts[variant] = get_or_build_dictionary(
                variant, self.dicts_dir, self.hdf5_path, self.manifest_dir,
                atoms_per_class=self.cfg.atoms_per_class, snr_min=self.cfg.dict_snr_min,
                seed=self.cfg.seed, device=self.device,
                n_orbit=ORBIT_COPIES, max_eps=CFO_MAX_EPS, max_shift=SHIFT_MAX,
                ksvd_sparsity=self.cfg.ksvd_sparsity, ksvd_iters=self.cfg.ksvd_iters,
                lc_alpha=self.cfg.lc_alpha,
            )
        return self.dicts[variant]


def build_context(
    cfg: LadderConfig,
    hdf5_path: Path,
    manifest_dir: Path,
    dicts_dir: Path,
    results_dir: Path,
    device: torch.device,
) -> LadderContext:
    """Load the deterministic test subsample onto the device and prepare paths."""
    manifest = load_manifest(manifest_dir, "test")
    sel = select_frames(manifest, cfg.per_cell_test, cfg.seed, snr_values=cfg.snr_values)
    x = load_complex_frames(hdf5_path, sel["frame_idx"].to_numpy(), device, normalize=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    return LadderContext(
        cfg=cfg, device=device, test_x=x,
        test_mods=sel["mod"].to_numpy(), test_snrs=sel["snr"].to_numpy(),
        test_frame_idx=sel["frame_idx"].to_numpy(),
        psi=inverse_dft_matrix(N, device),
        hdf5_path=Path(hdf5_path), manifest_dir=Path(manifest_dir),
        dicts_dir=Path(dicts_dir), results_dir=Path(results_dir),
    )


def energy_compaction_table(ctx: LadderContext, top_k: int = 32, snr_min: int = 10) -> pd.DataFrame:
    """Per-class top-k energy fraction under DFT vs DCT on a high-SNR sample.

    Informational Psi check printed by the driver; the pipeline itself uses the
    DFT synthesis basis.
    """
    F   = ctx.psi.conj().T                       # unitary forward DFT
    C   = dct_matrix(N, ctx.device).to(torch.complex64)
    rows = []
    for mod in MOD_CLASSES:
        mask = (ctx.test_mods == mod) & (ctx.test_snrs >= snr_min)
        if not mask.any():
            continue
        x = ctx.test_x[torch.from_numpy(np.flatnonzero(mask)).to(ctx.device)]
        for name, T in (("dft", F), ("dct", C)):
            e = (x @ T.T).abs().pow(2)
            frac = (e.topk(top_k, dim=1).values.sum(dim=1) / e.sum(dim=1)).mean().item()
            rows.append({"mod": mod, "basis": name, "topk_energy_frac": frac})
    return pd.DataFrame(rows).pivot(index="mod", columns="basis", values="topk_energy_frac")


def _bpdn_eps(y: torch.Tensor, snrs: torch.Tensor) -> torch.Tensor:
    """Per-frame noise-ball radius from the SNR label: eps = ||y|| sqrt(f) * slack."""
    f = 1.0 / (1.0 + torch.pow(10.0, snrs.float() / 10.0))
    return y.norm(dim=1) * f.sqrt() * BPDN_EPS_SLACK


def _error_dictionary(mode: str, phi: torch.Tensor, psi: torch.Tensor,
                      sensed: bool, device: torch.device) -> torch.Tensor | None:
    """Error dictionary for sparse_time/sparse_freq, in the raw or sensed domain."""
    if mode == "sparse_time":
        E = torch.eye(N, dtype=torch.complex64, device=device) if not sensed else phi.clone()
    elif mode == "sparse_freq":
        E = psi if not sensed else phi @ psi
    else:
        return None
    return E / E.norm(dim=0, keepdim=True).clamp_min(1e-12)


def run_spec(spec: LadderSpec, ctx: LadderContext) -> tuple[pd.DataFrame, dict]:
    """Execute one ladder spec over the full test set; returns (per-frame df, meta)."""
    cfg    = ctx.cfg
    name   = spec_name(spec)
    t0     = time.time()
    device = ctx.device
    op_seed = seed_from_name(name)

    phi = build_sensing_matrix(spec.operator_family, spec.rho, N, op_seed, device)
    m   = phi.shape[0]

    x = ctx.test_x
    if spec.corruption == "impulsive":
        gen = torch.Generator(device=device).manual_seed(op_seed + 1)
        x = inject_impulsive(x, IMP_BURSTS, IMP_LEN, IMP_AMP, gen)
    elif spec.corruption == "cci":
        gen = torch.Generator(device=device).manual_seed(op_seed + 1)
        x = inject_cci(x, CCI_TONES, CCI_SIR_DB, gen)

    d = ctx.get_dict(spec.dict_variant)
    snrs = torch.from_numpy(ctx.test_snrs.astype(np.float32)).to(device)

    if spec.pipeline == "reconstruct":
        A_cls = d.atoms
        E     = _error_dictionary(spec.error_mode, phi, ctx.psi, sensed=False, device=device)
        A_rec = phi @ ctx.psi
        L_rec    = operator_norm_sq(A_rec)
        gram_rec = A_rec.conj().T @ A_rec
    else:
        A_cls = phi @ d.atoms
        A_cls = A_cls / A_cls.norm(dim=0, keepdim=True).clamp_min(1e-12)
        E     = _error_dictionary(spec.error_mode, phi, ctx.psi, sensed=True, device=device)

    solver = "bpdn" if spec.error_mode == "bpdn" else "fista"
    preds, margins, res_norms, rel_errs = [], [], [], []
    for lo in range(0, x.shape[0], cfg.frame_batch):
        xb = x[lo:lo + cfg.frame_batch]
        yb = xb @ phi.T
        eps = _bpdn_eps(yb, snrs[lo:lo + cfg.frame_batch]) if solver == "bpdn" else None

        if spec.pipeline == "reconstruct":
            AhY = yb @ A_rec.conj()
            lam = cfg.lambda_rel * AhY.abs().amax(dim=1)
            s, _ = fista_batch(A_rec, yb, lam, cfg.fista_iters,
                               L=L_rec, gram=gram_rec, AhY=AhY)
            s = debias_batch(A_rec, yb, s, max(1, int(DEBIAS_FRAC * m)))
            x_hat = s @ ctx.psi.T
            rel_errs.append(((x_hat - xb).norm(dim=1) / xb.norm(dim=1).clamp_min(1e-12)).cpu())
            z = x_hat / x_hat.norm(dim=1, keepdim=True).clamp_min(1e-12)
            if eps is not None:
                eps = _bpdn_eps(z, snrs[lo:lo + cfg.frame_batch])
        else:
            z = yb

        result = src_classify(A_cls, d.atom_labels, z, solver=solver,
                              lam_rel=cfg.lambda_rel, n_iter=cfg.fista_iters,
                              eps=eps, E=E, n_classes=len(MOD_CLASSES))
        preds.append(result.pred.cpu())
        margins.append(result.margin.cpu())
        res_norms.append(result.res_norm.cpu())

    pred = torch.cat(preds).numpy()
    rel  = torch.cat(rel_errs).numpy() if rel_errs else np.full(x.shape[0], np.nan, np.float32)

    df = pd.DataFrame({
        "rung":            spec.rung,
        "operator_family": spec.operator_family,
        "rho":             np.float32(spec.rho),
        "pipeline":        spec.pipeline,
        "dict_variant":    spec.dict_variant,
        "error_mode":      spec.error_mode,
        "corruption":      spec.corruption,
        "seed":            spec.seed,
        "frame_idx":       ctx.test_frame_idx,
        "mod_true":        ctx.test_mods,
        "mod_pred":        np.array(MOD_CLASSES, dtype=object)[pred],
        "snr":             ctx.test_snrs,
        "residual_margin": torch.cat(margins).numpy().astype(np.float32),
        "recovered":       (rel <= RECOVERY_THRESH) if rel_errs else False,
        "recon_rel_err":   rel.astype(np.float32),
        "solver_res_norm": torch.cat(res_norms).numpy().astype(np.float32),
    })

    meta = {
        **asdict(spec),
        "m": int(m), "n": N, "psi": "dft",
        "lambda_policy": f"lam = {cfg.lambda_rel} * ||A^H y||_inf per frame",
        "debias": f"LS refit on top-{DEBIAS_FRAC} * m support (reconstruct pipeline)",
        "fista_iters": cfg.fista_iters,
        "dict_cache_key": f"dict_{spec.dict_variant}_k{cfg.atoms_per_class * len(MOD_CLASSES)}"
                          f"_snr{cfg.dict_snr_min}_s{cfg.seed}",
        "mu_phid": mutual_coherence(A_cls if spec.pipeline == "smashed" else phi @ d.atoms),
        "mu_phid_phie": mutual_coherence(phi @ d.atoms, E if spec.pipeline == "smashed"
                                         else (phi @ E if E is not None else None))
                        if E is not None else None,
        "wallclock_s": time.time() - t0,
        "git_sha": _git_sha(),
        "accuracy": float((df["mod_true"] == df["mod_pred"]).mean()),
    }
    return df, meta


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def save_shard(df: pd.DataFrame, meta: dict, results_dir: Path, name: str) -> Path:
    """Atomically write {name}.parquet + {name}.meta.json (tmp + rename)."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tmp = results_dir / f"{name}.parquet.tmp"
    df.to_parquet(tmp, index=False)
    path = results_dir / f"{name}.parquet"
    tmp.rename(path)
    meta_tmp = results_dir / f"{name}.meta.json.tmp"
    meta_tmp.write_text(json.dumps(meta, indent=2))
    meta_tmp.rename(results_dir / f"{name}.meta.json")
    return path


def shard_exists(results_dir: Path, name: str) -> bool:
    return (Path(results_dir) / f"{name}.parquet").exists()


def wait_for_shards(
    names: list[str], results_dir: Path, timeout_s: float, poll_s: float = 60.0
) -> None:
    """Block until every named shard exists; raise TimeoutError with the missing list.

    Recovery from a crashed peer is resubmission -- completed shards are skipped,
    so the barrier heals on the next run.
    """
    deadline = time.time() + timeout_s
    while True:
        missing = [n for n in names if not shard_exists(results_dir, n)]
        if not missing:
            return
        if time.time() >= deadline:
            raise TimeoutError(
                f"barrier timed out after {timeout_s:.0f}s; missing shards: {missing}")
        time.sleep(poll_s)
