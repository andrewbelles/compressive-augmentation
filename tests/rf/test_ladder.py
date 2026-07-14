import numpy as np
import pandas as pd
import pytest
import torch

from rf.analysis import load_ladder_results, select_best_config
from rf.ladder import (
    LadderSpec,
    build_context,
    build_stage_a,
    build_stage_b,
    run_spec,
    save_shard,
    shard_exists,
    smoke_config,
    spec_name,
    seed_from_name,
    wait_for_shards,
)
from rf.preprocess.manifests import write_manifests
from tests.rf.test_frames import _make_fake_hdf5

SCHEMA = {
    "rung", "operator_family", "rho", "pipeline", "dict_variant", "error_mode",
    "corruption", "seed", "frame_idx", "mod_true", "mod_pred", "snr",
    "residual_margin", "recovered", "recon_rel_err", "solver_res_norm",
}


@pytest.fixture(scope="module")
def data_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("ladder")
    hdf5 = _make_fake_hdf5(root / "fake.hdf5")
    write_manifests(hdf5, root / "manifests")
    return hdf5, root / "manifests", root


@pytest.fixture(scope="module")
def ctx(data_dirs):
    hdf5, manifests, root = data_dirs
    cfg = smoke_config()
    cfg.per_cell_test = 2
    cfg.fista_iters = 10
    cfg.atoms_per_class = 4
    return build_context(cfg, hdf5, manifests, root / "dicts", root / "results",
                         torch.device("cpu"))


class TestSpecNames:
    def test_stage_a_names_unique(self):
        cfg = smoke_config()
        names = [spec_name(s) for s in build_stage_a(cfg)]
        assert len(names) == len(set(names))

    def test_name_stable(self):
        spec = LadderSpec(3, "fourier", 0.375, "smashed", "v0")
        assert spec_name(spec) == "r3_fourier_rho0p375_smashed_v0_enone_cnone_s0"

    def test_seed_from_name_stable(self):
        assert seed_from_name("abc") == seed_from_name("abc")
        assert seed_from_name("abc") != seed_from_name("abd")

    def test_stage_a_structure(self):
        from rf.ladder import LadderConfig
        cfg = LadderConfig()
        specs = build_stage_a(cfg)
        assert specs[0] == LadderSpec(1, "identity", 1.0, "smashed", "v0")
        r2 = [s for s in specs if s.rung == 2]
        r3 = [s for s in specs if s.rung == 3]
        assert len(specs) == 1 + 2 * len(cfg.families) * len(cfg.rho_grid)
        assert len(r2) == len(r3)
        for a, b in zip(r2, r3):
            assert (a.operator_family, a.rho) == (b.operator_family, b.rho)
            assert (a.pipeline, b.pipeline) == ("reconstruct", "smashed")

    def test_stage_b_structure(self):
        cfg = smoke_config()
        specs = build_stage_b(("gaussian", 0.25, "smashed"), cfg)
        assert len(specs) == 9
        assert specs[0].rung == 4 and specs[0].dict_variant == "v1"
        r5 = [s for s in specs if s.rung == 5]
        assert len(r5) == 6
        assert {(s.corruption, s.error_mode) for s in r5} == {
            ("impulsive", "none"), ("impulsive", "sparse_time"), ("impulsive", "bpdn"),
            ("cci", "none"), ("cci", "sparse_freq"), ("cci", "bpdn"),
        }
        assert [s.dict_variant for s in specs if s.rung == 6] == ["v2", "v3"]


class TestSelectBestConfig:
    def _fake_results(self):
        rows = []
        for family, rho, pipeline, acc in [
            ("gaussian", 0.25, "smashed", 0.9),
            ("gaussian", 0.25, "reconstruct", 0.5),
            ("fourier",  0.5,  "smashed", 0.7),
        ]:
            for i in range(10):
                rows.append({
                    "operator_family": family, "rho": rho, "pipeline": pipeline,
                    "snr": 10, "mod_true": "BPSK",
                    "mod_pred": "BPSK" if i < acc * 10 else "QPSK",
                })
            rows.append({**rows[-1], "snr": -10, "mod_pred": "QPSK"})
        return pd.DataFrame(rows)

    def test_picks_highest_accuracy_above_snr_min(self):
        best = select_best_config(self._fake_results(), snr_min=10)
        assert best == ("gaussian", 0.25, "smashed")

    def test_identity_excluded(self):
        df = self._fake_results()
        ident = df.iloc[:5].copy()
        ident["operator_family"] = "identity"
        ident["mod_pred"] = ident["mod_true"]
        best = select_best_config(pd.concat([df, ident]), snr_min=10)
        assert best[0] != "identity"


class TestRunSpec:
    def test_smashed_schema(self, ctx):
        df, meta = run_spec(LadderSpec(3, "gaussian", 0.25, "smashed", "v0"), ctx)
        assert set(df.columns) == SCHEMA
        assert len(df) == ctx.test_x.shape[0]
        assert df["recon_rel_err"].isna().all()
        assert not df["recovered"].any()
        assert set(df["mod_pred"]).issubset(set(df["mod_true"]))
        assert meta["m"] == 256 and meta["accuracy"] >= 0.0

    def test_reconstruct_schema_and_identity_recovery(self, ctx):
        df, _ = run_spec(LadderSpec(2, "identity", 1.0, "reconstruct", "v0"), ctx)
        assert set(df.columns) == SCHEMA
        assert np.isfinite(df["recon_rel_err"]).all()
        assert (df["recon_rel_err"] < 0.1).all()      # frequency-sparse fake frames
        assert df["recovered"].all()

    def test_shard_roundtrip_and_barrier(self, ctx, tmp_path):
        spec = LadderSpec(3, "gaussian", 0.25, "smashed", "v0")
        df, meta = run_spec(spec, ctx)
        name = spec_name(spec)
        save_shard(df, meta, tmp_path, name)
        assert shard_exists(tmp_path, name)
        wait_for_shards([name], tmp_path, timeout_s=1.0, poll_s=0.1)
        loaded = load_ladder_results(tmp_path)
        assert len(loaded) == len(df)
        assert (loaded["spec_name"] == name).all()

    def test_barrier_timeout(self, tmp_path):
        with pytest.raises(TimeoutError):
            wait_for_shards(["missing_shard"], tmp_path, timeout_s=0.2, poll_s=0.05)
