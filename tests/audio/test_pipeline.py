import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from audio.encoder import AudioBarlowModel, AudioSTFTEncoder
from common.dataset import BaseBarlowDataset
from csmath.losses import barlow_twins_loss
from train import (
    BARLOW_LAMBDA,
    EPOCHS,
    PEAK_LR,
    SUPCON_TEMP,
    WAVE_AUGMENT,
    WARMUP_EPOCHS,
    RunSpec,
    build_run_list,
    cache_raw_on_gpu,
    cache_supcon_on_gpu,
    cosine_lr,
    run_barlow_epoch,
    run_supcon_epoch,
    source_name,
)

B         = 4
T         = 22050
SR        = 22050
N_FFT     = 1024
HOP       = 256
N_MELS    = 128
N_BLOCKS  = 2
BASE_CH   = 4
EMBED_DIM = 32
PROJ_H    = 64
PROJ_DIM  = 48
SUPCON_P  = 16


def _barlow_model(device):
    return AudioBarlowModel(
        embedding_dim=EMBED_DIM,
        base_channels=BASE_CH,
        projection_hidden_dim=PROJ_H,
        projection_dim=PROJ_DIM,
        n_fft=N_FFT,
        hop_length=HOP,
        n_blocks=N_BLOCKS,
        n_mels=N_MELS,
        sample_rate=SR,
    ).to(device)


def _encoder(device):
    return AudioSTFTEncoder(
        embedding_dim=EMBED_DIM,
        base_channels=BASE_CH,
        n_fft=N_FFT,
        hop_length=HOP,
        n_blocks=N_BLOCKS,
        n_mels=N_MELS,
        sample_rate=SR,
    ).to(device)


def _proj(device):
    return nn.Sequential(
        nn.Linear(EMBED_DIM, EMBED_DIM, bias=False),
        nn.BatchNorm1d(EMBED_DIM),
        nn.ReLU(inplace=True),
        nn.Linear(EMBED_DIM, SUPCON_P),
    ).to(device)


def _wave(device):
    return torch.randn(B, 1, T, device=device)


def _raw(device):
    return torch.randn(B, T, device=device)


def _labels(device):
    return torch.tensor([0, 0, 1, 1], device=device)


def _optimizer(params):
    return torch.optim.AdamW(params, lr=1e-3)


def _scaler(device):
    return torch.amp.GradScaler(device.type, enabled=device.type == "cuda")


class TestAudioSTFTEncoder:
    def test_to_mel_shape(self, device):
        enc    = _encoder(device).eval()
        mel    = enc.to_mel(_wave(device))
        frames = T // HOP + 1
        assert mel.shape == (B, 1, N_MELS, frames)

    def test_forward_shape(self, device):
        enc = _encoder(device).eval()
        out = enc(_wave(device))
        assert out.shape == (B, EMBED_DIM)

    def test_no_nan(self, device):
        enc = _encoder(device).eval()
        out = enc(_wave(device))
        assert torch.isfinite(out).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
    def test_cpu_cuda_parity(self):
        x   = _wave(torch.device("cpu"))
        cpu = _encoder(torch.device("cpu"))
        gpu = _encoder(torch.device("cuda"))
        gpu.load_state_dict(cpu.state_dict())
        with torch.no_grad():
            out_cpu = cpu(x)
            out_gpu = gpu(x.cuda()).cpu()
        assert torch.allclose(out_cpu, out_gpu, atol=1e-3)


class TestAudioBarlowModel:
    def test_forward_shapes(self, device):
        model         = _barlow_model(device).eval()
        x             = _wave(device)
        h1, h2, z1, z2 = model(x, x.clone())
        assert h1.shape == h2.shape == (B, EMBED_DIM)
        assert z1.shape == z2.shape == (B, PROJ_DIM)

    def test_barlow_loss_computable(self, device):
        model        = _barlow_model(device).eval()
        x            = _wave(device)
        _, _, z1, z2 = model(x, x.clone())
        loss, _, _   = barlow_twins_loss(z1, z2, lambd=5e-5)
        assert torch.isfinite(loss)

    def test_gradient_flows(self, device):
        model        = _barlow_model(device).train()
        x            = _wave(device)
        _, _, z1, z2 = model(x, x.clone())
        loss, _, _   = barlow_twins_loss(z1, z2, lambd=5e-5)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)


class TestBarlowEpochCS:
    def test_train_step_cs_biased(self, device):
        model  = _barlow_model(device).train()
        opt    = _optimizer(model.parameters())
        scaler = _scaler(device)
        spec   = RunSpec("cs_biased", seed=0, ratio=20)
        result = run_barlow_epoch(model, _raw(device), opt, scaler, device, spec, epoch=0, train=True)
        assert torch.isfinite(torch.tensor(result["loss"]))
        assert torch.isfinite(torch.tensor(result["on_diag"]))
        assert torch.isfinite(torch.tensor(result["off_diag"]))

    def test_train_step_cs_uniform(self, device):
        model  = _barlow_model(device).train()
        opt    = _optimizer(model.parameters())
        scaler = _scaler(device)
        spec   = RunSpec("cs_uniform", seed=0, ratio=20, uniform=True)
        result = run_barlow_epoch(model, _raw(device), opt, scaler, device, spec, epoch=0, train=True)
        assert torch.isfinite(torch.tensor(result["loss"]))

    def test_train_step_cs_srht(self, device):
        model  = _barlow_model(device).train()
        opt    = _optimizer(model.parameters())
        scaler = _scaler(device)
        spec   = RunSpec("cs_srht", seed=0, ratio=20, srht=True)
        result = run_barlow_epoch(model, _raw(device), opt, scaler, device, spec, epoch=0, train=True)
        assert torch.isfinite(torch.tensor(result["loss"]))

    def test_val_step_no_grad(self, device):
        model  = _barlow_model(device)
        opt    = _optimizer(model.parameters())
        scaler = _scaler(device)
        spec   = RunSpec("cs_biased", seed=0, ratio=20)
        run_barlow_epoch(model, _raw(device), opt, scaler, device, spec, epoch=0, train=False)
        for p in model.parameters():
            assert p.grad is None

    def test_gradient_flows(self, device):
        model  = _barlow_model(device).train()
        opt    = _optimizer(model.parameters())
        scaler = _scaler(device)
        spec   = RunSpec("cs_biased", seed=0, ratio=20)
        run_barlow_epoch(model, _raw(device), opt, scaler, device, spec, epoch=0, train=True)
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)


class TestBarlowEpochTraditional:
    def test_train_step_w2(self, device):
        model  = _barlow_model(device).train()
        opt    = _optimizer(model.parameters())
        scaler = _scaler(device)
        spec   = RunSpec("traditional", seed=0, policy="w2")
        result = run_barlow_epoch(model, _raw(device), opt, scaler, device, spec, epoch=0, train=True)
        assert torch.isfinite(torch.tensor(result["loss"]))

    def test_train_step_w3(self, device):
        model  = _barlow_model(device).train()
        opt    = _optimizer(model.parameters())
        scaler = _scaler(device)
        spec   = RunSpec("traditional", seed=0, policy="w3")
        result = run_barlow_epoch(model, _raw(device), opt, scaler, device, spec, epoch=0, train=True)
        assert torch.isfinite(torch.tensor(result["loss"]))

    def test_train_step_w4(self, device):
        model  = _barlow_model(device).train()
        opt    = _optimizer(model.parameters())
        scaler = _scaler(device)
        spec   = RunSpec("traditional", seed=0, policy="w4")
        result = run_barlow_epoch(model, _raw(device), opt, scaler, device, spec, epoch=0, train=True)
        assert torch.isfinite(torch.tensor(result["loss"]))


class TestSupConEpoch:
    def test_train_step(self, device):
        enc    = _encoder(device).train()
        proj   = _proj(device).train()
        opt    = _optimizer(list(enc.parameters()) + list(proj.parameters()))
        scaler = _scaler(device)
        loss   = run_supcon_epoch(enc, proj, _raw(device), _labels(device), opt, scaler, device, epoch=0, train=True)
        assert torch.isfinite(torch.tensor(loss))
        assert loss >= 0.0

    def test_val_step_no_grad(self, device):
        enc    = _encoder(device)
        proj   = _proj(device)
        opt    = _optimizer(list(enc.parameters()) + list(proj.parameters()))
        scaler = _scaler(device)
        run_supcon_epoch(enc, proj, _raw(device), _labels(device), opt, scaler, device, epoch=0, train=False)
        for p in list(enc.parameters()) + list(proj.parameters()):
            assert p.grad is None

    def test_gradient_flows(self, device):
        enc    = _encoder(device).train()
        proj   = _proj(device).train()
        params = list(enc.parameters()) + list(proj.parameters())
        opt    = _optimizer(params)
        scaler = _scaler(device)
        run_supcon_epoch(enc, proj, _raw(device), _labels(device), opt, scaler, device, epoch=0, train=True)
        grads = [p.grad for p in params if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)


class TestSweep:
    def test_run_list_total_count(self):
        runs = build_run_list()
        from train import SEEDS, SEEDS_TRAD, RATIOS, POLICIES
        expected = (
            len(SEEDS) * len(RATIOS) * 3   # cs_biased, cs_uniform, cs_srht
            + len(SEEDS_TRAD) * len(POLICIES)  # traditional
            + len(SEEDS_TRAD)                   # supcon
        )
        assert len(runs) == expected

    def test_half_split_exhaustive(self):
        runs  = build_run_list()
        half0 = [r for i, r in enumerate(runs) if i % 2 == 0]
        half1 = [r for i, r in enumerate(runs) if i % 2 == 1]
        assert len(half0) + len(half1) == len(runs)
        assert set(range(len(runs))) == set(
            [i for i, _ in enumerate(runs) if i % 2 == 0] +
            [i for i, _ in enumerate(runs) if i % 2 == 1]
        )

    def test_source_name_unique(self):
        runs   = build_run_list()
        names  = [source_name(r) for r in runs]
        assert len(names) == len(set(names))

    def test_source_name_contains_kind_info(self):
        assert "srht"    in source_name(RunSpec("cs_srht",    seed=0, ratio=20, srht=True))
        assert "uniform" in source_name(RunSpec("cs_uniform", seed=0, ratio=20, uniform=True))
        assert "w3"      in source_name(RunSpec("traditional", seed=0, policy="w3"))
        assert "supcon"  in source_name(RunSpec("supcon",     seed=0))


class TestLRSchedule:
    def test_warmup_monotone(self):
        lrs = [cosine_lr(e) for e in range(WARMUP_EPOCHS)]
        assert all(lrs[i] < lrs[i + 1] for i in range(len(lrs) - 1))

    def test_cosine_decay_monotone(self):
        lrs = [cosine_lr(e) for e in range(WARMUP_EPOCHS, EPOCHS)]
        assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))

    def test_boundary_finite(self):
        for epoch in [0, WARMUP_EPOCHS - 1, WARMUP_EPOCHS, EPOCHS - 1]:
            lr = cosine_lr(epoch)
            assert lr > 0.0 and lr <= PEAK_LR


class _FakeDataset(BaseBarlowDataset):
    """In-memory fake dataset for testing cache functions without disk I/O."""

    def __init__(self, n, t, labels=None):
        self._data   = [np.random.randn(t).astype(np.float32) for _ in range(n)]
        self._labels = labels
        self._raw_only = True
        self.is_train  = False

    def __len__(self):
        return len(self._data)

    def load_sample(self, index):
        return self._data[index]

    def make_views(self, index, rng1, rng2):
        y = torch.from_numpy(self._data[index])
        return y, y.clone()

    def __getitem__(self, index):
        if self._labels is not None:
            return torch.from_numpy(self._data[index]), self._labels[index]
        return (torch.from_numpy(self._data[index]),)


class TestCacheGPU:
    def test_cache_raw_shape(self):
        n, t = 6, T
        ds   = _FakeDataset(n, t)
        raw  = cache_raw_on_gpu(ds, torch.device("cpu"))
        assert raw.shape == (n, t)
        assert raw.dtype == torch.float32

    def test_cache_supcon_shapes_and_labels(self):
        n, t   = 6, T
        labels = list(range(n))
        ds     = _FakeDataset(n, t, labels=labels)
        raw, lbls = cache_supcon_on_gpu(ds, torch.device("cpu"))
        assert raw.shape   == (n, t)
        assert lbls.shape  == (n,)
        assert lbls.tolist() == labels

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
    def test_cache_raw_on_cuda(self):
        ds  = _FakeDataset(4, T)
        raw = cache_raw_on_gpu(ds, torch.device("cuda"))
        assert raw.device.type == "cuda"


class TestCompile:
    @pytest.mark.skipif(
        not hasattr(torch, "compile"),
        reason="torch.compile not available",
    )
    def test_compiled_barlow_epoch_runs(self):
        device = torch.device("cpu")
        model  = torch.compile(_barlow_model(device).train())
        opt    = _optimizer(model.parameters())
        scaler = _scaler(device)
        spec   = RunSpec("cs_biased", seed=0, ratio=20)
        result = run_barlow_epoch(model, _raw(device), opt, scaler, device, spec, epoch=0, train=True)
        assert torch.isfinite(torch.tensor(result["loss"]))

    @pytest.mark.skipif(
        not hasattr(torch, "compile") or not torch.cuda.is_available(),
        reason="torch.compile or cuda not available",
    )
    def test_compiled_barlow_epoch_cuda(self):
        device = torch.device("cuda")
        model  = torch.compile(_barlow_model(device).train())
        opt    = _optimizer(model.parameters())
        scaler = _scaler(device)
        spec   = RunSpec("cs_biased", seed=0, ratio=20)
        result = run_barlow_epoch(model, _raw(device), opt, scaler, device, spec, epoch=0, train=True)
        assert torch.isfinite(torch.tensor(result["loss"]))
