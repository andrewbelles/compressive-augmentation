import pytest
import torch
import torch.nn.functional as F

from audio.encoder import AudioBarlowModel, AudioSTFTEncoder
from csmath.losses import barlow_twins_loss

B            = 2
T            = 4096
SR           = 22050
N_FFT        = 1024
HOP          = 256
N_MELS       = 128
N_BLOCKS     = 2
BASE_CH      = 4
EMBED_DIM    = 32
PROJ_HIDDEN  = 64
PROJ_DIM     = 48


def _encoder(device):
    return AudioSTFTEncoder(
        embedding_dim=EMBED_DIM,
        base_channels=BASE_CH,
        n_fft=N_FFT,
        hop_length=HOP,
        n_blocks=N_BLOCKS,
        n_mels=N_MELS,
        sample_rate=SR,
    ).to(device).eval()


def _model(device):
    return AudioBarlowModel(
        embedding_dim=EMBED_DIM,
        base_channels=BASE_CH,
        projection_hidden_dim=PROJ_HIDDEN,
        projection_dim=PROJ_DIM,
        n_fft=N_FFT,
        hop_length=HOP,
        n_blocks=N_BLOCKS,
        n_mels=N_MELS,
        sample_rate=SR,
    ).to(device).eval()


def _wave(device):
    return torch.randn(B, 1, T, device=device)


class TestAudioSTFTEncoder:
    def test_to_mel_shape(self, device):
        enc  = _encoder(device)
        x    = _wave(device)
        mel  = enc.to_mel(x)
        frames = T // HOP + 1
        assert mel.shape == (B, 1, N_MELS, frames)

    def test_forward_shape(self, device):
        enc = _encoder(device)
        out = enc(_wave(device))
        assert out.shape == (B, EMBED_DIM)

    def test_no_nan(self, device):
        enc = _encoder(device)
        out = enc(_wave(device))
        assert torch.isfinite(out).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
    def test_cpu_cuda_parity(self):
        x   = torch.randn(B, 1, T)
        cpu = _encoder(torch.device("cpu"))
        gpu = _encoder(torch.device("cuda"))
        gpu.load_state_dict(cpu.state_dict())
        with torch.no_grad():
            out_cpu = cpu(x)
            out_gpu = gpu(x.cuda()).cpu()
        assert torch.allclose(out_cpu, out_gpu, atol=1e-3)


class TestAudioBarlowModel:
    def test_forward_shapes(self, device):
        model   = _model(device)
        x       = _wave(device)
        h1, h2, z1, z2 = model(x, x.clone())
        assert h1.shape == h2.shape == (B, EMBED_DIM)
        assert z1.shape == z2.shape == (B, PROJ_DIM)

    def test_barlow_loss_computable(self, device):
        model = _model(device)
        x     = _wave(device)
        _, _, z1, z2 = model(x, x.clone())
        loss, _, _ = barlow_twins_loss(z1, z2, lambd=5e-5)
        assert torch.isfinite(loss)

    def test_gradient_flows(self, device):
        model = _model(device).train()
        x     = _wave(device)
        _, _, z1, z2 = model(x, x.clone())
        loss, _, _ = barlow_twins_loss(z1, z2, lambd=5e-5)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)
