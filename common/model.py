import torch
import torch.nn as nn
import torchaudio.functional as AF

EPS = 1e-12


def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    n, m = matrix.shape
    if n != m:
        raise ValueError("expected square matrix")
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_twins_loss(
    left: torch.Tensor,
    right: torch.Tensor,
    lambd: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = left.size(0)
    left  = (left  - left.mean(dim=0))  / left.std(dim=0).clamp_min(EPS)
    right = (right - right.mean(dim=0)) / right.std(dim=0).clamp_min(EPS)
    correlation = left.T @ right / batch_size
    on_diag  = torch.diagonal(correlation).add_(-1.0).pow_(2).sum()
    off_diag = off_diagonal(correlation).pow_(2).sum()
    return on_diag + float(lambd) * off_diag, on_diag, off_diag


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

    def to_mel(self, x: torch.Tensor) -> torch.Tensor:
        y    = x.squeeze(1)
        spec = torch.stft(y, n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.n_fft, window=self.window, return_complex=True)
        mel  = torch.einsum("bft,fm->bmt", spec.abs(), self.mel_fb)
        mel  = torch.log1p(mel).unsqueeze(1)
        mean = mel.mean(dim=(2, 3), keepdim=True)
        std  = mel.std(dim=(2, 3), keepdim=True).clamp_min(EPS)
        return (mel - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat   = self.features(self.to_mel(x))
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
        self.encoder   = WaveSTFTEncoder(
            embedding_dim, 
            base_channels, 
            n_fft, 
            hop_length, 
            n_blocks, 
            n_mels, 
            sample_rate
        )
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, projection_hidden_dim, bias=False),
            nn.BatchNorm1d(projection_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(projection_hidden_dim, projection_dim, bias=False),
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        return h1, h2, self.projector(h1), self.projector(h2)
