"""
channel.py
Wireless channel models integrated as non-trainable layers.

Implements:
  - AWGN channel              (Eq. 2 in paper with h=1)
  - Rayleigh flat fading      (h ~ CN(0,1))
  - Rician fading             (h = LoS + scatter component)
  - Channel equalization      (Eq. 3: ŷ = x̂ + n̂)

All channels accept/return complex-valued symbol tensors.
Internally we work with (real, imag) stacked as real tensors
to keep compatibility with PyTorch autograd.
"""

import torch
import torch.nn as nn
import math
import config as C


def snr_db_to_sigma(snr_db: float) -> float:
    """
    Convert SNR in dB to noise standard deviation σ.
    We assume average signal power = 1 (enforced by power normalization).
    SNR = 1/σ²  →  σ = 1/sqrt(10^(SNR_dB/10))
    """
    snr_linear = 10.0 ** (snr_db / 10.0)
    return math.sqrt(1.0 / snr_linear)


def power_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Power normalization as in Eq. (5) of the paper.
    x: (N, 2b) real tensor [real parts | imag parts concatenated]
    Returns x̂ with E[||x̂||²]/b = 1
    """
    # b = half the feature dim (since real+imag)
    b = x.shape[-1] // 2
    norm = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True) / (2 * b) + 1e-8)
    return x / norm


class AWGNChannel(nn.Module):
    """
    Additive White Gaussian Noise channel.
    y = x + n,  n ~ CN(0, σ²I)
    After equalization (h=1): ŷ = x + n̂  (same since h=1)
    """
    def __init__(self):
        super().__init__()

    def forward(self,
                x: torch.Tensor,
                snr_db: float = C.SNR_TEST_DB) -> tuple:
        """
        Args:
            x:      (N, feature_dim) normalized real feature tensor
            snr_db: Signal-to-noise ratio in dB

        Returns:
            y_hat:   Channel output after equalization (same shape as x)
            sigma_sq: Noise variance σ² (scalar, for gated net input)
        """
        sigma    = snr_db_to_sigma(snr_db)
        noise    = torch.randn_like(x) * sigma
        y        = x + noise
        sigma_sq = torch.tensor(sigma ** 2,
                                dtype=x.dtype,
                                device=x.device)
        return y, sigma_sq


class RayleighChannel(nn.Module):
    """
    Frequency-flat block Rayleigh fading channel.
    y = h·x + n,  h ~ CN(0,1),  n ~ CN(0, σ²I)
    After equalization: ŷ = x + n/h  = x + n̂
    Equivalent noise variance: σ̂² = σ²/|h|²
    """
    def __init__(self):
        super().__init__()

    def forward(self,
                x: torch.Tensor,
                snr_db: float = C.SNR_TEST_DB) -> tuple:
        sigma    = snr_db_to_sigma(snr_db)
        N        = x.shape[0]
        device   = x.device
        dtype    = x.dtype

        # Sample complex channel coefficient h ~ CN(0,1) per image
        h_real   = torch.randn(N, 1, device=device, dtype=dtype) / math.sqrt(2)
        h_imag   = torch.randn(N, 1, device=device, dtype=dtype) / math.sqrt(2)
        h_mag_sq = h_real ** 2 + h_imag ** 2           # |h|²  shape (N,1)

        # Split x into real/imag halves for complex multiplication
        half     = x.shape[-1] // 2
        x_r, x_i = x[..., :half], x[..., half:]

        # y = h·x  (complex multiply)
        y_r      = h_real * x_r - h_imag * x_i
        y_i      = h_real * x_i + h_imag * x_r

        # Add noise
        noise_r  = torch.randn_like(x_r) * sigma
        noise_i  = torch.randn_like(x_i) * sigma
        y_r      = y_r + noise_r
        y_i      = y_i + noise_i

        # Channel equalization: divide by h*  (conjugate division = divide by |h|²)
        # ŷ_r = (y_r·h_r + y_i·h_i) / |h|²
        # ŷ_i = (y_i·h_r - y_r·h_i) / |h|²
        eps      = 1e-8
        y_hat_r  = (y_r * h_real + y_i * h_imag) / (h_mag_sq + eps)
        y_hat_i  = (y_i * h_real - y_r * h_imag) / (h_mag_sq + eps)
        y_hat    = torch.cat([y_hat_r, y_hat_i], dim=-1)

        # Equivalent noise variance σ̂² = σ²/|h|² (mean over batch)
        sigma_sq = (sigma ** 2 / (h_mag_sq.mean() + eps)).detach()
        return y_hat, sigma_sq


class RicianChannel(nn.Module):
    """
    Frequency-flat Rician fading channel.
    LoS component + scattered component.
    h = sqrt(K/(K+1))·exp(jφ) + sqrt(1/(K+1))·h_scatter,
    where h_scatter ~ CN(0,1) and φ is random LoS phase.
    """
    def __init__(self, K_factor: float = C.RICIAN_K):
        super().__init__()
        self.K = K_factor

    def forward(self,
                x: torch.Tensor,
                snr_db: float = C.SNR_TEST_DB) -> tuple:
        sigma    = snr_db_to_sigma(snr_db)
        N        = x.shape[0]
        device   = x.device
        dtype    = x.dtype
        K        = self.K

        # LoS component
        phi      = 2 * math.pi * torch.rand(N, 1, device=device, dtype=dtype)
        h_los_r  = math.sqrt(K / (K + 1)) * torch.cos(phi)
        h_los_i  = math.sqrt(K / (K + 1)) * torch.sin(phi)

        # Scatter component
        h_sc_r   = math.sqrt(1 / (2*(K+1))) * torch.randn(N,1,device=device,dtype=dtype)
        h_sc_i   = math.sqrt(1 / (2*(K+1))) * torch.randn(N,1,device=device,dtype=dtype)

        h_real   = h_los_r + h_sc_r
        h_imag   = h_los_i + h_sc_i
        h_mag_sq = h_real ** 2 + h_imag ** 2

        half     = x.shape[-1] // 2
        x_r, x_i = x[..., :half], x[..., half:]

        y_r      = h_real * x_r - h_imag * x_i + torch.randn_like(x_r) * sigma
        y_i      = h_real * x_i + h_imag * x_r + torch.randn_like(x_i) * sigma

        eps      = 1e-8
        y_hat_r  = (y_r * h_real + y_i * h_imag) / (h_mag_sq + eps)
        y_hat_i  = (y_i * h_real - y_r * h_imag) / (h_mag_sq + eps)
        y_hat    = torch.cat([y_hat_r, y_hat_i], dim=-1)

        sigma_sq = (sigma ** 2 / (h_mag_sq.mean() + eps)).detach()
        return y_hat, sigma_sq


def get_channel(channel_type: str = C.CHANNEL_TYPE) -> nn.Module:
    """Factory function to select channel model."""
    channels = {
        "awgn":     AWGNChannel,
        "rayleigh": RayleighChannel,
        "rician":   RicianChannel,
    }
    assert channel_type in channels, \
        f"Unknown channel type '{channel_type}'. Choose from {list(channels)}"
    return channels[channel_type]()


def sample_snr_db(snr_min: float = C.SNR_MIN_DB,
                  snr_max: float = C.SNR_MAX_DB) -> float:
    """
    Uniformly sample SNR in dB for domain randomization (Algorithm 2, paper).
    Called once per training batch.
    """
    return snr_min + (snr_max - snr_min) * torch.rand(1).item()
