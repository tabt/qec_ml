"""
qec_ml.models.lstm_corrector
==============================
LSTM and 1D-CNN models for analog IQ readout classification and denoising.

These models operate on time-series IQ trajectories from dispersive
qubit readout, rather than discrete syndrome bits.

Two tasks are supported:

1. **Classification** — predict the qubit state |0⟩ / |1⟩ from the
   raw IQ trajectory.  This is analogous to "soft" decoding.

2. **Denoising** — reconstruct a cleaner IQ trajectory from a noisy
   one (autoencoder framing).  Useful for downstream threshold decisions.

Models
------
- LSTMClassifier   : bidirectional LSTM → class probabilities
- Conv1DClassifier : 1D-CNN for IQ time series
- IQAutoencoder    : convolutional encoder-decoder for trajectory denoising

References
----------
- Magesan et al. (2015). Machine Learning for Discriminating Quantum
  Measurement Trajectories. PRL 114, 200501.
- Baireuther et al. (2018). Machine-learning-assisted correction of
  a superconducting qubit clock. npj Quantum Information 4, 48.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ======================================================================
# LSTM Classifier
# ======================================================================

class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for IQ trajectory classification.

    The IQ signal is treated as a sequence of (I, Q) pairs over time.
    A bidirectional LSTM captures both causal and anti-causal patterns
    (e.g., state relaxation visible only late in the trajectory).

    Parameters
    ----------
    input_size : int
        Feature dimension per timestep (2 for a single qubit: I, Q).
    hidden_size : int
        LSTM hidden state dimension (per direction).
    n_layers : int
        Number of stacked LSTM layers.
    dropout : float
    n_classes : int
        Number of output classes (2 for |0⟩ / |1⟩).

    Examples
    --------
    >>> model = LSTMClassifier(input_size=2, hidden_size=64)
    >>> x = torch.randn(32, 100, 2)   # (B, T, 2)
    >>> logits = model(x)             # (32, 2)
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
        n_classes: int = 2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(2 * hidden_size)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, input_size)

        Returns
        -------
        logits : (B, n_classes)
        """
        out, _ = self.lstm(x)           # (B, T, 2*H)
        # Use mean pooling over time (more robust than last-step readout)
        pooled = out.mean(dim=1)        # (B, 2*H)
        pooled = self.norm(pooled)
        return self.head(pooled)


# ======================================================================
# 1D-CNN Classifier
# ======================================================================

class Conv1DClassifier(nn.Module):
    """
    1D Convolutional classifier for IQ time-series data.

    Faster and often competitive with LSTM for readout classification.
    Uses dilated convolutions to capture multi-scale temporal patterns.

    Parameters
    ----------
    input_size : int
    n_filters : int
    n_blocks : int
    dropout : float
    n_classes : int
    """

    def __init__(
        self,
        input_size: int = 2,
        n_filters: int = 64,
        n_blocks: int = 4,
        dropout: float = 0.1,
        n_classes: int = 2,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(input_size, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_filters),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            DilatedResBlock1D(n_filters, dilation=2 ** i, dropout=dropout)
            for i in range(n_blocks)
        ])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_filters, n_filters // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_filters // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, input_size)  — note: time-last after transpose

        Returns
        -------
        logits : (B, n_classes)
        """
        x = x.transpose(1, 2)          # → (B, input_size, T)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class DilatedResBlock1D(nn.Module):
    """Dilated 1D residual block."""

    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = dilation
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=pad,
                      dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=pad,
                      dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


# ======================================================================
# IQ Autoencoder (Denoiser)
# ======================================================================

class IQAutoencoder(nn.Module):
    """
    Convolutional autoencoder for IQ trajectory denoising.

    Encoder compresses the noisy time series into a latent vector;
    decoder reconstructs a cleaner version.

    Loss: MSE between decoded trajectory and noiseless template.

    Parameters
    ----------
    input_size : int
        IQ channels (2 for single-qubit readout).
    n_filters : int
    latent_dim : int
        Bottleneck dimension.
    seq_len : int
        Time-series length (needed for decoder upsampling).
    """

    def __init__(
        self,
        input_size: int = 2,
        n_filters: int = 32,
        latent_dim: int = 16,
        seq_len: int = 100,
    ):
        super().__init__()
        self.seq_len = seq_len

        # Encoder: (B, input_size, T) → (B, latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_size, n_filters, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(n_filters, n_filters * 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(n_filters * 2, n_filters * 4, 4, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_filters * 4, latent_dim),
        )

        # Decoder: (B, latent_dim) → (B, input_size, T)
        self.decoder_fc = nn.Linear(latent_dim, n_filters * 4 * (seq_len // 8))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(n_filters * 4, n_filters * 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(n_filters * 2, n_filters, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(n_filters, input_size, 4, stride=2, padding=1),
        )
        self.n_filters = n_filters

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → latent: (B, latent_dim)"""
        return self.encoder(x.transpose(1, 2))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → (B, T, C)"""
        B = z.shape[0]
        x = self.decoder_fc(z).view(B, self.n_filters * 4, self.seq_len // 8)
        x = self.decoder_conv(x)
        return x[:, :, :self.seq_len].transpose(1, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, T, C)

        Returns
        -------
        (reconstructed: (B, T, C), latent: (B, latent_dim))
        """
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z
