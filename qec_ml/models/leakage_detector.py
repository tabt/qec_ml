"""
qec_ml.models.leakage_detector
================================
ML models for leakage detection and classification.

Task framing
------------
Given a spatio-temporal syndrome pattern, predict:
  - Binary: is there a leakage event in this shot? (detection)
  - 3-class: no error / Pauli logical error / leakage event (classification)

The key signal for leakage is the "dark detector" pattern:
  - In normal operation, ancilla detectors fire stochastically
  - A leaked data qubit SILENCES all adjacent ancillas for the
    entire duration of the leakage
  - This creates a spatially localised, temporally persistent
    region of zeros in the syndrome — unlike Pauli errors which
    produce isolated syndrome flips

Models in this module
---------------------
LeakageDetectorCNN
    2D+time CNN that recognises dark-detector spatial patterns.
    Input: (B, R, H, W) syndrome grid.
    Output: binary leakage flag logit.

LeakageClassifierTransformer
    Transformer that jointly predicts logical error and leakage class.
    Uses the AncillaTransformer backbone with a multi-task head.

SyndromeAnomalyDetector
    Unsupervised approach: train an autoencoder on normal (no-leakage)
    syndromes, then use reconstruction error as anomaly score.
    High reconstruction error → likely leakage (out-of-distribution).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ======================================================================
# LeakageDetectorCNN
# ======================================================================

class LeakageDetectorCNN(nn.Module):
    """
    2D convolutional leakage detector.

    Processes the syndrome as a (rounds × H × W) video, where each
    frame is one round of ancilla measurements.  Temporal convolutions
    across rounds capture the "persistence" that characterises leakage.

    Architecture
    ------------
    Spatial branch:  2D conv on each round independently
    Temporal branch: 1D conv across rounds per spatial position
    Joint:           concatenated features → MLP → binary logit

    Parameters
    ----------
    distance : int
    rounds : int
    base_channels : int
    dropout : float
    """

    def __init__(
        self,
        distance: int = 5,
        rounds: int = 5,
        base_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.h = distance - 1
        self.w = distance
        self.rounds = rounds

        # Spatial feature extractor (shared across rounds)
        self.spatial = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels), nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels), nn.GELU(),
            nn.AdaptiveAvgPool2d(2),   # → (B*R, C, 2, 2)
        )

        # Temporal feature extractor (across rounds)
        spatial_out_dim = base_channels * 4   # 2x2 pool
        self.temporal = nn.Sequential(
            nn.Conv1d(spatial_out_dim, base_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 2, base_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_channels, 1),  # binary: leakage or not
        )

    def forward(self, syndrome: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        syndrome : (B, syndrome_length) flat syndrome

        Returns
        -------
        logits : (B,) — positive = leakage predicted
        """
        B = syndrome.shape[0]
        R, H, W = self.rounds, self.h, self.w
        n_use = R * H * W

        # Pad/crop to fit grid
        s = syndrome[:, :n_use]
        if s.shape[1] < n_use:
            pad = torch.zeros(B, n_use - s.shape[1], device=syndrome.device)
            s = torch.cat([s, pad], dim=1)

        # Reshape: (B, R, H, W)
        grid = s.view(B, R, H, W)

        # Apply spatial conv to each round independently
        # Reshape to (B*R, 1, H, W)
        grid_flat = grid.view(B * R, 1, H, W)
        spatial_feat = self.spatial(grid_flat)          # (B*R, C, 2, 2)
        spatial_feat = spatial_feat.view(B, R, -1)      # (B, R, C*4)
        spatial_feat = spatial_feat.permute(0, 2, 1)    # (B, C*4, R)

        # Temporal conv across rounds
        temporal_feat = self.temporal(spatial_feat)     # (B, C*2, 1)

        return self.head(temporal_feat).squeeze(-1)


# ======================================================================
# LeakageClassifierTransformer  (multi-task)
# ======================================================================

class LeakageClassifierTransformer(nn.Module):
    """
    Transformer with two output heads:
      1. Logical error prediction (binary)
      2. Leakage detection (binary)

    Both tasks share the Transformer encoder — the shared representation
    benefits from multi-task learning: leakage context helps correct
    Pauli error decoding, and vice versa.

    Parameters
    ----------
    distance, rounds, d_model, n_heads, n_layers, dropout : standard
    """

    def __init__(
        self,
        distance: int = 5,
        rounds: int = 5,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.h = distance - 1
        self.w = distance
        self.rounds = rounds
        self.n_ancilla = self.h * self.w

        # Token embedding
        self.token_embed = nn.Embedding(2, d_model)

        # 2D + temporal positional embeddings (same as AncillaTransformer)
        self.row_embed   = nn.Embedding(self.h, d_model)
        self.col_embed   = nn.Embedding(self.w, d_model)
        self.round_embed = nn.Embedding(rounds, d_model)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model)
        )

        # Two separate heads
        def _head():
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d_model // 2, 1),
            )
        self.logical_head  = _head()   # P(logical error)
        self.leakage_head  = _head()   # P(leakage event)

        # Precompute positional indices
        rows, cols, rnds = [], [], []
        for r in range(rounds):
            for i in range(self.h):
                for j in range(self.w):
                    rows.append(i); cols.append(j); rnds.append(r)
        self.register_buffer("_rows", torch.tensor(rows, dtype=torch.long))
        self.register_buffer("_cols", torch.tensor(cols, dtype=torch.long))
        self.register_buffer("_rnds", torch.tensor(rnds, dtype=torch.long))

    def forward(self, syndrome: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        syndrome : (B, syndrome_length)

        Returns
        -------
        logical_logits  : (B,)
        leakage_logits  : (B,)
        """
        B = syndrome.shape[0]
        n_tok = self.rounds * self.n_ancilla

        s = syndrome[:, :n_tok].long()
        if s.shape[1] < n_tok:
            pad = torch.zeros(B, n_tok - s.shape[1], dtype=torch.long, device=s.device)
            s = torch.cat([s, pad], dim=1)

        tok = self.token_embed(s)
        pos = (self.row_embed(self._rows)
               + self.col_embed(self._cols)
               + self.round_embed(self._rnds))
        tok = tok + pos.unsqueeze(0)

        cls = self.cls_token.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1)
        out = self.encoder(tok)
        cls_out = out[:, 0]   # (B, d_model)

        return (
            self.logical_head(cls_out).squeeze(-1),
            self.leakage_head(cls_out).squeeze(-1),
        )


# ======================================================================
# SyndromeAnomalyDetector  (unsupervised)
# ======================================================================

class SyndromeAnomalyDetector(nn.Module):
    """
    Convolutional autoencoder trained on *normal* (no-leakage) syndromes.

    At inference time, syndromes with leakage produce high reconstruction
    error because they contain out-of-distribution "dark" patterns.

    This is the *unsupervised* approach: no leakage labels needed for
    training.  The anomaly score is the per-sample MSE reconstruction loss.

    Parameters
    ----------
    syndrome_length : int
    latent_dim : int     — bottleneck dimension
    hidden_dim : int
    """

    def __init__(
        self,
        syndrome_length: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.syndrome_length = syndrome_length

        # Encoder: syndrome → latent
        self.encoder = nn.Sequential(
            nn.Linear(syndrome_length, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Decoder: latent → syndrome
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, syndrome_length), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        (reconstructed, latent) : shapes (B, L) and (B, latent_dim)
        """
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample reconstruction MSE — high score → likely leakage."""
        recon, _ = self.forward(x)
        return F.mse_loss(recon, x, reduction="none").mean(dim=1)
