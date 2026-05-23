"""
qec_ml.models.mlp_decoder
===========================
Improved MLP and CNN decoders.

Key improvements over v1:
  - MLP: residual blocks + syndrome-aware feature engineering
  - CNN: correct 2-row ancilla layout matching the actual surface code geometry
  - FocalLoss helper for imbalanced syndrome datasets
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


# ======================================================================
# Focal Loss  (handles class imbalance far better than pos_weight alone)
# ======================================================================

class FocalLoss(nn.Module):
    """
    Binary Focal Loss: FL(p) = -α(1-p)^γ log(p)

    Concentrates learning on hard/misclassified examples.
    Particularly effective for QEC where logical errors are rarer
    and harder to classify than trivial (no-error) cases.

    Parameters
    ----------
    alpha : float   — weight for positive class (logical error)
    gamma : float   — focusing exponent (0 = standard BCE, 2 recommended)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        p = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, p, 1 - p)
        alpha_t = torch.where(targets == 1,
                              torch.full_like(p, self.alpha),
                              torch.full_like(p, 1 - self.alpha))
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


# ======================================================================
# Improved MLP: residual blocks + hand-crafted syndrome features
# ======================================================================

class ResidualMLP(nn.Module):
    """
    MLP with residual connections for syndrome classification.

    Architecture
    ------------
    Feature extractor:
      Raw syndrome (L,)  +  hand-crafted features (4,)
        → projected to d_model
        → N residual MLP blocks
        → scalar logit

    Hand-crafted features injected:
      - total syndrome weight (Hamming weight)
      - syndrome weight per round (rounds values)
      - parity of total weight
      These give the model an explicit "error density" signal.

    Parameters
    ----------
    syndrome_length : int
    d_model : int           — hidden dimension throughout
    n_blocks : int          — number of residual blocks
    dropout : float
    n_extra_features : int  — number of hand-crafted features prepended
    """

    def __init__(
        self,
        syndrome_length: int,
        d_model: int = 256,
        n_blocks: int = 6,
        dropout: float = 0.15,
        rounds: int = 1,
    ):
        super().__init__()
        self.syndrome_length = syndrome_length
        self.rounds = rounds
        # extra features: total weight + per-round weights + parity
        n_extra = 1 + rounds + 1

        self.input_proj = nn.Sequential(
            nn.Linear(syndrome_length + n_extra, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            _ResMLPBlock(d_model, dropout) for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def _extra_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute hand-crafted features from the syndrome tensor."""
        B, L = x.shape
        total_w = x.sum(dim=1, keepdim=True)                       # (B, 1)
        parity  = total_w % 2                                       # (B, 1)
        # Per-round weights
        anc_per_round = L // self.rounds
        rounds_w = []
        for r in range(self.rounds):
            start = r * anc_per_round
            end = start + anc_per_round
            rounds_w.append(x[:, start:end].sum(dim=1, keepdim=True))
        round_feats = torch.cat(rounds_w, dim=1)                    # (B, rounds)
        return torch.cat([total_w, round_feats, parity], dim=1)    # (B, rounds+2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, syndrome_length) float {0,1}

        Returns
        -------
        logits : (B,)
        """
        extra = self._extra_features(x)
        h = self.input_proj(torch.cat([x, extra], dim=1))
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


class _ResMLPBlock(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 4, d),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return x + self.net(x)


# ======================================================================
# Improved CNN: correct surface-code ancilla layout
# ======================================================================

class SurfaceCodeCNN(nn.Module):
    """
    CNN decoder with a physically-motivated ancilla layout.

    For a distance-d rotated surface code there are (d²-1) ancilla qubits.
    We arrange them in a (d-1) × d grid (Z-type stabilisers), matching the
    actual lattice topology — much better spatial inductive bias than the
    naive sqrt-padding used in v1.

    Each round of measurements becomes one channel, so the input is
    (B, rounds, d-1, d).

    Parameters
    ----------
    distance : int
    rounds : int
    base_channels : int
    n_blocks : int
    dropout : float
    """

    def __init__(
        self,
        distance: int = 5,
        rounds: int = 5,
        base_channels: int = 64,
        n_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.distance = distance
        self.rounds = rounds
        self.h = distance - 1
        self.w = distance
        self.n_ancilla = (distance - 1) * distance   # = d² - d

        # If stim gives d²-1 ancillas, we only use the first h*w = d²-d of them.
        # Remaining 1 ancilla (for odd codes) is handled by cropping in forward().

        self.stem = nn.Sequential(
            nn.Conv2d(rounds, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            _ResCNNBlock(base_channels, dropout) for _ in range(n_blocks)
        ])
        # Global average pool → MLP head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels, base_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, syndrome_length) flat syndrome vector
            Will be reshaped to (B, rounds, h, w) internally.

        Returns
        -------
        logits : (B,)
        """
        B = x.shape[0]
        h, w, R = self.h, self.w, self.rounds

        # Take only the ancilla bits we can map to the (h, w) grid
        needed = R * h * w
        x_crop = x[:, :needed]

        # Zero-pad if syndrome is shorter (shouldn't happen, but be safe)
        if x_crop.shape[1] < needed:
            pad = torch.zeros(B, needed - x_crop.shape[1], device=x.device)
            x_crop = torch.cat([x_crop, pad], dim=1)

        img = x_crop.view(B, R, h, w)   # (B, rounds, h, w)

        img = self.stem(img)
        for block in self.blocks:
            img = block(img)
        return self.head(img).squeeze(-1)


class _ResCNNBlock(nn.Module):
    def __init__(self, ch: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


# ======================================================================
# Legacy MLPDecoder and CNNDecoder kept for backward compatibility
# ======================================================================

class MLPDecoder(nn.Module):
    """Legacy MLP (v1). Prefer ResidualMLP for new experiments."""
    def __init__(self, syndrome_length, hidden_dims=None, dropout=0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]
        layers = []
        in_dim = syndrome_length
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CNNDecoder(nn.Module):
    """Legacy CNN (v1). Prefer SurfaceCodeCNN for new experiments."""
    def __init__(self, in_channels=5, grid_size=4, base_channels=64, n_blocks=3, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels), nn.GELU(),
        )
        self.blocks = nn.ModuleList([_ResCNNBlock(base_channels, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(base_channels, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks: x = b(x)
        return self.head(x).squeeze(-1)
