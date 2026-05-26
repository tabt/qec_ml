"""
qec_ml.models.leakage_detector
================================
ML models for leakage detection.

v2 key changes
--------------
- LeakageDetectorCNN now takes (N, R, H, W) directly — temporal structure
  is preserved as the channel/time dimension, not flattened away
- PersistenceMLP: simple MLP on the persistence_map feature
  (hand-engineered: per-ancilla max consecutive-zero streak)
  This is interpretable and often the strongest baseline
- LeakageClassifierTransformer: unchanged, still multi-task
- SyndromeAnomalyDetector: unchanged
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PersistenceMLP(nn.Module):
    """
    MLP that classifies leakage from the persistence_map feature.

    persistence_map[i, a] = max consecutive rounds that ancilla a
    was silent (zero) during shot i.  This is the hand-engineered
    leakage signal: a normal shot has persistence ~0-1 everywhere,
    a leaked shot has persistence >= 2-3 near the leaked qubit.

    This is intentionally simple and interpretable — it serves as
    the strong baseline that confirms the feature is meaningful
    before applying more complex models to the raw syndrome.

    Input:  (B, n_ancilla) persistence map
    Output: (B,) binary logit
    """
    def __init__(self, n_ancilla: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_ancilla, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),    nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_ancilla) float"""
        return self.net(x).squeeze(-1)


class LeakageDetectorCNN(nn.Module):
    """
    Spatio-temporal CNN for leakage detection.

    Input: (B, R, H, W) round-by-round syndrome grid.
         NOT the flat syndrome — callers must reshape first.
         Use LeakageDataset.round_syndromes for this.

    The temporal dimension R is treated as the channel axis in a
    3D conv: the conv kernel slides spatially (H, W) while seeing
    all R rounds at once, learning the persistence pattern directly.

    Architecture
    ------------
    (B, R, H, W)
      → Conv2d(R→C, 3×3)           — spatial features across all rounds
      → [ResBlocks]
      → GlobalAvgPool
      → MLP → (B,)
    """
    def __init__(
        self,
        distance: int = 5,
        rounds:   int = 5,
        base_channels: int = 64,
        n_blocks: int = 4,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.h = distance - 1
        self.w = distance
        self.rounds = rounds

        # Treat rounds as input channels — the network sees all rounds
        # simultaneously and learns persistence patterns across them
        self.stem = nn.Sequential(
            nn.Conv2d(rounds, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels), nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            _ResCNNBlock(base_channels, dropout) for _ in range(n_blocks)
        ])
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
        x: (B, R, H, W) — round-structured syndrome grid
           OR (B, syndrome_length) flat — will be auto-reshaped
        """
        if x.dim() == 2:
            B = x.shape[0]
            n_use = self.rounds * self.h * self.w
            s = x[:, :n_use]
            if s.shape[1] < n_use:
                s = F.pad(s, (0, n_use - s.shape[1]))
            x = s.reshape(B, self.rounds, self.h, self.w)

        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


class _ResCNNBlock(nn.Module):
    def __init__(self, ch, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.GELU(), nn.Dropout2d(dropout),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.block(x))


class LeakageClassifierTransformer(nn.Module):
    """
    Transformer with two output heads: logical error + leakage.
    Uses 2D spatial + temporal positional embeddings.
    Input: (B, syndrome_length) flat syndrome.
    """
    def __init__(self, distance=5, rounds=5, d_model=128, n_heads=8,
                 n_layers=6, dropout=0.1):
        super().__init__()
        self.h = distance - 1
        self.w = distance
        self.rounds = rounds
        self.n_ancilla = self.h * self.w

        self.token_embed  = nn.Embedding(2, d_model)
        self.row_embed    = nn.Embedding(self.h,  d_model)
        self.col_embed    = nn.Embedding(self.w,  d_model)
        self.round_embed  = nn.Embedding(rounds,  d_model)
        self.cls_token    = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                             norm=nn.LayerNorm(d_model))
        def _head():
            return nn.Sequential(
                nn.Linear(d_model, d_model//2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d_model//2, 1),
            )
        self.logical_head = _head()
        self.leakage_head = _head()

        rows, cols, rnds = [], [], []
        for r in range(rounds):
            for i in range(self.h):
                for j in range(self.w):
                    rows.append(i); cols.append(j); rnds.append(r)
        self.register_buffer("_rows", torch.tensor(rows, dtype=torch.long))
        self.register_buffer("_cols", torch.tensor(cols, dtype=torch.long))
        self.register_buffer("_rnds", torch.tensor(rnds, dtype=torch.long))

    def forward(self, syndrome: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = syndrome.shape[0]
        n_tok = self.rounds * self.n_ancilla
        s = syndrome[:, :n_tok].long()
        if s.shape[1] < n_tok:
            s = F.pad(s, (0, n_tok - s.shape[1]))
        tok = self.token_embed(s)
        pos = (self.row_embed(self._rows)
               + self.col_embed(self._cols)
               + self.round_embed(self._rnds))
        tok = tok + pos.unsqueeze(0)
        cls = self.cls_token.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1)
        out = self.encoder(tok)[:, 0]
        return self.logical_head(out).squeeze(-1), self.leakage_head(out).squeeze(-1)


class SyndromeAnomalyDetector(nn.Module):
    """Autoencoder trained on normal syndromes; high recon error = leakage."""
    def __init__(self, syndrome_length, latent_dim=32, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(syndrome_length, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),       nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,  hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim,  hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim,  syndrome_length), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def anomaly_score(self, x):
        recon, _ = self.forward(x)
        return F.mse_loss(recon, x, reduction="none").mean(dim=1)
