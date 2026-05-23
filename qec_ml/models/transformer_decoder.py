"""
qec_ml.models.transformer_decoder
====================================
Improved Transformer decoders for surface-code syndrome classification.

Key improvements over v1:
  - SyndromeTransformer: Pre-LN (norm_first), lower default LR target,
    syndrome-aware positional encoding (row/col embeddings for ancilla grid)
  - SpatialTemporalTransformer: unchanged, already good
  - New: AncillaTransformer — 2D spatial position embeddings that match
    the actual (d-1)×d ancilla layout of the rotated surface code.

Architecture insight
--------------------
The failure mode in v1 was the model learning to always predict the
majority class.  Root causes:
  1. Positional encodings were 1D (sequential), but the syndrome has
     2D spatial + 1D temporal structure.
  2. LR schedule reached high LR before the model had learned anything
     useful, causing it to collapse to a trivial solution.
  3. No class weighting — combined with warmup instability.

Fixes:
  - 2D row/col positional embeddings for AncillaTransformer
  - Default to lower peak LR (use 1e-4 in TrainingConfig, not 3e-4)
  - Use FocalLoss (in mlp_decoder.py) instead of BCE
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ======================================================================
# v1 SyndromeTransformer  (kept, with one fix: norm_first=True enforced)
# ======================================================================

class SyndromeTransformer(nn.Module):
    """
    Transformer encoder over flat syndrome bits.
    Each of the L syndrome bits becomes one token.

    Improvements over v1:
    - norm_first=True (Pre-LN) is now always on — critical for stable training
    - Default d_model bumped to 128 (was fine), n_layers to 6
    - CLS token used by default
    """

    def __init__(
        self,
        syndrome_length: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.syndrome_length = syndrome_length
        self.d_model = d_model
        self.use_cls = use_cls_token
        if d_ff is None:
            d_ff = 4 * d_model

        self.token_embed = nn.Embedding(2, d_model)
        seq_len = syndrome_length + (1 if use_cls_token else 0)
        self.pos_embed = nn.Embedding(seq_len, d_model)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,   # Pre-LN: essential for stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model)
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
        if self.use_cls:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, syndrome: torch.Tensor) -> torch.Tensor:
        B, L = syndrome.shape
        tok = self.token_embed(syndrome.long())
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)
            tok = torch.cat([cls, tok], dim=1)
        positions = torch.arange(tok.size(1), device=syndrome.device)
        tok = tok + self.pos_embed(positions)
        out = self.encoder(tok)
        cls_out = out[:, 0] if self.use_cls else out.mean(dim=1)
        return self.head(cls_out).squeeze(-1)


# ======================================================================
# AncillaTransformer — physically-motivated 2D+temporal positional encoding
# ======================================================================

class AncillaTransformer(nn.Module):
    """
    Transformer with 2D spatial + temporal positional embeddings.

    Instead of treating the syndrome as an arbitrary sequence, we assign
    each ancilla bit a (row, col, round) coordinate matching the actual
    rotated surface code geometry.

    Layout for distance-d code:
      - Ancillas arranged in (d-1) rows × d columns
      - R measurement rounds → R × (d-1) × d tokens
      - Separate learnable embeddings for row, col, round
      - Total positional embedding = embed_row + embed_col + embed_round

    This gives the model explicit knowledge of the lattice topology.

    Parameters
    ----------
    distance : int
    rounds : int
    d_model : int
    n_heads : int
    n_layers : int
    d_ff : int, optional
    dropout : float
    """

    def __init__(
        self,
        distance: int = 5,
        rounds: int = 5,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.distance = distance
        self.rounds = rounds
        self.h = distance - 1        # number of ancilla rows
        self.w = distance            # number of ancilla columns
        self.n_ancilla = self.h * self.w
        if d_ff is None:
            d_ff = 4 * d_model

        # Token embedding (binary syndrome bit)
        self.token_embed = nn.Embedding(2, d_model)

        # 2D spatial + temporal positional embeddings
        self.row_embed   = nn.Embedding(self.h, d_model)
        self.col_embed   = nn.Embedding(self.w, d_model)
        self.round_embed = nn.Embedding(rounds, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model)
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, 1),
        )

        # Precompute coordinate tensors (row, col, round) for each token
        rows, cols, rnds = [], [], []
        for r in range(rounds):
            for i in range(self.h):
                for j in range(self.w):
                    rows.append(i); cols.append(j); rnds.append(r)
        self.register_buffer("_rows", torch.tensor(rows, dtype=torch.long))
        self.register_buffer("_cols", torch.tensor(cols, dtype=torch.long))
        self.register_buffer("_rnds", torch.tensor(rnds, dtype=torch.long))

        self._init_weights()

    def _init_weights(self):
        for emb in [self.row_embed, self.col_embed, self.round_embed, self.token_embed]:
            nn.init.trunc_normal_(emb.weight, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, syndrome: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        syndrome : (B, syndrome_length) — flat syndrome vector {0,1}
            syndrome_length should be >= rounds * (d-1) * d.
            Extra bits (if any) are ignored.

        Returns
        -------
        logits : (B,)
        """
        B = syndrome.shape[0]
        n_tok = self.rounds * self.n_ancilla

        # Crop/pad to exactly n_tok bits
        s = syndrome[:, :n_tok].long()
        if s.shape[1] < n_tok:
            pad = torch.zeros(B, n_tok - s.shape[1], dtype=torch.long, device=s.device)
            s = torch.cat([s, pad], dim=1)

        # Token embeddings: (B, n_tok, d_model)
        tok = self.token_embed(s)

        # Add 2D+temporal positional embeddings
        pos = (self.row_embed(self._rows)
               + self.col_embed(self._cols)
               + self.round_embed(self._rnds))   # (n_tok, d_model)
        tok = tok + pos.unsqueeze(0)

        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1)        # (B, n_tok+1, d_model)

        out = self.encoder(tok)
        return self.head(out[:, 0]).squeeze(-1)   # CLS readout


# ======================================================================
# SpatialTemporalTransformer  (v1, unchanged — already has factored pos)
# ======================================================================

class SpatialTemporalTransformer(nn.Module):
    """
    2D spatial + temporal Transformer (v1 architecture, unchanged).
    For comparison with AncillaTransformer.
    """

    def __init__(
        self,
        n_ancilla: int,
        rounds: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_ancilla = n_ancilla
        self.rounds = rounds
        if d_ff is None:
            d_ff = 4 * d_model

        self.token_embed  = nn.Embedding(2, d_model)
        self.spatial_pos  = nn.Embedding(n_ancilla, d_model)
        self.temporal_pos = nn.Embedding(rounds, d_model)
        self.cls_token    = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model)
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, 1),
        )

    def forward(self, syndrome: torch.Tensor) -> torch.Tensor:
        B = syndrome.shape[0]
        R, A = self.rounds, self.n_ancilla
        s = syndrome.long().view(B, R, A)
        tok = self.token_embed(s)
        sp = self.spatial_pos(torch.arange(A, device=syndrome.device))
        tp = self.temporal_pos(torch.arange(R, device=syndrome.device))
        tok = tok + sp[None, None] + tp[None, :, None]
        tok = tok.view(B, R * A, -1)
        cls = self.cls_token.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1)
        out = self.encoder(tok)
        return self.head(out[:, 0]).squeeze(-1)
