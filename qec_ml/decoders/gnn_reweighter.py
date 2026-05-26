"""
qec_ml.decoders.gnn_reweighter
================================
GNN-based MWPM edge weight reweighting — a hybrid ML+classical decoder.

Motivation
----------
Instead of replacing MWPM entirely, we use a GNN to *improve* the
edge weights in the matching graph, then pass the reweighted graph
to PyMatching.  This gives us:
  - The combinatorial optimality guarantees of MWPM
  - The representational power of GNNs for learning noise structure
  - A principled way to handle correlated / non-IID noise

Architecture
------------
Input:  syndrome vector → build PyMatching-style detector graph
GNN:    message-passing over the detector graph to produce
        per-edge weight corrections Δw_ij
Output: original_weight + Δw_ij → reweighted matching graph
Decode: run PyMatching with reweighted graph

Training
--------
Supervised: train GNN to minimise logical error rate.
  Loss = BCE(MWPM_with_reweighted_graph(syndrome), logical_error_label)
  But computing this is expensive (MWPM is not differentiable).

Practical surrogate: train GNN to predict per-edge error probabilities
directly, then use -log(p/(1-p)) as MWPM weight.  This is exactly
the Bayesian optimal weighting, and the GNN learns to predict p_error
for each edge under the true (possibly correlated) noise model.

The surrogate loss is BCE(predicted_edge_error, actual_edge_error).
Edge labels come from Stim's detector_error_model.

References
----------
- Bausch et al. (2023). Learning to Decode the Surface Code with a
  Recurrent, Transformer-Based Neural Network. arXiv:2310.05900
- Overwater et al. (2022). Neural-Network Decoders for Quantum Error
  Correction Using Surface Codes. IEEE TQE 3.
- Higgott & Gidney (2023). Sparse Blossom. arXiv:2303.15933
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
import pymatching
import stim

from qec_ml.decoders.base_decoder import BaseDecoder
from qec_ml.utils.config import QECConfig


# ======================================================================
# Detector Graph Builder
# ======================================================================

class DetectorGraph:
    """
    Represents the MWPM detector graph with learnable edge weights.

    Nodes = detectors (ancilla measurement outcomes).
    Edges = pairs of detectors that a single error can flip simultaneously.

    Parameters
    ----------
    circuit : stim.Circuit
    """

    def __init__(self, circuit: stim.Circuit):
        dem = circuit.detector_error_model(decompose_errors=True)
        self._dem = dem
        self._edges, self._weights, self._obs_masks = self._parse_dem(dem)
        self.n_detectors = dem.num_detectors
        self.n_edges = len(self._edges)

    @property
    def edges(self) -> List[Tuple[int, int]]:
        return self._edges

    @property
    def base_weights(self) -> np.ndarray:
        return np.array(self._weights)

    def build_matching(self, weight_delta: Optional[np.ndarray] = None) -> pymatching.Matching:
        """
        Build a PyMatching object with optionally adjusted weights.

        Parameters
        ----------
        weight_delta : (n_edges,) array, optional
            Additive corrections to log-likelihood-ratio weights.
        """
        m = pymatching.Matching()
        weights = self.base_weights.copy()
        if weight_delta is not None:
            weights = weights + weight_delta
            weights = np.clip(weights, 0.01, 50.0)  # prevent degenerate weights

        for (u, v), w, obs in zip(self._edges, weights, self._obs_masks):
            m.add_edge(u, v, fault_ids=obs, weight=w)
        m.set_boundary_and_detector_index(
            num_detectors=self.n_detectors
        )
        return m

    def _parse_dem(self, dem) -> Tuple[List, List, List]:
        edges, weights, obs_masks = [], [], []
        for inst in dem.flattened():
            if inst.type == "error":
                p = inst.args_copy()[0]
                weight = max(0.01, -np.log(p / (1 - p + 1e-9)))
                dets = []
                obs = []
                for t in inst.targets_copy():
                    if t.is_relative_detector_id():
                        dets.append(t.val)
                    elif t.is_logical_observable_id():
                        obs.append(t.val)
                if len(dets) == 1:
                    dets = [dets[0], self.n_detectors]  # boundary
                if len(dets) >= 2:
                    edges.append((dets[0], dets[1]))
                    weights.append(weight)
                    obs_masks.append(obs)
        return edges, weights, obs_masks


# ======================================================================
# GNN Edge Weight Predictor
# ======================================================================

class EdgeWeightGNN(nn.Module):
    """
    GNN that predicts per-edge weight corrections for the MWPM graph.

    Input:  syndrome vector (B, L)
    Output: weight deltas (B, n_edges)

    The model:
    1. Projects syndrome bits to node features (detector embeddings)
    2. Runs message-passing over the detector graph
    3. For each edge (u, v), predicts Δw from node features at u and v

    Parameters
    ----------
    n_detectors : int
    n_edges : int
    edge_list : list of (int, int)
    syndrome_length : int
    d_model : int
    n_mp_rounds : int     — message passing rounds
    dropout : float
    """

    def __init__(
        self,
        n_detectors: int,
        n_edges: int,
        edge_list: List[Tuple[int, int]],
        syndrome_length: int,
        d_model: int = 64,
        n_mp_rounds: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_det = n_detectors
        self.n_edges = n_edges
        self.d = d_model
        self.n_mp = n_mp_rounds

        # Register edge indices as buffers
        src = torch.tensor([e[0] for e in edge_list], dtype=torch.long)
        dst = torch.tensor([e[1] for e in edge_list], dtype=torch.long)
        self.register_buffer("_src", src)
        self.register_buffer("_dst", dst)

        # Node feature initialisation (syndrome bit → d_model)
        self.node_embed = nn.Embedding(2, d_model)
        self.node_proj  = nn.Linear(d_model, d_model)

        # Message-passing layers
        self.mp_layers = nn.ModuleList([
            _MPLayer(d_model, dropout) for _ in range(n_mp_rounds)
        ])

        # Edge prediction: MLP on concatenated endpoint features
        self.edge_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),   # scalar weight delta per edge
        )

    def forward(self, syndrome: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        syndrome : (B, syndrome_length) — only first n_detectors bits used

        Returns
        -------
        weight_deltas : (B, n_edges)
        """
        B = syndrome.shape[0]
        n = self.n_det

        # Initialise node features from syndrome bits
        s = syndrome[:, :n].long().clamp(0, 1)   # (B, n_det)
        node_feat = self.node_embed(s)             # (B, n_det, d)
        node_feat = self.node_proj(node_feat)

        # Message passing
        for layer in self.mp_layers:
            node_feat = layer(node_feat, self._src, self._dst)

        # Edge features: concatenate endpoint node features
        src_feat = node_feat[:, self._src]         # (B, n_edges, d)
        dst_feat = node_feat[:, self._dst]         # (B, n_edges, d)
        edge_feat = torch.cat([src_feat, dst_feat], dim=-1)  # (B, n_edges, 2d)

        delta = self.edge_head(edge_feat).squeeze(-1)  # (B, n_edges)
        return delta


class _MPLayer(nn.Module):
    """One round of message passing: aggregate neighbour features."""
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.msg = nn.Linear(d, d)
        self.upd = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        B, N, D = h.shape
        # Aggregate messages: for each node, sum messages from its neighbours
        msgs = self.msg(h[:, src])                 # (B, n_edges, D)
        agg = torch.zeros(B, N, D, device=h.device)
        agg.scatter_add_(1, dst[None, :, None].expand(B, -1, D), msgs)
        h_new = self.upd(torch.cat([h, agg], dim=-1))
        return self.norm(h + h_new)


# ======================================================================
# Hybrid GNN-MWPM Decoder
# ======================================================================

class GNNMWPMDecoder(BaseDecoder):
    """
    Hybrid decoder: GNN reweights the MWPM graph, then MWPM decodes.

    Training:
      1. Generate syndromes from Stim (with correlated noise).
      2. Train EdgeWeightGNN to minimise per-edge error probability.
      3. At inference: GNN predicts weight deltas → MWPM decodes.

    Parameters
    ----------
    config : QECConfig
    circuit : stim.Circuit
    d_model : int
    n_mp_rounds : int
    """

    def __init__(
        self,
        config: QECConfig,
        circuit: stim.Circuit,
        d_model: int = 64,
        n_mp_rounds: int = 4,
    ):
        self.config = config
        self.det_graph = DetectorGraph(circuit)
        self.gnn = EdgeWeightGNN(
            n_detectors=self.det_graph.n_detectors,
            n_edges=self.det_graph.n_edges,
            edge_list=self.det_graph.edges,
            syndrome_length=config.syndrome_length,
            d_model=d_model,
            n_mp_rounds=n_mp_rounds,
        )
        self._device = "cpu"

    @property
    def name(self) -> str:
        return "GNN-MWPM (Reweighted)"

    def to(self, device: str) -> "GNNMWPMDecoder":
        self._device = device
        self.gnn = self.gnn.to(device)
        return self

    @torch.no_grad()
    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Decode with GNN-reweighted MWPM."""
        x = torch.from_numpy(syndromes.astype(np.float32)).to(self._device)
        deltas = self.gnn(x).cpu().numpy()  # (N, n_edges)

        preds = np.zeros(len(syndromes), dtype=np.uint8)
        for i, (syn, delta) in enumerate(zip(syndromes, deltas)):
            matcher = self.det_graph.build_matching(weight_delta=delta)
            pred = matcher.decode(syn.astype(bool))
            preds[i] = int(pred[0]) if len(pred) > 0 else 0
        return preds

    def decode(self, syndrome: np.ndarray) -> int:
        return int(self.decode_batch(syndrome[None])[0])

    def gnn_parameters(self):
        return self.gnn.parameters()
