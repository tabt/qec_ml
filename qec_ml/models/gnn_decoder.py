"""
qec_ml.models.gnn_decoder
===========================
Graph Neural Network decoder for surface-code syndromes.

Motivation
----------
The surface code has a natural graph structure: data qubits and ancilla
qubits form a lattice, and errors propagate along edges of this graph.
A GNN can exploit this inductive bias more explicitly than a CNN or MLP.

Architecture
------------
We represent each syndrome as a graph where:
  - Nodes = ancilla qubits (stabilisers) with binary feature = syndrome bit
  - Edges = adjacency on the syndrome graph (two ancillas are adjacent if
    they share a data qubit)
  - Optional: virtual "boundary" nodes for open boundary conditions

The GNN uses several layers of message passing (GCNConv or GATConv),
followed by a global readout (sum / mean pooling) and an MLP head.

Requirements
------------
    pip install torch-geometric

References
----------
- Nautrup et al. (2019). Optimizing Quantum Error Correction Codes
  with Reinforcement Learning. npj Quantum Information 5, 1.
- Overwater et al. (2022). Neural-Network Decoders for Quantum Error
  Correction Using Surface Codes. IEEE TQE 3.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

# GNN imports — only available if torch-geometric is installed
try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
    from torch_geometric.data import Data, Batch
    HAS_PYGEOMETRIC = True
except ImportError:
    HAS_PYGEOMETRIC = False


def _require_pygeometric():
    if not HAS_PYGEOMETRIC:
        raise ImportError(
            "torch-geometric is required for GNNDecoder. "
            "Install with: pip install torch-geometric"
        )


# ======================================================================
# Syndrome graph builder
# ======================================================================

class SyndromeGraphBuilder:
    """
    Builds the adjacency structure for a rotated surface code syndrome graph.

    For a distance-d code:
      - (d-1)*d Z-stabiliser ancillas arranged in a grid
      - Two ancillas are adjacent if they share at least one data qubit
      - Boundary nodes can be added for open boundaries

    Parameters
    ----------
    distance : int
    add_boundary : bool
        If True, add virtual boundary nodes connected to boundary ancillas.
    """

    def __init__(self, distance: int, add_boundary: bool = False):
        self.d = distance
        self.add_boundary = add_boundary
        self._edge_index: Optional[torch.Tensor] = None

    def build_edge_index(self) -> torch.Tensor:
        """
        Build edge_index tensor for the syndrome graph.

        Returns
        -------
        edge_index : (2, E) long tensor
        """
        if self._edge_index is not None:
            return self._edge_index

        d = self.d
        n_anc = (d - 1) * d  # Z-stabilisers in rotated code

        edges = []
        # Grid adjacency: ancilla at (r, c) is adjacent to (r±1, c) and (r, c±1)
        for r in range(d - 1):
            for c in range(d):
                idx = r * d + c
                if r + 1 < d - 1:
                    edges.append((idx, (r + 1) * d + c))
                if c + 1 < d:
                    edges.append((idx, r * d + c + 1))

        # Make undirected
        src = [e[0] for e in edges] + [e[1] for e in edges]
        dst = [e[1] for e in edges] + [e[0] for e in edges]

        if self.add_boundary:
            # Add one boundary node per open edge
            bnd_node = n_anc
            for r in range(d - 1):
                edges.append((r * d, bnd_node))
                edges.append((bnd_node, r * d))
            n_anc += 1

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        self._edge_index = edge_index
        return edge_index

    def syndrome_to_graph(
        self,
        syndrome: torch.Tensor,
        batch_size: int,
    ) -> "Batch":
        """
        Convert a batch of syndrome vectors to a PyG Batch object.

        Parameters
        ----------
        syndrome : (B, n_ancilla) tensor of {0,1}
        batch_size : int

        Returns
        -------
        PyG Batch
        """
        _require_pygeometric()
        edge_index = self.build_edge_index().to(syndrome.device)
        graphs = []
        for i in range(batch_size):
            x = syndrome[i].float().unsqueeze(-1)  # (n_ancilla, 1)
            g = Data(x=x, edge_index=edge_index)
            graphs.append(g)
        return Batch.from_data_list(graphs)


# ======================================================================
# GNN Decoder models
# ======================================================================

class GCNDecoder(nn.Module):
    """
    Graph Convolutional Network decoder (Kipf & Welling 2017).

    Parameters
    ----------
    n_ancilla : int
        Number of ancilla nodes (= syndrome_length for single-round).
    hidden_dim : int
    n_layers : int
    dropout : float
    distance : int
    """

    def __init__(
        self,
        n_ancilla: int,
        hidden_dim: int = 64,
        n_layers: int = 4,
        dropout: float = 0.1,
        distance: int = 5,
        add_boundary: bool = False,
    ):
        _require_pygeometric()
        super().__init__()
        self.graph_builder = SyndromeGraphBuilder(distance, add_boundary)

        self.input_proj = nn.Linear(1, hidden_dim)
        self.convs = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, syndrome: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        syndrome : (B, n_ancilla) float/int tensor

        Returns
        -------
        logits : (B,)
        """
        B = syndrome.shape[0]
        batch = self.graph_builder.syndrome_to_graph(syndrome.float(), B)
        x, edge_index, batch_vec = batch.x, batch.edge_index, batch.batch

        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(x + self.dropout(F.gelu(conv(x, edge_index))))

        x = global_mean_pool(x, batch_vec)   # (B, hidden_dim)
        return self.head(x).squeeze(-1)


class GATDecoder(nn.Module):
    """
    Graph Attention Network decoder.  Uses multi-head attention over
    edges, allowing the model to weight neighbours differently.

    Parameters
    ----------
    n_ancilla, hidden_dim, n_layers, dropout, distance : see GCNDecoder
    n_heads : int
        Number of attention heads per GAT layer.
    """

    def __init__(
        self,
        n_ancilla: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        distance: int = 5,
        add_boundary: bool = False,
    ):
        _require_pygeometric()
        super().__init__()
        self.graph_builder = SyndromeGraphBuilder(distance, add_boundary)
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        self.input_proj = nn.Linear(1, hidden_dim)
        self.convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // n_heads, heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, syndrome: torch.Tensor) -> torch.Tensor:
        B = syndrome.shape[0]
        batch = self.graph_builder.syndrome_to_graph(syndrome.float(), B)
        x, edge_index, batch_vec = batch.x, batch.edge_index, batch.batch

        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(x + self.dropout(F.gelu(conv(x, edge_index))))

        x = global_mean_pool(x, batch_vec)
        return self.head(x).squeeze(-1)
