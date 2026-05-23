"""
qec_ml.models
==============
ML decoders for quantum error correction.

Available models
----------------
MLP
    SyndromeTransformer — Transformer encoder over syndrome bits
    SpatialTemporalTransformer — 2D spatial + temporal Transformer
    MLPDecoder — fully-connected baseline
    CNNDecoder — 2D-CNN with residual blocks
    GCNDecoder — Graph Convolutional Network (requires torch-geometric)
    GATDecoder — Graph Attention Network (requires torch-geometric)
    LSTMClassifier — Bidirectional LSTM for IQ time series
    Conv1DClassifier — Dilated 1D-CNN for IQ time series
    IQAutoencoder — Convolutional autoencoder for IQ denoising
"""

from qec_ml.models.mlp_decoder import MLPDecoder, CNNDecoder
from qec_ml.models.transformer_decoder import SyndromeTransformer, SpatialTemporalTransformer
from qec_ml.models.lstm_corrector import LSTMClassifier, Conv1DClassifier, IQAutoencoder

try:
    from qec_ml.models.gnn_decoder import GCNDecoder, GATDecoder
    _GNN_AVAILABLE = True
except ImportError:
    _GNN_AVAILABLE = False

__all__ = [
    "MLPDecoder",
    "CNNDecoder",
    "SyndromeTransformer",
    "SpatialTemporalTransformer",
    "LSTMClassifier",
    "Conv1DClassifier",
    "IQAutoencoder",
    "GCNDecoder",
    "GATDecoder",
]

# v2 improved models
from qec_ml.models.mlp_decoder import ResidualMLP, SurfaceCodeCNN, FocalLoss
from qec_ml.models.transformer_decoder import AncillaTransformer
