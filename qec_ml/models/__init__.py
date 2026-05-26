"""
qec_ml.models
==============
ML decoders and detectors for quantum error correction.

v1 baselines
------------
MLPDecoder, CNNDecoder
SyndromeTransformer, SpatialTemporalTransformer
LSTMClassifier, Conv1DClassifier, IQAutoencoder

v2 improved models
------------------
ResidualMLP         — MLP with residual blocks + syndrome features
SurfaceCodeCNN      — CNN with physically correct (d-1)×d ancilla layout
AncillaTransformer  — Transformer with 2D row/col/round positional encodings
FocalLoss           — loss for imbalanced syndrome datasets

v3 leakage & GNN
-----------------
LeakageDetectorCNN              — spatio-temporal CNN for dark-detector detection
LeakageClassifierTransformer    — multi-task: logical error + leakage jointly
SyndromeAnomalyDetector         — unsupervised autoencoder anomaly detector

Optional (requires torch-geometric)
------------------------------------
GCNDecoder, GATDecoder
"""

# v1
from qec_ml.models.mlp_decoder import MLPDecoder, CNNDecoder
from qec_ml.models.transformer_decoder import SyndromeTransformer, SpatialTemporalTransformer
from qec_ml.models.lstm_corrector import LSTMClassifier, Conv1DClassifier, IQAutoencoder

# v2
from qec_ml.models.mlp_decoder import ResidualMLP, SurfaceCodeCNN, FocalLoss
from qec_ml.models.transformer_decoder import AncillaTransformer

# v3
from qec_ml.models.leakage_detector import (
    LeakageDetectorCNN, LeakageClassifierTransformer, SyndromeAnomalyDetector,
)

# Optional GNN (torch-geometric)
try:
    from qec_ml.models.gnn_decoder import GCNDecoder, GATDecoder
    _GNN_AVAILABLE = True
except ImportError:
    _GNN_AVAILABLE = False

__all__ = [
    "MLPDecoder", "CNNDecoder",
    "SyndromeTransformer", "SpatialTemporalTransformer",
    "LSTMClassifier", "Conv1DClassifier", "IQAutoencoder",
    "ResidualMLP", "SurfaceCodeCNN", "FocalLoss",
    "AncillaTransformer",
    "LeakageDetectorCNN", "LeakageClassifierTransformer", "SyndromeAnomalyDetector",
    "GCNDecoder", "GATDecoder",
]
