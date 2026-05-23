"""
qec_ml — Machine Learning for Quantum Error Correction
========================================================

A research library for comparing classical and ML-based decoders
on surface codes and analog readout correction.

Modules
-------
data        : syndrome generation, noise models, datasets
decoders    : classical decoders (MWPM, lookup table)
models      : ML decoders (MLP, CNN, GNN, Transformer)
benchmarks  : metrics, runners, visualization
utils       : training loops, config management
"""

from qec_ml.utils.config import QECConfig, NoiseConfig, TrainingConfig

__version__ = "0.1.0"
__all__ = ["QECConfig", "NoiseConfig", "TrainingConfig"]
