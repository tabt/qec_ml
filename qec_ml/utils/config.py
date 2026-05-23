"""
qec_ml.utils.config
====================
Centralized configuration via Python dataclasses.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class NoiseConfig:
    """Parameters for the quantum noise model."""

    model: Literal["depolarizing", "bit_flip", "phase_flip", "amplitude_damping", "circuit_level"] = "depolarizing"
    p: float = 0.01                  # Physical error rate
    p_meas: Optional[float] = None  # Measurement error rate (defaults to p if None)
    rounds: int = 1                  # Number of syndrome measurement rounds

    def __post_init__(self):
        if self.p_meas is None:
            self.p_meas = self.p
        assert 0 < self.p < 0.5, "Physical error rate must be in (0, 0.5)"


@dataclass
class QECConfig:
    """Surface code / QEC problem configuration."""

    distance: int = 5                # Code distance d (d×d surface code)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    n_samples_train: int = 50_000
    n_samples_val: int = 10_000
    n_samples_test: int = 20_000
    seed: int = 42

    @property
    def n_data_qubits(self) -> int:
        return self.distance ** 2

    @property
    def n_ancilla_qubits(self) -> int:
        return self.distance ** 2 - 1

    @property
    def syndrome_length(self) -> int:
        """Total syndrome bits = ancillas × rounds."""
        return self.n_ancilla_qubits * self.noise.rounds


@dataclass
class TrainingConfig:
    """Hyperparameters for ML model training."""

    model_type: Literal["mlp", "cnn", "gnn", "transformer", "lstm"] = "transformer"
    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    scheduler: Literal["cosine", "step", "none"] = "cosine"
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    device: str = "auto"            # "auto" | "cpu" | "cuda" | "mps"
    num_workers: int = 4
    early_stopping_patience: int = 10
    checkpoint_dir: str = "checkpoints"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
