"""
qec_ml.data.syndrome_generator
================================
Generates surface-code syndrome data using the Stim simulator.

Supports:
  - Rotated surface code (standard)
  - Multiple noise models (depolarizing, bit-flip, circuit-level)
  - Multi-round syndrome measurements
  - Both X- and Z-type stabilizer syndromes

References
----------
- Gidney, C. (2021). Stim: a fast stabilizer circuit simulator.
  Quantum 5, 497. https://doi.org/10.22331/q-2021-07-06-497
- Google Quantum AI (2022). Suppressing quantum errors by scaling a
  surface code logical qubit. Nature 614, 676–681.
"""

from __future__ import annotations

import numpy as np
import stim
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from qec_ml.utils.config import QECConfig, NoiseConfig


@dataclass
class SyndromeDataset:
    """Container for a generated syndrome dataset."""

    syndromes: np.ndarray          # shape (N, syndrome_length), dtype uint8
    logical_errors: np.ndarray     # shape (N,), dtype uint8 — 0 or 1
    observables: np.ndarray        # shape (N, n_observables), dtype uint8
    config: QECConfig
    metadata: Dict[str, Any]

    def __len__(self) -> int:
        return len(self.syndromes)

    def split(self, train: float = 0.7, val: float = 0.15) -> Tuple[
        "SyndromeDataset", "SyndromeDataset", "SyndromeDataset"
    ]:
        """Split into train/val/test by fraction."""
        n = len(self)
        i1 = int(n * train)
        i2 = int(n * (train + val))
        idx = np.arange(n)
        np.random.shuffle(idx)

        def _subset(i, j):
            return SyndromeDataset(
                syndromes=self.syndromes[idx[i:j]],
                logical_errors=self.logical_errors[idx[i:j]],
                observables=self.observables[idx[i:j]],
                config=self.config,
                metadata={**self.metadata, "split": f"{i}:{j}"},
            )

        return _subset(0, i1), _subset(i1, i2), _subset(i2, n)


class SyndromeGenerator:
    """
    Generates syndrome + logical-error pairs for a rotated surface code
    using Stim as the underlying simulator.

    Parameters
    ----------
    config : QECConfig
        Full QEC configuration (distance, noise, sample counts).

    Examples
    --------
    >>> cfg = QECConfig(distance=5, noise=NoiseConfig(p=0.01, rounds=5))
    >>> gen = SyndromeGenerator(cfg)
    >>> dataset = gen.generate(n_samples=10_000)
    >>> print(dataset.syndromes.shape)
    (10000, 120)
    """

    def __init__(self, config: QECConfig):
        self.config = config
        self._circuit: Optional[stim.Circuit] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, n_samples: int, seed: Optional[int] = None) -> SyndromeDataset:
        """Sample `n_samples` syndrome / logical-error pairs."""
        circuit = self._build_circuit()
        sampler = circuit.compile_detector_sampler(seed=seed or self.config.seed)

        # stim returns (detections, observables) arrays
        detections, observables = sampler.sample(
            shots=n_samples, separate_observables=True
        )

        # detections: bool array (N, n_detectors)
        syndromes = detections.astype(np.uint8)

        # observables: bool array (N, n_observables) — logical error flags
        logical_errors = observables[:, 0].astype(np.uint8)

        return SyndromeDataset(
            syndromes=syndromes,
            logical_errors=logical_errors,
            observables=observables.astype(np.uint8),
            config=self.config,
            metadata={
                "distance": self.config.distance,
                "noise_model": self.config.noise.model,
                "p": self.config.noise.p,
                "rounds": self.config.noise.rounds,
                "n_detectors": syndromes.shape[1],
                "n_observables": observables.shape[1],
            },
        )

    def get_circuit(self) -> stim.Circuit:
        """Return the Stim circuit (build if needed)."""
        return self._build_circuit()

    def logical_error_rate_mwpm(self, n_samples: int = 10_000) -> float:
        """Quick sanity check: MWPM logical error rate on generated data."""
        import pymatching
        circuit = self._build_circuit()
        model = circuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(model)

        sampler = circuit.compile_detector_sampler(seed=self.config.seed)
        detections, observables = sampler.sample(
            shots=n_samples, separate_observables=True
        )
        predictions = matcher.decode_batch(detections)
        n_errors = np.sum(predictions[:, 0] != observables[:, 0])
        return n_errors / n_samples

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def _build_circuit(self) -> stim.Circuit:
        if self._circuit is not None:
            return self._circuit
        self._circuit = self._make_circuit()
        return self._circuit

    def _make_circuit(self) -> stim.Circuit:
        """Build a Stim surface code circuit for the configured noise model."""
        d = self.config.distance
        p = self.config.noise.p
        p_meas = self.config.noise.p_meas
        rounds = self.config.noise.rounds
        model = self.config.noise.model

        if model == "depolarizing":
            return stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                rounds=rounds,
                distance=d,
                after_clifford_depolarization=p,
                after_reset_flip_probability=p,
                before_measure_flip_probability=p_meas,
                before_round_data_depolarization=p,
            )
        elif model == "bit_flip":
            return stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                rounds=rounds,
                distance=d,
                before_round_data_depolarization=0,
                after_clifford_depolarization=0,
                after_reset_flip_probability=p,
                before_measure_flip_probability=p_meas,
            )
        elif model == "circuit_level":
            # Circuit-level noise: every gate location gets noise
            return stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                rounds=rounds,
                distance=d,
                after_clifford_depolarization=p,
                after_reset_flip_probability=p,
                before_measure_flip_probability=p_meas,
                before_round_data_depolarization=p / 10,
            )
        else:
            raise ValueError(f"Unknown noise model: {model!r}. "
                             f"Choose from: depolarizing, bit_flip, circuit_level")
