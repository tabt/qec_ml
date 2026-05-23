"""
qec_ml.decoders.base_decoder
==============================
Abstract base class that all decoders (classical and ML) must implement.
Provides a unified interface for the benchmark runner.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import numpy as np


class BaseDecoder(ABC):
    """
    Abstract decoder interface.

    Every decoder — whether MWPM, Lookup Table, MLP, or Transformer —
    must implement `decode_batch`, `fit` (optional), and expose `name`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable decoder name."""
        ...

    def fit(self, syndromes: np.ndarray, labels: np.ndarray, **kwargs) -> None:
        """
        Train the decoder on labelled syndrome data.
        Classical decoders may leave this as a no-op.
        """
        pass  # default: no training needed

    @abstractmethod
    def decode(self, syndrome: np.ndarray) -> int:
        """
        Decode a single syndrome vector.

        Parameters
        ----------
        syndrome : 1-D uint8 array of length syndrome_length

        Returns
        -------
        int : 0 (no logical error predicted) or 1 (logical error predicted)
        """
        ...

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """
        Decode a batch of syndromes.  Override for vectorized implementations.

        Parameters
        ----------
        syndromes : (N, syndrome_length) uint8 array

        Returns
        -------
        np.ndarray of shape (N,) with values in {0, 1}
        """
        return np.array([self.decode(s) for s in syndromes], dtype=np.uint8)

    # ------------------------------------------------------------------
    # Shared evaluation helpers
    # ------------------------------------------------------------------

    def evaluate(
        self,
        syndromes: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run the decoder on a test set and collect metrics.

        Returns
        -------
        dict with keys: logical_error_rate, accuracy, decoding_time_ms_per_shot
        """
        t0 = time.perf_counter()
        preds = self.decode_batch(syndromes)
        elapsed = time.perf_counter() - t0

        n = len(labels)
        logical_error_rate = float(np.mean(preds != labels))
        accuracy = 1.0 - logical_error_rate
        time_per_shot_ms = (elapsed / n) * 1000

        return {
            "decoder": self.name,
            "logical_error_rate": logical_error_rate,
            "accuracy": accuracy,
            "decoding_time_ms_per_shot": time_per_shot_ms,
            "n_shots": n,
        }
