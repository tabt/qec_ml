"""
qec_ml.data.noise_models
==========================
Additional noise model utilities for generating synthetic quantum error data
outside of Stim — useful for custom experiments and ablation studies.

Provides:
  - DepolarizingChannel : apply depolarizing noise to a Pauli frame
  - BitFlipChannel      : single-qubit X errors
  - AmplitudeDamping    : approximate T1 errors on syndrome bits
  - PauliFrame          : lightweight Pauli error tracker
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class PauliFrame:
    """
    Lightweight Pauli frame tracker for n qubits.

    Represents the current error state as X and Z Pauli frames.
    """

    def __init__(self, n_qubits: int, rng: Optional[np.random.Generator] = None):
        self.n = n_qubits
        self.rng = rng or np.random.default_rng()
        self.x_frame = np.zeros(n_qubits, dtype=np.uint8)
        self.z_frame = np.zeros(n_qubits, dtype=np.uint8)

    def reset(self):
        self.x_frame[:] = 0
        self.z_frame[:] = 0

    def apply_depolarizing(self, p: float, qubits: Optional[np.ndarray] = None):
        """Apply depolarizing noise to given qubits (default: all)."""
        qs = qubits if qubits is not None else np.arange(self.n)
        r = self.rng.random(len(qs))
        for i, q in enumerate(qs):
            if r[i] < p / 3:       # X error
                self.x_frame[q] ^= 1
            elif r[i] < 2 * p / 3: # Z error
                self.z_frame[q] ^= 1
            elif r[i] < p:          # Y error
                self.x_frame[q] ^= 1
                self.z_frame[q] ^= 1

    def apply_bit_flip(self, p: float, qubits: Optional[np.ndarray] = None):
        """Apply X (bit-flip) errors."""
        qs = qubits if qubits is not None else np.arange(self.n)
        flips = self.rng.random(len(qs)) < p
        for i, q in enumerate(qs):
            if flips[i]:
                self.x_frame[q] ^= 1

    def apply_phase_flip(self, p: float, qubits: Optional[np.ndarray] = None):
        """Apply Z (phase-flip) errors."""
        qs = qubits if qubits is not None else np.arange(self.n)
        flips = self.rng.random(len(qs)) < p
        for i, q in enumerate(qs):
            if flips[i]:
                self.z_frame[q] ^= 1


class SimpleRepetitionCodeSimulator:
    """
    Fast numpy-based simulator for the 1D repetition code.

    Useful for quick experiments without Stim, and for
    illustrating the threshold phenomenon with small codes.

    Parameters
    ----------
    n_bits : int
        Code length (= distance for repetition code).
    p : float
        Bit-flip error rate per physical qubit.
    p_meas : float, optional
        Measurement error rate.
    """

    def __init__(self, n_bits: int, p: float, p_meas: Optional[float] = None,
                 rng: Optional[np.random.Generator] = None):
        self.n = n_bits
        self.p = p
        self.p_meas = p_meas if p_meas is not None else p
        self.rng = rng or np.random.default_rng()

    def generate(self, n_samples: int):
        """
        Generate (syndrome, logical_error) pairs for the repetition code.

        Returns
        -------
        syndromes : (N, n-1) uint8 array — parity check outcomes
        logical_errors : (N,) uint8 array — majority vote error
        """
        # Random bit-flip errors
        errors = self.rng.random((n_samples, self.n)) < self.p
        # Noiseless syndrome: XOR of adjacent bits
        syndrome_noiseless = (errors[:, :-1] ^ errors[:, 1:]).astype(np.uint8)
        # Add measurement noise
        meas_noise = self.rng.random(syndrome_noiseless.shape) < self.p_meas
        syndromes = (syndrome_noiseless ^ meas_noise).astype(np.uint8)
        # Logical error: majority vote — total errors > n//2
        logical_errors = (errors.sum(axis=1) > self.n // 2).astype(np.uint8)
        return syndromes, logical_errors

    def mwpm_decode(self, syndromes: np.ndarray) -> np.ndarray:
        """
        Greedy minimum-weight matching for the repetition code.

        Finds syndrome bit positions and pairs adjacent ones.
        """
        predictions = np.zeros(len(syndromes), dtype=np.uint8)
        for i, s in enumerate(syndromes):
            positions = np.where(s == 1)[0]
            # Count total correction flips using greedy pairing
            total_flips = 0
            pos = list(positions)
            while len(pos) >= 2:
                total_flips += pos[1] - pos[0] + 1
                pos = pos[2:]
            predictions[i] = total_flips % 2
        return predictions
