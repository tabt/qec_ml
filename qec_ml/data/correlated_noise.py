"""
qec_ml.data.correlated_noise
==============================
Correlated noise models beyond IID depolarizing.

Three models are implemented:

1. **SpatiallyCorrelatedNoise** — errors on nearby qubits are
   correlated (e.g., crosstalk, two-qubit gate errors, cosmic rays).
   Modeled via a Gaussian spatial correlation kernel applied to
   independent error samples.

2. **BurstNoiseModel** — rare high-error events (cosmic rays,
   control electronics glitches) hit multiple qubits simultaneously.
   Standard MWPM assigns these very high weight and often fails.

3. **TemporallyCorrelatedNoise** — error probability at round t
   depends on whether errors occurred at round t-1 (non-Markovian,
   models TLS fluctuators and charge noise).

Why correlated noise hurts MWPM
---------------------------------
MWPM's edge weights are calibrated assuming independent errors.
When errors cluster (spatially or temporally), the most likely
correction is no longer the minimum-weight one.  ML models that
observe the full syndrome pattern can recognise these clusters.
"""

from __future__ import annotations

import numpy as np
import stim
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal
from scipy.ndimage import gaussian_filter

from qec_ml.utils.config import QECConfig
from qec_ml.data.syndrome_generator import SyndromeDataset


@dataclass
class CorrelatedNoiseConfig:
    """Parameters for correlated noise injection."""
    mode: Literal["spatial", "burst", "temporal"] = "spatial"

    # Spatial correlation
    spatial_sigma: float = 1.5      # correlation length in qubit units
    spatial_scale: float = 2.0      # amplification factor for correlated errors

    # Burst (cosmic ray) model
    burst_rate: float = 0.002       # probability of a burst event per shot
    burst_radius: int = 2           # qubits affected around epicenter
    burst_p_error: float = 0.5      # error prob inside burst zone

    # Temporal correlation
    temporal_p_persist: float = 0.4  # prob that error persists to next round
    temporal_p_new: float = 0.005    # base rate without persistence

    seed: int = 42


@dataclass
class CorrelatedSyndromeDataset:
    """Syndrome dataset with correlated noise labels."""
    syndromes: np.ndarray           # (N, L) uint8
    logical_errors: np.ndarray      # (N,) uint8
    correlation_labels: np.ndarray  # (N,) uint8 — 1 if correlated event occurred
    config: QECConfig
    noise_config: CorrelatedNoiseConfig
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self): return len(self.syndromes)

    def split(self, train=0.7, val=0.15):
        n = len(self)
        idx = np.random.permutation(n)
        i1, i2 = int(n * train), int(n * (train + val))
        def _sub(a, b):
            sl = idx[a:b]
            return CorrelatedSyndromeDataset(
                syndromes=self.syndromes[sl],
                logical_errors=self.logical_errors[sl],
                correlation_labels=self.correlation_labels[sl],
                config=self.config, noise_config=self.noise_config,
            )
        return _sub(0, i1), _sub(i1, i2), _sub(i2, n)


class CorrelatedNoiseGenerator:
    """
    Generates syndrome data with controlled correlated noise.

    Strategy: use Stim for the circuit structure, then post-process
    the raw error frame to inject spatial/burst/temporal correlations
    before computing the syndrome.

    Parameters
    ----------
    config : QECConfig
    noise_config : CorrelatedNoiseConfig

    Examples
    --------
    >>> cfg = QECConfig(distance=5, noise=NoiseConfig('depolarizing', 0.005, rounds=5))
    >>> ncfg = CorrelatedNoiseConfig(mode='burst', burst_rate=0.003)
    >>> gen = CorrelatedNoiseGenerator(cfg, ncfg)
    >>> ds = gen.generate(n_samples=20_000)
    """

    def __init__(self, config: QECConfig, noise_config: CorrelatedNoiseConfig):
        self.cfg = config
        self.ncfg = noise_config
        self.rng = np.random.default_rng(noise_config.seed)

    def generate(self, n_samples: int) -> CorrelatedSyndromeDataset:
        """Generate n_samples syndrome shots with correlated noise."""
        mode = self.ncfg.mode
        if mode == "spatial":
            return self._generate_spatial(n_samples)
        elif mode == "burst":
            return self._generate_burst(n_samples)
        elif mode == "temporal":
            return self._generate_temporal(n_samples)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # ------------------------------------------------------------------
    # Spatial correlation
    # ------------------------------------------------------------------

    def _generate_spatial(self, n: int) -> CorrelatedSyndromeDataset:
        """
        IID errors smoothed through a Gaussian kernel, then thresholded.
        Creates spatially clustered error patterns.
        """
        d = self.cfg.distance
        R = self.cfg.noise.rounds
        p = self.cfg.noise.p

        syndromes_list, labels, corr_labels = [], [], []

        for _ in range(n):
            # Independent errors on d×d grid
            error_field = self.rng.random((d, d)) < p

            # Spatial smoothing → correlated errors
            smoothed = gaussian_filter(error_field.astype(float),
                                       sigma=self.ncfg.spatial_sigma)
            threshold = p * self.ncfg.spatial_scale
            corr_errors = smoothed > threshold

            # Convert error pattern to syndrome via parity checks
            syndrome, log_err = self._errors_to_syndrome(corr_errors, d, R)
            syndromes_list.append(syndrome)
            labels.append(log_err)
            corr_labels.append(1 if corr_errors.sum() > error_field.sum() * 1.5 else 0)

        return CorrelatedSyndromeDataset(
            syndromes=np.array(syndromes_list, dtype=np.uint8),
            logical_errors=np.array(labels, dtype=np.uint8),
            correlation_labels=np.array(corr_labels, dtype=np.uint8),
            config=self.cfg, noise_config=self.ncfg,
            metadata={"mode": "spatial", "sigma": self.ncfg.spatial_sigma},
        )

    # ------------------------------------------------------------------
    # Burst (cosmic ray) noise
    # ------------------------------------------------------------------

    def _generate_burst(self, n: int) -> CorrelatedSyndromeDataset:
        """
        Rare burst events hit a cluster of qubits simultaneously.
        Each burst: random epicenter, radius r, high error prob inside.
        """
        d = self.cfg.distance
        R = self.cfg.noise.rounds
        p = self.cfg.noise.p

        syndromes_list, labels, corr_labels = [], [], []

        for _ in range(n):
            error_field = self.rng.random((d, d)) < p
            is_burst = 0

            if self.rng.random() < self.ncfg.burst_rate:
                # Burst event: hit cluster around random epicenter
                cx = self.rng.integers(0, d)
                cy = self.rng.integers(0, d)
                r = self.ncfg.burst_radius
                for i in range(max(0, cx - r), min(d, cx + r + 1)):
                    for j in range(max(0, cy - r), min(d, cy + r + 1)):
                        if self.rng.random() < self.ncfg.burst_p_error:
                            error_field[i, j] = True
                is_burst = 1

            syndrome, log_err = self._errors_to_syndrome(error_field, d, R)
            syndromes_list.append(syndrome)
            labels.append(log_err)
            corr_labels.append(is_burst)

        return CorrelatedSyndromeDataset(
            syndromes=np.array(syndromes_list, dtype=np.uint8),
            logical_errors=np.array(labels, dtype=np.uint8),
            correlation_labels=np.array(corr_labels, dtype=np.uint8),
            config=self.cfg, noise_config=self.ncfg,
            metadata={"mode": "burst", "burst_rate": self.ncfg.burst_rate},
        )

    # ------------------------------------------------------------------
    # Temporal correlation
    # ------------------------------------------------------------------

    def _generate_temporal(self, n: int) -> CorrelatedSyndromeDataset:
        """
        Errors at round r are correlated with errors at round r-1
        (TLS fluctuators, charge noise).
        """
        d = self.cfg.distance
        R = self.cfg.noise.rounds
        p_new = self.ncfg.temporal_p_new
        p_persist = self.ncfg.temporal_p_persist

        syndromes_list, labels, corr_labels = [], [], []

        for _ in range(n):
            all_round_syndromes = []
            prev_errors = np.zeros((d, d), dtype=bool)
            had_temporal_event = 0

            for r in range(R):
                # Errors: persist from last round + new
                persist = prev_errors & (self.rng.random((d, d)) < p_persist)
                new_errs = self.rng.random((d, d)) < p_new
                errors = persist | new_errs

                if persist.any():
                    had_temporal_event = 1

                # One-round syndrome (parity of adjacent pairs)
                syn_r = self._single_round_syndrome(errors, d)
                all_round_syndromes.append(syn_r)
                prev_errors = errors

            syndrome = np.concatenate(all_round_syndromes)
            # Logical error: total X errors on top row > d//2
            log_err = int(errors[0, :].sum() > d // 2)
            syndromes_list.append(syndrome)
            labels.append(log_err)
            corr_labels.append(had_temporal_event)

        return CorrelatedSyndromeDataset(
            syndromes=np.array(syndromes_list, dtype=np.uint8),
            logical_errors=np.array(labels, dtype=np.uint8),
            correlation_labels=np.array(corr_labels, dtype=np.uint8),
            config=self.cfg, noise_config=self.ncfg,
            metadata={"mode": "temporal", "p_persist": self.ncfg.temporal_p_persist},
        )

    # ------------------------------------------------------------------
    # Helpers: error pattern → syndrome
    # ------------------------------------------------------------------

    def _errors_to_syndrome(self, errors: np.ndarray, d: int, R: int):
        """Convert a 2D error grid to a multi-round syndrome vector."""
        all_rounds = []
        for _ in range(R):
            # Add small per-round fluctuation
            fluc = self.rng.random((d, d)) < self.cfg.noise.p / 3
            round_errors = errors ^ fluc
            all_rounds.append(self._single_round_syndrome(round_errors, d))
        syndrome = np.concatenate(all_rounds)
        log_err = int(errors[0, :].sum() > d // 2)  # simplified logical Z
        return syndrome.astype(np.uint8), log_err

    def _single_round_syndrome(self, errors: np.ndarray, d: int) -> np.ndarray:
        """Compute Z-stabiliser syndrome for one round of errors."""
        syndrome = []
        for r in range(d - 1):
            for c in range(d):
                # Z-stabiliser at (r,c): parity of errors[r,c] and errors[r+1,c]
                parity = int(errors[r, c]) ^ int(errors[r + 1, c])
                # Add measurement noise
                if self.rng.random() < self.cfg.noise.p_meas:
                    parity ^= 1
                syndrome.append(parity)
        return np.array(syndrome, dtype=np.uint8)
