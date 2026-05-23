"""
qec_ml.data.analog_signal
==========================
Simulation of analog IQ (In-phase / Quadrature) readout signals from
superconducting qubits, and a dataset builder for ML-based readout
classification and denoising.

Physical background
-------------------
In superconducting qubit experiments, the dispersive readout produces
an IQ point whose position in the complex plane depends on the qubit
state |0⟩ or |1⟩.  Real devices exhibit:
  - Gaussian shot noise (Johnson-Nyquist + amplifier noise)
  - State relaxation (T1 decay) during measurement
  - Measurement-induced state transitions (confusion matrix)
  - Readout crosstalk for multi-qubit systems

References
----------
- Krantz et al. (2019). A quantum engineer's guide to superconducting
  qubits. Applied Physics Reviews 6, 021318.
- Magesan et al. (2015). Machine Learning for Discriminating Quantum
  Measurement Trajectories. PRL 114, 200501.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from scipy.ndimage import gaussian_filter1d


@dataclass
class ReadoutConfig:
    """Parameters describing a dispersive readout setup."""

    # IQ-plane positions of |0⟩ and |1⟩ centroids
    iq_0: Tuple[float, float] = (1.0, 0.0)    # I, Q for |0⟩
    iq_1: Tuple[float, float] = (-1.0, 0.0)   # I, Q for |1⟩

    # Gaussian noise (equal for I and Q channels)
    sigma_noise: float = 0.4

    # T1 relaxation: probability |1⟩→|0⟩ during measurement window
    t1_error_prob: float = 0.02

    # State preparation error
    state_prep_error: float = 0.005

    # Time series: number of time bins per shot
    n_time_bins: int = 100

    # Bandwidth of the resonator (affects temporal profile)
    kappa_fraction: float = 0.1   # fraction of n_time_bins for rise time

    n_qubits: int = 1
    seed: int = 42


@dataclass
class AnalogDataset:
    """Dataset of analog IQ trajectories with ground-truth labels."""

    # (N, n_qubits, n_time_bins, 2)  — last dim: [I, Q]
    trajectories: np.ndarray

    # (N, n_qubits)  — true prepared state (0 or 1)
    true_states: np.ndarray

    # (N, n_qubits)  — threshold-based readout result
    threshold_readout: np.ndarray

    config: ReadoutConfig

    def __len__(self) -> int:
        return len(self.trajectories)

    @property
    def integrated_iq(self) -> np.ndarray:
        """Time-integrated IQ point, shape (N, n_qubits, 2)."""
        return self.trajectories.mean(axis=2)

    @property
    def threshold_accuracy(self) -> float:
        return np.mean(self.threshold_readout == self.true_states)


class AnalogSignalSimulator:
    """
    Simulates superconducting qubit dispersive readout IQ trajectories.

    The simulator generates:
      1. Static IQ points — for simple threshold / ML classification.
      2. Time-series IQ trajectories — for temporal ML models (LSTM, Conv1D).

    Parameters
    ----------
    config : ReadoutConfig

    Examples
    --------
    >>> cfg = ReadoutConfig(sigma_noise=0.35, t1_error_prob=0.03)
    >>> sim = AnalogSignalSimulator(cfg)
    >>> ds = sim.generate(n_samples=5000, state_fractions=[0.5, 0.5])
    >>> print(ds.trajectories.shape)
    (5000, 1, 100, 2)
    """

    def __init__(self, config: ReadoutConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        n_samples: int,
        state_fractions: Optional[List[float]] = None,
    ) -> AnalogDataset:
        """
        Generate `n_samples` IQ trajectory shots.

        Parameters
        ----------
        n_samples : int
        state_fractions : list of float, optional
            Fraction of shots in |0⟩ and |1⟩.  Defaults to [0.5, 0.5].
        """
        cfg = self.config
        if state_fractions is None:
            state_fractions = [0.5, 0.5]

        # True prepared states (before prep error)
        ideal_states = self.rng.choice(
            [0, 1], size=(n_samples, cfg.n_qubits),
            p=state_fractions
        )

        # State preparation errors
        prep_flip = self.rng.random((n_samples, cfg.n_qubits)) < cfg.state_prep_error
        true_states = np.where(prep_flip, 1 - ideal_states, ideal_states)

        # T1 relaxation during measurement: |1⟩ may decay to |0⟩
        t1_flip = (true_states == 1) & (
            self.rng.random((n_samples, cfg.n_qubits)) < cfg.t1_error_prob
        )
        measured_states = np.where(t1_flip, 0, true_states)

        # Build time-series trajectories
        trajectories = self._build_trajectories(measured_states)

        # Threshold readout on integrated signal
        integrated = trajectories.mean(axis=2)          # (N, Q, 2)
        c0 = np.array(cfg.iq_0)
        c1 = np.array(cfg.iq_1)
        dist0 = np.linalg.norm(integrated - c0, axis=-1)
        dist1 = np.linalg.norm(integrated - c1, axis=-1)
        threshold_readout = (dist1 < dist0).astype(np.uint8)

        return AnalogDataset(
            trajectories=trajectories,
            true_states=true_states.astype(np.uint8),
            threshold_readout=threshold_readout,
            config=cfg,
        )

    def add_crosstalk(
        self, dataset: AnalogDataset, crosstalk_matrix: Optional[np.ndarray] = None
    ) -> AnalogDataset:
        """
        Simulate linear IQ crosstalk between qubits.

        Parameters
        ----------
        crosstalk_matrix : (n_qubits, n_qubits) array
            Off-diagonal elements are leakage fractions.  Identity = no crosstalk.
        """
        cfg = self.config
        if crosstalk_matrix is None:
            eps = 0.05
            cm = np.eye(cfg.n_qubits) + eps * (1 - np.eye(cfg.n_qubits))
        else:
            cm = crosstalk_matrix

        # Apply linear mixing along qubit axis
        traj = dataset.trajectories  # (N, Q, T, 2)
        mixed = np.einsum("qr,nrTc->nqTc", cm, traj)
        return AnalogDataset(
            trajectories=mixed,
            true_states=dataset.true_states,
            threshold_readout=dataset.threshold_readout,
            config=cfg,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_trajectories(self, states: np.ndarray) -> np.ndarray:
        """
        Build (N, n_qubits, n_time_bins, 2) array of IQ trajectories.

        The trajectory shape is modeled as a smooth step function
        (exponential rise) convolved with Gaussian shot noise.
        """
        cfg = self.config
        n, q = states.shape
        T = cfg.n_time_bins

        # IQ centroids per qubit per shot: (N, Q, 2)
        iq0 = np.array(cfg.iq_0)
        iq1 = np.array(cfg.iq_1)
        centroids = np.where(
            states[:, :, None] == 0, iq0[None, None, :], iq1[None, None, :]
        )  # (N, Q, 2)

        # Temporal envelope: exponential rise
        rise_time = int(cfg.kappa_fraction * T)
        t = np.arange(T)
        envelope = 1 - np.exp(-t / max(rise_time, 1))  # (T,)
        envelope = envelope[None, None, :, None]         # (1, 1, T, 1)

        # Broadcast centroids across time
        signal = centroids[:, :, None, :] * envelope    # (N, Q, T, 2)

        # Add Gaussian noise
        noise = self.rng.normal(
            scale=cfg.sigma_noise, size=(n, q, T, 2)
        )
        trajectories = signal + noise

        # Smooth with low-pass filter (mimics finite bandwidth)
        if rise_time > 1:
            trajectories = gaussian_filter1d(trajectories, sigma=rise_time / 4, axis=2)

        return trajectories.astype(np.float32)
