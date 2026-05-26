"""
qec_ml.data.leakage_noise
==========================
Leakage error simulation for superconducting qubits.

Physical background
-------------------
Real transmon qubits are weakly anharmonic oscillators with levels
|0⟩, |1⟩, |2⟩, |3⟩, ...  During gate operations, population can
escape from the computational subspace {|0⟩, |1⟩} into |2⟩ — this
is called *leakage*.

Why MWPM fails on leakage
--------------------------
MWPM assumes a Pauli error model: every qubit is either correct or
has an X, Y, or Z error — always staying in {|0⟩, |1⟩}.  A leaked
qubit |2⟩ behaves as a "frozen" qubit that *never* triggers adjacent
stabilisers, producing a characteristic pattern of persistently DARK
detectors (no syndrome firing) for multiple rounds.  MWPM interprets
this as "no error" and assigns no correction, accumulating logical
errors silently.

ML advantage
------------
A classifier that sees the full spatio-temporal syndrome pattern can
learn the "frozen detector" signature of leakage and flag it for a
separate leakage-reset operation, which MWPM cannot do at all.

Simulation approach
-------------------
We use Stim for the base circuit, then inject leakage as a post-
processing layer:
1. Generate normal syndrome data from Stim.
2. Randomly select data qubits to be leaked at each round.
3. Leaked qubits stop contributing to stabiliser measurements
   (both X and Z), zeroing out the syndrome bits they would affect.
4. After a geometrically-distributed dwell time, the leakage resets
   (modeling leakage reduction units or natural decay back).

This produces a dataset with three labels:
  0 — no logical error, no leakage
  1 — logical error (Pauli), no leakage
  2 — leakage event (one or more qubits leaked this round)

References
----------
- Fowler (2013). Coping with qubit leakage in topological codes.
  PRA 88, 042308.
- McEwen et al. (2021). Removing leakage-induced correlated errors
  in superconducting quantum error correction. Nature Comms 12, 1761.
- Varbanov et al. (2020). Leakage detection for a transmon-based
  surface code. npj Quantum Information 6, 102.
"""

from __future__ import annotations

import numpy as np
import stim
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

from qec_ml.utils.config import QECConfig, NoiseConfig


@dataclass
class LeakageConfig:
    """Parameters controlling leakage injection."""
    p_leakage: float = 0.005        # prob. per qubit per round to leak
    p_reset: float = 0.3            # prob. per round to return from |2⟩
    leakage_darkens: bool = True    # leaked qubit silences its detectors
    seed: int = 42


@dataclass
class LeakageDataset:
    """
    Syndrome dataset with leakage annotations.

    Attributes
    ----------
    syndromes       : (N, L) uint8 — syndrome bits (some zeroed by leakage)
    logical_errors  : (N,) uint8 — Pauli logical error flag
    leakage_flags   : (N,) uint8 — 1 if any qubit leaked this round
    leakage_qubits  : list of sets — which qubits leaked per sample
    n_leaked_rounds : (N,) int — number of rounds with active leakage
    config          : QECConfig
    leakage_config  : LeakageConfig
    """
    syndromes: np.ndarray
    logical_errors: np.ndarray
    leakage_flags: np.ndarray
    leakage_qubits: List[set]
    n_leaked_rounds: np.ndarray
    config: QECConfig
    leakage_config: LeakageConfig
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.syndromes)

    def split(self, train=0.7, val=0.15):
        n = len(self)
        idx = np.random.permutation(n)
        i1, i2 = int(n * train), int(n * (train + val))

        def _sub(a, b):
            sl = idx[a:b]
            return LeakageDataset(
                syndromes=self.syndromes[sl],
                logical_errors=self.logical_errors[sl],
                leakage_flags=self.leakage_flags[sl],
                leakage_qubits=[self.leakage_qubits[i] for i in sl],
                n_leaked_rounds=self.n_leaked_rounds[sl],
                config=self.config,
                leakage_config=self.leakage_config,
            )
        return _sub(0, i1), _sub(i1, i2), _sub(i2, n)

    @property
    def leakage_rate(self) -> float:
        return float(self.leakage_flags.mean())

    @property
    def class_counts(self) -> Dict[str, int]:
        no_err = int(np.sum((self.logical_errors == 0) & (self.leakage_flags == 0)))
        log_err = int(np.sum((self.logical_errors == 1) & (self.leakage_flags == 0)))
        leak = int(np.sum(self.leakage_flags == 1))
        return {"no_error": no_err, "logical_error": log_err, "leakage": leak}


class LeakageSyndromeGenerator:
    """
    Generates surface-code syndromes with injected leakage errors.

    The generator:
    1. Samples clean Stim syndromes (circuit-level noise).
    2. Simulates leakage as a Markov chain per data qubit per round.
    3. Zeros out syndrome bits from leaked qubits (dark-detector effect).
    4. Returns a LeakageDataset with full annotations.

    Parameters
    ----------
    config : QECConfig
    leakage_config : LeakageConfig

    Examples
    --------
    >>> cfg = QECConfig(distance=5, noise=NoiseConfig('circuit_level', 0.005, rounds=5))
    >>> lcfg = LeakageConfig(p_leakage=0.005, p_reset=0.3)
    >>> gen = LeakageSyndromeGenerator(cfg, lcfg)
    >>> ds = gen.generate(n_samples=10_000)
    >>> print(f"Leakage rate: {ds.leakage_rate:.3f}")
    """

    def __init__(self, config: QECConfig, leakage_config: LeakageConfig):
        self.cfg = config
        self.lcfg = leakage_config
        self.rng = np.random.default_rng(leakage_config.seed)
        self._stim_circuit: Optional[stim.Circuit] = None

        d = config.distance
        R = config.noise.rounds
        # Ancilla-to-data-qubit adjacency (simplified for rotated code)
        # For detector i in round r, affected data qubits:
        self._n_ancilla = config.n_ancilla_qubits
        self._n_data = config.n_data_qubits
        self._adjacency = self._build_adjacency(d)

    def generate(self, n_samples: int) -> LeakageDataset:
        """Generate n_samples syndrome shots with leakage."""
        # Step 1: Get base syndromes from Stim
        circuit = self._get_circuit()
        sampler = circuit.compile_detector_sampler(seed=int(self.rng.integers(0, 2**31)))
        detections, observables = sampler.sample(n_samples, separate_observables=True)
        syndromes = detections.astype(np.uint8)
        logical_errors = observables[:, 0].astype(np.uint8)

        # Step 2: Inject leakage
        syndromes_leaked, leakage_flags, leakage_qubits, n_leaked = \
            self._inject_leakage(syndromes, n_samples)

        return LeakageDataset(
            syndromes=syndromes_leaked,
            logical_errors=logical_errors,
            leakage_flags=leakage_flags,
            leakage_qubits=leakage_qubits,
            n_leaked_rounds=n_leaked,
            config=self.cfg,
            leakage_config=self.lcfg,
            metadata={
                "distance": self.cfg.distance,
                "p_noise": self.cfg.noise.p,
                "p_leakage": self.lcfg.p_leakage,
                "n_samples": n_samples,
            },
        )

    def get_circuit(self) -> stim.Circuit:
        return self._get_circuit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_circuit(self) -> stim.Circuit:
        if self._stim_circuit is None:
            cfg = self.cfg
            self._stim_circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                rounds=cfg.noise.rounds,
                distance=cfg.distance,
                after_clifford_depolarization=cfg.noise.p,
                after_reset_flip_probability=cfg.noise.p,
                before_measure_flip_probability=cfg.noise.p_meas,
                before_round_data_depolarization=cfg.noise.p / 10,
            )
        return self._stim_circuit

    def _inject_leakage(self, syndromes: np.ndarray, n: int):
        """
        Simulate leakage Markov chain and zero out affected syndrome bits.

        Returns modified syndromes + annotation arrays.
        """
        d = self.cfg.distance
        R = self.cfg.noise.rounds
        n_data = self._n_data
        n_anc = self._n_ancilla
        p_l = self.lcfg.p_leakage
        p_r = self.lcfg.p_reset

        syndromes_out = syndromes.copy()
        leakage_flags = np.zeros(n, dtype=np.uint8)
        n_leaked_rounds = np.zeros(n, dtype=np.int32)
        leakage_qubits_list = []

        for i in range(n):
            # Markov chain: leaked[q] = True if qubit q is currently leaked
            leaked = np.zeros(n_data, dtype=bool)
            leaked_qubits_this_shot = set()
            total_leaked_rounds = 0

            for r in range(R):
                # Transition: leak new qubits
                new_leaks = self.rng.random(n_data) < p_l
                leaked = leaked | new_leaks

                # Transition: reset leaked qubits
                resets = leaked & (self.rng.random(n_data) < p_r)
                leaked = leaked & ~resets

                if leaked.any():
                    total_leaked_rounds += 1
                    leaked_qubits_this_shot.update(np.where(leaked)[0].tolist())

                    if self.lcfg.leakage_darkens:
                        # Zero out syndrome bits from ancillas adjacent to leaked qubits
                        for anc_idx in range(n_anc):
                            adj = self._adjacency.get(anc_idx, set())
                            if adj & set(np.where(leaked)[0].tolist()):
                                det_idx = r * n_anc + anc_idx
                                if det_idx < syndromes_out.shape[1]:
                                    syndromes_out[i, det_idx] = 0

            leakage_flags[i] = 1 if total_leaked_rounds > 0 else 0
            n_leaked_rounds[i] = total_leaked_rounds
            leakage_qubits_list.append(leaked_qubits_this_shot)

        return syndromes_out, leakage_flags, leakage_qubits_list, n_leaked_rounds

    def _build_adjacency(self, d: int) -> Dict[int, set]:
        """
        Build ancilla→data_qubit adjacency for rotated surface code.
        Ancilla i is adjacent to data qubits in its 2x2 plaquette.
        This is a simplified model — exact adjacency depends on the
        specific qubit layout, but captures the essential structure.
        """
        adj = {}
        n_anc = (d - 1) * d
        for anc in range(n_anc):
            row = anc // d
            col = anc % d
            # Each ancilla touches up to 4 data qubits
            neighbours = set()
            for dr, dc in [(-1, 0), (0, 0), (-1, 1), (0, 1)]:
                r2, c2 = row + dr, col + dc
                if 0 <= r2 < d and 0 <= c2 < d:
                    neighbours.add(r2 * d + c2)
            adj[anc] = neighbours
        return adj
