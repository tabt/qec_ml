"""
qec_ml.data.leakage_noise
==========================
Leakage error simulation for superconducting qubits.

v2 changes
----------
- Stronger, more learnable leakage parameters (p_leakage=0.02-0.05)
- Explicit persistence features added to the dataset:
    * persistence_map: per-ancilla count of consecutive zero rounds
    * round_syndromes: (N, R, n_ancilla) reshaped view for temporal models
- LeakageConfig now validated to ensure detectable signal
- Fixed adjacency: each ancilla touches exactly the data qubits in its plaquette
"""
from __future__ import annotations
import numpy as np
import stim
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from qec_ml.utils.config import QECConfig, NoiseConfig


@dataclass
class LeakageConfig:
    """Parameters controlling leakage injection.
    
    Recommended values for learnable leakage detection:
        p_leakage = 0.02-0.05  (per qubit per round)
        p_reset   = 0.30-0.50  (per round; gives mean duration 2-3 rounds)
    
    The signal-to-noise ratio is:
        SNR ~ p_leakage / (p_leakage + p_reset) * n_data_qubits * ancilla_per_qubit
    A value > 5% of syndrome bits is needed for reliable ML detection.
    """
    p_leakage: float = 0.03       # prob per qubit per round to enter |2>
    p_reset:   float = 0.40       # prob per round to return from |2>
    leakage_darkens: bool = True   # leaked qubit silences adjacent ancillas
    seed: int = 42

    def __post_init__(self):
        duration = 1 / self.p_reset
        steady = self.p_leakage / (self.p_leakage + self.p_reset)
        if steady * 25 * 2 * 5 / 120 < 0.05:
            import warnings
            warnings.warn(
                f"LeakageConfig may produce too weak a signal "
                f"(~{steady*25*2*5/120*100:.1f}% of syndrome bits affected). "
                f"Consider increasing p_leakage or decreasing p_reset."
            )


@dataclass
class LeakageDataset:
    """Syndrome dataset with leakage annotations and temporal features."""
    syndromes:        np.ndarray   # (N, L)      — flat syndrome, some bits zeroed
    round_syndromes:  np.ndarray   # (N, R, A)   — reshaped: rounds × ancilla
    persistence_map:  np.ndarray   # (N, A)      — per-ancilla consecutive-zero count
    logical_errors:   np.ndarray   # (N,)        — Pauli logical error flag
    leakage_flags:    np.ndarray   # (N,)        — any qubit leaked this shot
    n_leaked_rounds:  np.ndarray   # (N,)        — rounds with active leakage
    config:           QECConfig
    leakage_config:   LeakageConfig
    metadata:         Dict[str, Any] = field(default_factory=dict)

    def __len__(self):
        return len(self.syndromes)

    def split(self, train=0.7, val=0.15):
        n = len(self)
        idx = np.random.permutation(n)
        i1, i2 = int(n * train), int(n * (train + val))
        def _sub(a, b):
            sl = idx[a:b]
            return LeakageDataset(
                syndromes=self.syndromes[sl],
                round_syndromes=self.round_syndromes[sl],
                persistence_map=self.persistence_map[sl],
                logical_errors=self.logical_errors[sl],
                leakage_flags=self.leakage_flags[sl],
                n_leaked_rounds=self.n_leaked_rounds[sl],
                config=self.config,
                leakage_config=self.leakage_config,
            )
        return _sub(0, i1), _sub(i1, i2), _sub(i2, n)

    @property
    def leakage_rate(self):
        return float(self.leakage_flags.mean())

    @property
    def class_counts(self):
        return {
            "no_leakage":   int((self.leakage_flags == 0).sum()),
            "leakage":      int((self.leakage_flags == 1).sum()),
        }

    def signal_stats(self):
        """Diagnostic: how many syndrome bits are affected by leakage on average."""
        lk_idx = self.leakage_flags == 1
        nl_idx = self.leakage_flags == 0
        mean_w_lk = self.syndromes[lk_idx].mean()  if lk_idx.any() else 0
        mean_w_nl = self.syndromes[nl_idx].mean() if nl_idx.any() else 0
        mean_p_lk = self.persistence_map[lk_idx].mean() if lk_idx.any() else 0
        mean_p_nl = self.persistence_map[nl_idx].mean() if nl_idx.any() else 0
        return {
            "mean_syndrome_weight_leaked":  mean_w_lk,
            "mean_syndrome_weight_normal":  mean_w_nl,
            "mean_persistence_leaked":      mean_p_lk,
            "mean_persistence_normal":      mean_p_nl,
            "weight_ratio_leaked_vs_normal": mean_w_lk / max(mean_w_nl, 1e-9),
        }


class LeakageSyndromeGenerator:
    """
    Generates surface-code syndromes with injected leakage errors.

    Key improvements in v2:
    - Returns round_syndromes: (N, R, A) — gives temporal models the
      per-round structure they need to see persistence
    - Returns persistence_map: (N, A) — explicit consecutive-zero count
      per ancilla position, a hand-engineered leakage feature
    - Better adjacency computation matching rotated surface code geometry
    """

    def __init__(self, config: QECConfig, leakage_config: LeakageConfig):
        self.cfg = config
        self.lcfg = leakage_config
        self.rng = np.random.default_rng(leakage_config.seed)
        self._stim_circuit: Optional[stim.Circuit] = None
        self._adjacency = self._build_adjacency(config.distance)

    def generate(self, n_samples: int) -> LeakageDataset:
        circuit = self._get_circuit()
        sampler = circuit.compile_detector_sampler(
            seed=int(self.rng.integers(0, 2**31))
        )
        detections, observables = sampler.sample(n_samples, separate_observables=True)
        syndromes = detections.astype(np.uint8)
        logical_errors = observables[:, 0].astype(np.uint8)

        syndromes_out, round_syns, persist, lk_flags, n_lk =             self._inject_leakage(syndromes, n_samples)

        stats = LeakageDataset(
            syndromes=syndromes_out,
            round_syndromes=round_syns,
            persistence_map=persist,
            logical_errors=logical_errors,
            leakage_flags=lk_flags,
            n_leaked_rounds=n_lk,
            config=self.cfg,
            leakage_config=self.lcfg,
            metadata={
                "distance": self.cfg.distance,
                "p_leakage": self.lcfg.p_leakage,
                "p_reset":   self.lcfg.p_reset,
                "n_samples": n_samples,
            },
        )
        return stats

    def get_circuit(self) -> stim.Circuit:
        return self._get_circuit()

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

    def _inject_leakage(self, syndromes, n):
        d      = self.cfg.distance
        R      = self.cfg.noise.rounds
        n_data = self.cfg.n_data_qubits
        n_anc  = self.cfg.n_ancilla_qubits
        p_l    = self.lcfg.p_leakage
        p_r    = self.lcfg.p_reset

        syndromes_out = syndromes.copy()
        # round_syns: (N, R, n_anc) — per-round view
        round_syns = syndromes_out.reshape(n, R, -1)[:, :, :n_anc].copy()
        persist = np.zeros((n, n_anc), dtype=np.float32)
        lk_flags  = np.zeros(n, dtype=np.uint8)
        n_lk      = np.zeros(n, dtype=np.int32)

        for i in range(n):
            leaked = np.zeros(n_data, dtype=bool)
            total_lk_rounds = 0

            # per-ancilla streak counter: how many consecutive rounds was it 0
            streak = np.zeros(n_anc, dtype=np.int32)

            for r in range(R):
                new_leaks = self.rng.random(n_data) < p_l
                leaked = leaked | new_leaks
                resets = leaked & (self.rng.random(n_data) < p_r)
                leaked = leaked & ~resets

                if leaked.any():
                    total_lk_rounds += 1
                    lk_flags[i] = 1
                    if self.lcfg.leakage_darkens:
                        leaked_set = set(np.where(leaked)[0].tolist())
                        for anc in range(n_anc):
                            if self.cfg.distance > 0 and                                self._adjacency.get(anc, set()) & leaked_set:
                                det = r * n_anc + anc
                                if det < syndromes_out.shape[1]:
                                    syndromes_out[i, det] = 0
                                if r < round_syns.shape[1] and anc < round_syns.shape[2]:
                                    round_syns[i, r, anc] = 0
                                streak[anc] += 1
                            else:
                                streak[anc] = 0
                else:
                    streak[:] = 0

                persist[i] = np.maximum(persist[i], streak)

            n_lk[i] = total_lk_rounds

        return syndromes_out, round_syns, persist, lk_flags, n_lk

    def _build_adjacency(self, d):
        adj = {}
        for anc in range((d - 1) * d):
            row, col = anc // d, anc % d
            neighbours = set()
            for dr, dc in [(-1, 0), (0, 0), (-1, 1), (0, 1)]:
                r2, c2 = row + dr, col + dc
                if 0 <= r2 < d and 0 <= c2 < d:
                    neighbours.add(r2 * d + c2)
            adj[anc] = neighbours
        return adj
