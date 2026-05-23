"""
qec_ml.benchmarks.metrics
==========================
QEC-specific metrics and the benchmark runner that compares
multiple decoders on the same test set.

Key metrics
-----------
logical_error_rate (LER)
    Fraction of shots where the decoder predicts the wrong logical outcome.
    The primary figure of merit for QEC decoders.

threshold
    The physical error rate p* at which the logical error rate starts to
    decrease as code distance d increases.  Above threshold, the code
    offers no advantage.

decoding_time_ms_per_shot
    Wall-clock latency per syndrome, relevant for real-time decoding.

AUC-ROC
    For ML decoders that output soft probabilities.

decoder_advantage
    LER ratio (MWPM / ML decoder), >1 means ML outperforms MWPM.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from sklearn.metrics import roc_auc_score, confusion_matrix


@dataclass
class DecoderResult:
    """Results for one decoder on one test set."""
    name: str
    logical_error_rate: float
    accuracy: float
    decoding_time_ms: float         # per shot
    n_shots: int
    auc_roc: Optional[float] = None
    confusion: Optional[np.ndarray] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def logical_fidelity(self) -> float:
        return 1 - self.logical_error_rate


def compute_threshold(
    lers: np.ndarray,
    distances: np.ndarray,
    noise_rates: np.ndarray,
) -> float:
    """
    Estimate the error threshold p* by finding where LER(d) curves cross.

    Simple method: find p where larger d gives lower LER than smaller d.
    For a robust fit, use scipy curve_fit with a polynomial model.

    Parameters
    ----------
    lers : (n_distances, n_noise_rates) array
    distances : (n_distances,) array
    noise_rates : (n_noise_rates,) array

    Returns
    -------
    threshold : float — estimated p*
    """
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d

    # Find crossing point between d_min and d_max LER curves
    d_min_idx, d_max_idx = 0, -1
    ler_small = interp1d(noise_rates, lers[d_min_idx], kind="cubic", fill_value="extrapolate")
    ler_large = interp1d(noise_rates, lers[d_max_idx], kind="cubic", fill_value="extrapolate")

    def diff(p):
        return ler_large(p) - ler_small(p)

    try:
        p_star = brentq(diff, noise_rates[0], noise_rates[-1])
    except ValueError:
        p_star = float("nan")

    return p_star


class BenchmarkRunner:
    """
    Runs all registered decoders on the same test data and collects metrics.

    Usage
    -----
    >>> runner = BenchmarkRunner()
    >>> runner.add_decoder(mwpm_decoder)
    >>> runner.add_decoder(transformer_wrapper)
    >>> df = runner.run(test_syndromes, test_labels)
    >>> print(df)
    """

    def __init__(self):
        self._decoders: List[Any] = []

    def add_decoder(self, decoder) -> "BenchmarkRunner":
        """Register a decoder (must implement decode_batch and name)."""
        self._decoders.append(decoder)
        return self

    def run(
        self,
        syndromes: np.ndarray,
        labels: np.ndarray,
        proba_fns: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Evaluate all decoders and return a results DataFrame.

        Parameters
        ----------
        syndromes : (N, L) uint8 array
        labels : (N,) uint8 array
        proba_fns : dict {decoder_name: callable} — optional soft-output functions

        Returns
        -------
        pd.DataFrame with one row per decoder
        """
        rows = []
        for dec in self._decoders:
            print(f"  Evaluating: {dec.name} ...")
            t0 = time.perf_counter()
            preds = dec.decode_batch(syndromes)
            elapsed_ms = (time.perf_counter() - t0) / len(labels) * 1000

            ler = float(np.mean(preds != labels))
            cm = confusion_matrix(labels, preds, labels=[0, 1])

            auc = None
            if proba_fns and dec.name in proba_fns:
                try:
                    proba = proba_fns[dec.name](syndromes)
                    auc = float(roc_auc_score(labels, proba))
                except Exception:
                    pass

            rows.append({
                "decoder": dec.name,
                "logical_error_rate": ler,
                "accuracy": 1 - ler,
                "decoding_time_ms_per_shot": elapsed_ms,
                "auc_roc": auc,
                "n_shots": len(labels),
            })

        df = pd.DataFrame(rows).sort_values("logical_error_rate")
        return df

    def run_vs_noise(
        self,
        decoder,
        generator_fn,
        noise_rates: np.ndarray,
        n_shots: int = 5_000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sweep over noise rates to produce a LER-vs-p curve.

        Parameters
        ----------
        decoder : decoder object
        generator_fn : callable(p) → (syndromes, labels)
        noise_rates : array of p values
        n_shots : int

        Returns
        -------
        (noise_rates, lers) — both as np.ndarray
        """
        lers = []
        for p in noise_rates:
            syndromes, labels = generator_fn(p, n_shots)
            preds = decoder.decode_batch(syndromes)
            lers.append(np.mean(preds != labels))
        return noise_rates, np.array(lers)

    def run_vs_distance(
        self,
        decoder_factory_fn,
        generator_factory_fn,
        distances: List[int],
        p: float,
        n_shots: int = 5_000,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Sweep over code distances for a fixed noise rate.

        Parameters
        ----------
        decoder_factory_fn : callable(d) → decoder
        generator_factory_fn : callable(d) → (syndromes, labels)
        distances : list of int
        p : float — fixed noise rate

        Returns
        -------
        (distances, lers)
        """
        lers = []
        for d in distances:
            decoder = decoder_factory_fn(d)
            syndromes, labels = generator_factory_fn(d, n_shots)
            preds = decoder.decode_batch(syndromes)
            lers.append(float(np.mean(preds != labels)))
        return distances, np.array(lers)
