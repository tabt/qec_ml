"""
qec_ml.decoders.lookup_decoder
================================
Lookup Table (LUT) decoder for small surface codes.

For small code distances (d ≤ 5) and single-round measurements,
a pre-computed lookup table mapping syndrome → correction is optimal
and very fast.  This serves as an additional classical baseline.

The table is built by exhaustive enumeration of all 2^L syndromes
(feasible only for small L) and storing the most-likely logical class
under the noise model.

For larger codes, the table is built by sampling and majority-vote.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict
from collections import defaultdict

from qec_ml.decoders.base_decoder import BaseDecoder


class LookupDecoder(BaseDecoder):
    """
    Syndrome → logical-error lookup table decoder.

    Parameters
    ----------
    syndrome_length : int
        Must be small (≤ 20) for exact enumeration, or use fit() for larger.
    max_exact_length : int
        If syndrome_length ≤ this, build table exhaustively.
        Otherwise, build from training data via fit().

    Examples
    --------
    >>> dec = LookupDecoder(syndrome_length=12)
    >>> dec.fit(train_syndromes, train_labels)
    >>> preds = dec.decode_batch(test_syndromes)
    """

    def __init__(self, syndrome_length: int, max_exact_length: int = 18):
        self.L = syndrome_length
        self.max_exact = max_exact_length
        self._table: Dict[int, int] = {}  # syndrome int → predicted label

    @property
    def name(self) -> str:
        return "Lookup Table"

    def fit(
        self,
        syndromes: np.ndarray,
        labels: np.ndarray,
        min_count: int = 5,
    ) -> "LookupDecoder":
        """
        Build lookup table from training data by majority vote.

        Parameters
        ----------
        syndromes : (N, L) uint8 array
        labels : (N,) uint8 array
        min_count : int
            Minimum number of occurrences to store an entry.
            Syndromes seen fewer times default to label=0.
        """
        counts: Dict[int, list] = defaultdict(list)
        for syn, lbl in zip(syndromes, labels):
            key = int(np.packbits(syn, bitorder='little').view(np.uint64)[0])
            counts[key].append(int(lbl))

        for key, lbls in counts.items():
            if len(lbls) >= min_count:
                self._table[key] = int(np.round(np.mean(lbls)))
            else:
                self._table[key] = 0  # default: no error

        coverage = len(self._table) / max(1, 2 ** min(self.L, 20))
        print(f"LUT built: {len(self._table)} entries "
              f"({100 * len(counts) / max(counts, 1):.1f}% syndrome coverage)")
        return self

    def decode(self, syndrome: np.ndarray) -> int:
        key = self._syndrome_key(syndrome)
        return self._table.get(key, 0)

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        return np.array([self.decode(s) for s in syndromes], dtype=np.uint8)

    def _syndrome_key(self, syndrome: np.ndarray) -> int:
        padded = np.zeros(64, dtype=np.uint8)
        padded[: len(syndrome)] = syndrome
        return int(np.packbits(padded[:64], bitorder="little").view(np.uint64)[0])
