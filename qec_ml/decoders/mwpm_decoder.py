"""
qec_ml.decoders.mwpm_decoder
==============================
Minimum-Weight Perfect Matching decoder using PyMatching.

MWPM is the industry-standard classical decoder for surface codes.
It finds the minimum-weight correction that is consistent with the
observed syndrome by solving a minimum-weight perfect matching problem
on the syndrome graph.

References
----------
- Higgott, O. (2022). PyMatching: A Python package for decoding
  quantum codes with minimum-weight perfect matching.
  ACM Transactions on Quantum Computing 3(3).
- Dennis, E. et al. (2002). Topological quantum memory. JMP 43, 4452.
"""

from __future__ import annotations

import numpy as np
import stim
import pymatching

from qec_ml.decoders.base_decoder import BaseDecoder
from qec_ml.utils.config import QECConfig


class MWPMDecoder(BaseDecoder):
    """
    MWPM decoder using PyMatching v2.

    Parameters
    ----------
    config : QECConfig
        Must contain a valid noise configuration so that the detector
        error model can be derived from the Stim circuit.

    Examples
    --------
    >>> decoder = MWPMDecoder(config)
    >>> preds = decoder.decode_batch(test_syndromes)
    """

    def __init__(self, config: QECConfig):
        self.config = config
        self._matcher: pymatching.Matching | None = None

    @property
    def name(self) -> str:
        return "MWPM (PyMatching)"

    def build(self, circuit: stim.Circuit) -> "MWPMDecoder":
        """
        Initialise the matcher from a Stim circuit.

        Parameters
        ----------
        circuit : stim.Circuit
            The Stim circuit whose detector error model is used to
            build the matching graph.
        """
        dem = circuit.detector_error_model(decompose_errors=True)
        self._matcher = pymatching.Matching.from_detector_error_model(dem)
        return self

    def decode(self, syndrome: np.ndarray) -> int:
        self._check_ready()
        pred = self._matcher.decode(syndrome.astype(bool))
        return int(pred[0])

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        self._check_ready()
        preds = self._matcher.decode_batch(syndromes.astype(bool))
        return preds[:, 0].astype(np.uint8)

    def _check_ready(self) -> None:
        if self._matcher is None:
            raise RuntimeError(
                "MWPMDecoder is not built yet.  Call .build(circuit) first."
            )
