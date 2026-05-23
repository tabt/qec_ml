"""
qec_ml.decoders.ml_decoder_wrapper
=====================================
Adapter that wraps a trained PyTorch nn.Module so it conforms to the
BaseDecoder interface expected by BenchmarkRunner.

This allows ML models and classical decoders to be evaluated
with identical code.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Callable

from qec_ml.decoders.base_decoder import BaseDecoder


class MLDecoderWrapper(BaseDecoder):
    """
    Wraps a trained PyTorch model as a BaseDecoder.

    Parameters
    ----------
    model : nn.Module
        Trained model.  Must accept (B, L) float tensor and return (B,) logits
        (binary) or (B, C) logits (multi-class).
    model_name : str
    device : str
    threshold : float
        Decision threshold on sigmoid probability (binary case).
    preprocess_fn : callable, optional
        Applied to the raw numpy syndrome before feeding to the model.
        Default: identity (cast to float32 tensor).
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: str = "cpu",
        threshold: float = 0.5,
        preprocess_fn: Optional[Callable] = None,
    ):
        self.model = model.eval()
        self._name = model_name
        self.device = torch.device(device)
        self.model.to(self.device)
        self.threshold = threshold
        self._preprocess = preprocess_fn

    @property
    def name(self) -> str:
        return self._name

    def decode(self, syndrome: np.ndarray) -> int:
        return int(self.decode_batch(syndrome[None])[0])

    @torch.no_grad()
    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(syndromes.astype(np.float32)).to(self.device)
        if self._preprocess is not None:
            x = self._preprocess(x)
        logits = self.model(x)
        if logits.dim() == 1:
            # Binary classification
            preds = (torch.sigmoid(logits) > self.threshold).long()
        else:
            # Multi-class
            preds = logits.argmax(dim=-1)
        return preds.cpu().numpy().astype(np.uint8)

    @torch.no_grad()
    def predict_proba(self, syndromes: np.ndarray) -> np.ndarray:
        """Return soft probabilities for AUC-ROC computation."""
        x = torch.from_numpy(syndromes.astype(np.float32)).to(self.device)
        logits = self.model(x)
        if logits.dim() == 1:
            return torch.sigmoid(logits).cpu().numpy()
        return torch.softmax(logits, dim=-1).cpu().numpy()
