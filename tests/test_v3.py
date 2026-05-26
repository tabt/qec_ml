"""
tests/test_v3.py
=================
Unit tests for v3 additions: leakage, correlated noise, GNN reweighter.
"""
import numpy as np
import pytest
import torch

from qec_ml.utils.config import QECConfig, NoiseConfig
from qec_ml.data.leakage_noise import LeakageSyndromeGenerator, LeakageConfig
from qec_ml.data.correlated_noise import CorrelatedNoiseGenerator, CorrelatedNoiseConfig
from qec_ml.models.leakage_detector import (
    LeakageDetectorCNN, LeakageClassifierTransformer, SyndromeAnomalyDetector,
)

B = 8
CFG = QECConfig(distance=5, noise=NoiseConfig(model="depolarizing", p=0.01, rounds=5), seed=0)
LCFG = LeakageConfig(p_leakage=0.01, p_reset=0.3, seed=0)


class TestLeakageGenerator:
    def test_generate_shapes(self):
        gen = LeakageSyndromeGenerator(CFG, LCFG)
        ds = gen.generate(n_samples=100)
        assert ds.syndromes.shape == (100, CFG.syndrome_length)
        assert ds.leakage_flags.shape == (100,)
        assert ds.logical_errors.shape == (100,)

    def test_leakage_flag_nonzero(self):
        gen = LeakageSyndromeGenerator(CFG, LCFG)
        ds = gen.generate(n_samples=500)
        assert ds.leakage_rate > 0, "Expected some leakage events"

    def test_split(self):
        gen = LeakageSyndromeGenerator(CFG, LCFG)
        ds = gen.generate(n_samples=200)
        tr, va, te = ds.split(0.7, 0.15)
        assert len(tr) + len(va) + len(te) == 200


class TestCorrelatedNoiseGenerator:
    @pytest.mark.parametrize("mode", ["spatial", "burst", "temporal"])
    def test_modes(self, mode):
        ncfg = CorrelatedNoiseConfig(mode=mode, seed=0)
        gen = CorrelatedNoiseGenerator(CFG, ncfg)
        ds = gen.generate(n_samples=50)
        assert ds.syndromes.shape[0] == 50
        assert ds.logical_errors.shape == (50,)
        assert ds.correlation_labels.shape == (50,)

    def test_burst_has_events(self):
        ncfg = CorrelatedNoiseConfig(mode="burst", burst_rate=0.5, seed=0)
        gen = CorrelatedNoiseGenerator(CFG, ncfg)
        ds = gen.generate(n_samples=200)
        assert ds.correlation_labels.sum() > 0


class TestLeakageModels:
    L = CFG.syndrome_length

    def test_detector_cnn_forward(self):
        model = LeakageDetectorCNN(distance=5, rounds=5, base_channels=16)
        x = torch.zeros(B, self.L)
        out = model(x)
        assert out.shape == (B,)

    def test_classifier_transformer_forward(self):
        model = LeakageClassifierTransformer(distance=5, rounds=5, d_model=32, n_heads=4, n_layers=2)
        x = torch.zeros(B, self.L)
        log_logit, lk_logit = model(x)
        assert log_logit.shape == (B,)
        assert lk_logit.shape == (B,)

    def test_anomaly_detector_forward(self):
        model = SyndromeAnomalyDetector(syndrome_length=self.L, latent_dim=8, hidden_dim=32)
        x = torch.rand(B, self.L)
        recon, z = model(x)
        assert recon.shape == (B, self.L)
        assert z.shape == (B, 8)

    def test_anomaly_score(self):
        model = SyndromeAnomalyDetector(syndrome_length=self.L, latent_dim=8, hidden_dim=32)
        x = torch.rand(B, self.L)
        scores = model.anomaly_score(x)
        assert scores.shape == (B,)
        assert (scores >= 0).all()

    def test_gradients_flow(self):
        model = LeakageClassifierTransformer(distance=5, rounds=5, d_model=32, n_heads=4, n_layers=2)
        x = torch.zeros(B, self.L)
        log_l, lk_l = model(x)
        loss = log_l.mean() + lk_l.mean()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                assert not torch.isnan(p.grad).any()
