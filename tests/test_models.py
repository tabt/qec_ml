"""
tests/test_models.py
=====================
Unit tests for ML decoder models (forward pass shapes, no training needed).
"""

import numpy as np
import pytest
import torch

from qec_ml.models.mlp_decoder import MLPDecoder, CNNDecoder
from qec_ml.models.transformer_decoder import SyndromeTransformer, SpatialTemporalTransformer
from qec_ml.models.lstm_corrector import LSTMClassifier, Conv1DClassifier, IQAutoencoder


DEVICE = "cpu"
B = 8  # batch size for tests


class TestMLPDecoder:

    def test_forward_shape(self):
        model = MLPDecoder(syndrome_length=48, hidden_dims=[64, 32])
        x = torch.randint(0, 2, (B, 48)).float()
        out = model(x)
        assert out.shape == (B,), f"Expected ({B},), got {out.shape}"

    def test_default_hidden_dims(self):
        model = MLPDecoder(syndrome_length=120)
        x = torch.zeros(B, 120)
        out = model(x)
        assert out.shape == (B,)

    def test_dropout_train_vs_eval(self):
        model = MLPDecoder(syndrome_length=48, dropout=0.5)
        x = torch.rand(B, 48)
        model.train()
        out_train = model(x)
        model.eval()
        out_eval = model(x)
        # In eval mode outputs should be deterministic
        assert torch.allclose(model(x), out_eval)


class TestCNNDecoder:

    def test_forward_shape(self):
        model = CNNDecoder(in_channels=5, grid_size=4, base_channels=16, n_blocks=2)
        x = torch.randint(0, 2, (B, 5, 4, 4)).float()
        out = model(x)
        assert out.shape == (B,)


class TestSyndromeTransformer:

    def test_forward_shape_cls(self):
        model = SyndromeTransformer(syndrome_length=48, d_model=32, n_heads=4,
                                     n_layers=2, use_cls_token=True)
        x = torch.randint(0, 2, (B, 48)).float()
        out = model(x)
        assert out.shape == (B,)

    def test_forward_shape_mean_pool(self):
        model = SyndromeTransformer(syndrome_length=48, d_model=32, n_heads=4,
                                     n_layers=2, use_cls_token=False)
        x = torch.randint(0, 2, (B, 48)).float()
        out = model(x)
        assert out.shape == (B,)

    def test_gradient_flows(self):
        model = SyndromeTransformer(syndrome_length=24, d_model=32, n_heads=4, n_layers=2)
        x = torch.randint(0, 2, (B, 24)).float()
        loss = model(x).mean()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                assert not torch.isnan(p.grad).any()


class TestSpatialTemporalTransformer:

    def test_forward_shape(self):
        model = SpatialTemporalTransformer(n_ancilla=24, rounds=5,
                                            d_model=32, n_heads=4, n_layers=2)
        x = torch.randint(0, 2, (B, 24 * 5)).float()
        out = model(x)
        assert out.shape == (B,)


class TestLSTMClassifier:

    def test_forward_shape_binary(self):
        model = LSTMClassifier(input_size=2, hidden_size=32, n_layers=1, n_classes=2)
        x = torch.randn(B, 100, 2)
        out = model(x)
        assert out.shape == (B, 2)

    def test_bidirectional(self):
        model = LSTMClassifier(input_size=2, hidden_size=32, n_layers=2, n_classes=2)
        x = torch.randn(B, 50, 2)
        out = model(x)
        assert out.shape == (B, 2)


class TestConv1DClassifier:

    def test_forward_shape(self):
        model = Conv1DClassifier(input_size=2, n_filters=32, n_blocks=3, n_classes=2)
        x = torch.randn(B, 100, 2)
        out = model(x)
        assert out.shape == (B, 2)


class TestIQAutoencoder:

    def test_forward_shapes(self):
        model = IQAutoencoder(input_size=2, n_filters=16, latent_dim=8, seq_len=100)
        x = torch.randn(B, 100, 2)
        recon, z = model(x)
        assert recon.shape == (B, 100, 2), f"Recon shape: {recon.shape}"
        assert z.shape == (B, 8), f"Latent shape: {z.shape}"

    def test_encode_decode_roundtrip(self):
        model = IQAutoencoder(input_size=2, n_filters=16, latent_dim=8, seq_len=100)
        model.eval()
        x = torch.randn(B, 100, 2)
        z = model.encode(x)
        x_hat = model.decode(z)
        assert x_hat.shape == x.shape
