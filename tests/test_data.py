"""
tests/test_data.py
===================
Unit tests for qec_ml.data modules.
Run with: pytest tests/
"""

import numpy as np
import pytest

from qec_ml.utils.config import QECConfig, NoiseConfig
from qec_ml.data.syndrome_generator import SyndromeGenerator, SyndromeDataset
from qec_ml.data.analog_signal import AnalogSignalSimulator, ReadoutConfig
from qec_ml.data.noise_models import SimpleRepetitionCodeSimulator


# ======================================================================
# SyndromeGenerator
# ======================================================================

class TestSyndromeGenerator:

    @pytest.fixture
    def small_cfg(self):
        return QECConfig(
            distance=3,
            noise=NoiseConfig(model="depolarizing", p=0.01, rounds=3),
            seed=42,
        )

    def test_generate_shape(self, small_cfg):
        gen = SyndromeGenerator(small_cfg)
        ds = gen.generate(n_samples=100)
        assert ds.syndromes.shape[0] == 100
        assert ds.syndromes.shape[1] == small_cfg.syndrome_length
        assert ds.logical_errors.shape == (100,)

    def test_binary_values(self, small_cfg):
        gen = SyndromeGenerator(small_cfg)
        ds = gen.generate(n_samples=200)
        assert set(np.unique(ds.syndromes)).issubset({0, 1})
        assert set(np.unique(ds.logical_errors)).issubset({0, 1})

    def test_split_sizes(self, small_cfg):
        gen = SyndromeGenerator(small_cfg)
        ds = gen.generate(n_samples=1000)
        train, val, test = ds.split(train=0.7, val=0.15)
        assert len(train) == 700
        assert len(val) == 150
        assert len(test) == 150

    def test_logical_error_rate_nonzero(self, small_cfg):
        gen = SyndromeGenerator(small_cfg)
        ds = gen.generate(n_samples=500)
        # With p=0.01 and d=3, some logical errors should occur
        assert ds.logical_errors.mean() > 0

    @pytest.mark.parametrize("noise_model", ["depolarizing", "bit_flip", "circuit_level"])
    def test_noise_models(self, noise_model):
        cfg = QECConfig(
            distance=3,
            noise=NoiseConfig(model=noise_model, p=0.01, rounds=1),
        )
        gen = SyndromeGenerator(cfg)
        ds = gen.generate(n_samples=50)
        assert ds.syndromes.shape[0] == 50

    def test_config_properties(self, small_cfg):
        assert small_cfg.n_data_qubits == 9
        assert small_cfg.n_ancilla_qubits == 8
        assert small_cfg.syndrome_length == 8 * 3  # ancillas * rounds


# ======================================================================
# AnalogSignalSimulator
# ======================================================================

class TestAnalogSignalSimulator:

    @pytest.fixture
    def cfg(self):
        return ReadoutConfig(
            sigma_noise=0.4,
            t1_error_prob=0.02,
            n_time_bins=50,
            n_qubits=1,
            seed=42,
        )

    def test_generate_shape(self, cfg):
        sim = AnalogSignalSimulator(cfg)
        ds = sim.generate(n_samples=100)
        assert ds.trajectories.shape == (100, 1, 50, 2)
        assert ds.true_states.shape == (100, 1)

    def test_state_fractions(self, cfg):
        sim = AnalogSignalSimulator(cfg)
        ds = sim.generate(n_samples=2000, state_fractions=[0.3, 0.7])
        frac_1 = ds.true_states[:, 0].mean()
        # Allow for prep errors: should be close to 0.7
        assert abs(frac_1 - 0.7) < 0.05

    def test_threshold_accuracy_reasonable(self, cfg):
        sim = AnalogSignalSimulator(cfg)
        ds = sim.generate(n_samples=1000)
        # With sigma=0.4 and separation=2, expect >80% accuracy
        assert ds.threshold_accuracy > 0.80

    def test_integrated_iq_shape(self, cfg):
        sim = AnalogSignalSimulator(cfg)
        ds = sim.generate(n_samples=100)
        iq = ds.integrated_iq
        assert iq.shape == (100, 1, 2)

    def test_multiqubit(self):
        cfg = ReadoutConfig(n_qubits=3, n_time_bins=50, seed=0)
        sim = AnalogSignalSimulator(cfg)
        ds = sim.generate(n_samples=50)
        assert ds.trajectories.shape == (50, 3, 50, 2)

    def test_crosstalk(self, cfg):
        cfg_multi = ReadoutConfig(n_qubits=2, n_time_bins=50, seed=0)
        sim = AnalogSignalSimulator(cfg_multi)
        ds = sim.generate(n_samples=100)
        ct = np.array([[1.0, 0.1], [0.1, 1.0]])
        ds_ct = sim.add_crosstalk(ds, ct)
        assert ds_ct.trajectories.shape == ds.trajectories.shape
        # Crosstalk changes the trajectories
        assert not np.allclose(ds.trajectories, ds_ct.trajectories)


# ======================================================================
# RepetitionCode (noise_models)
# ======================================================================

class TestRepetitionCode:

    def test_generate(self):
        sim = SimpleRepetitionCodeSimulator(n_bits=5, p=0.05)
        syns, labels = sim.generate(n_samples=200)
        assert syns.shape == (200, 4)
        assert labels.shape == (200,)
        assert set(np.unique(labels)).issubset({0, 1})

    def test_mwpm_decode_no_errors(self):
        sim = SimpleRepetitionCodeSimulator(n_bits=5, p=0.0)
        syns, labels = sim.generate(n_samples=100)
        preds = sim.mwpm_decode(syns)
        # With p=0, no errors → all syndromes zero → all predictions 0
        assert np.all(preds == 0)
        assert np.all(labels == 0)

    def test_ler_decreases_with_distance(self):
        """Fundamental QEC property: larger d → lower LER below threshold."""
        p = 0.05  # below threshold for repetition code (~0.5)
        lers = []
        for d in [3, 5, 7, 9]:
            sim = SimpleRepetitionCodeSimulator(n_bits=d, p=p, rng=np.random.default_rng(0))
            syns, labels = sim.generate(n_samples=2000)
            preds = sim.mwpm_decode(syns)
            lers.append(np.mean(preds != labels))
        # LER should generally decrease with d
        assert lers[0] > lers[-1], f"LERs did not decrease: {lers}"
