"""
qec_ml.data.datasets
=====================
PyTorch Dataset wrappers for syndrome and analog signal data.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple

from qec_ml.data.syndrome_generator import SyndromeDataset
from qec_ml.data.analog_signal import AnalogDataset
from qec_ml.utils.config import TrainingConfig


class SyndromeDatasetTorch(Dataset):
    """
    PyTorch Dataset for surface-code syndrome classification.

    Each sample: (syndrome_vector, logical_error_label).

    Parameters
    ----------
    data : SyndromeDataset
        Raw numpy dataset from SyndromeGenerator.
    augment : bool
        If True, randomly flip syndrome bits with low probability
        (simple augmentation to improve generalization).
    """

    def __init__(self, data: SyndromeDataset, augment: bool = False):
        self.syndromes = torch.from_numpy(data.syndromes).float()
        self.labels = torch.from_numpy(data.logical_errors).long()
        self.augment = augment
        self.syndrome_length = data.syndromes.shape[1]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.syndromes[idx]
        y = self.labels[idx]
        if self.augment and torch.rand(1).item() < 0.1:
            # Random bit-flip augmentation (noise-robust training)
            flip_mask = torch.rand_like(x) < 0.02
            x = (x + flip_mask.float()) % 2
        return x, y


class SyndromeSpatialDatasetTorch(Dataset):
    """
    Dataset variant that reshapes syndromes into a 2D spatial grid
    for CNN-based decoders.

    For a distance-d surface code, the ancilla layout is approximately
    (d-1) × d or d × (d-1) for X and Z stabilizers separately.
    We use a simplified square-ish layout here.

    Each sample: (2D_syndrome_grid [C, H, W], label).
    """

    def __init__(self, data: SyndromeDataset, augment: bool = False):
        self.labels = torch.from_numpy(data.logical_errors).long()
        self.augment = augment

        rounds = data.config.noise.rounds
        n_anc = data.syndromes.shape[1] // rounds

        # Tile ancilla into nearest square grid
        side = int(np.ceil(np.sqrt(n_anc)))
        pad_len = side * side * rounds - data.syndromes.shape[1]
        if pad_len > 0:
            padded = np.pad(data.syndromes, ((0, 0), (0, pad_len)))
        else:
            padded = data.syndromes

        # Shape: (N, rounds, side, side)
        reshaped = padded.reshape(len(padded), rounds, side, side)
        self.syndromes = torch.from_numpy(reshaped).float()

        # Expose for downstream use (e.g. CNN preprocess_fn in wrappers)
        self.rounds = rounds
        self.side = side
        self.n_ancilla = n_anc
        self.padded_length = side * side * rounds

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.syndromes[idx], self.labels[idx]


class AnalogDatasetTorch(Dataset):
    """
    PyTorch Dataset for analog IQ readout classification.

    Each sample: (iq_trajectory [n_qubits, n_time_bins, 2], true_state [n_qubits]).
    """

    def __init__(self, data: AnalogDataset, single_qubit_idx: Optional[int] = None):
        """
        Parameters
        ----------
        single_qubit_idx : int, optional
            If given, return single-qubit data (for binary classification).
        """
        self.qi = single_qubit_idx

        traj = data.trajectories  # (N, Q, T, 2)
        states = data.true_states  # (N, Q)

        if single_qubit_idx is not None:
            self.trajectories = torch.from_numpy(traj[:, single_qubit_idx])  # (N, T, 2)
            self.labels = torch.from_numpy(states[:, single_qubit_idx]).long()
        else:
            self.trajectories = torch.from_numpy(traj)  # (N, Q, T, 2)
            self.labels = torch.from_numpy(states).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.trajectories[idx], self.labels[idx]


# ------------------------------------------------------------------
# DataLoader factory
# ------------------------------------------------------------------

def make_dataloaders(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    cfg: TrainingConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train / val / test DataLoaders from datasets."""
    common = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader
