"""
qec_ml.benchmarks.visualization
=================================
Publication-quality plotting utilities for QEC benchmark results.

All functions return matplotlib Figure objects so they can be
displayed inline in notebooks or saved to disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, List, Optional, Tuple, Any


# ── Shared style ──────────────────────────────────────────────────────

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "legend.framealpha": 0.9,
})


# ======================================================================
# 1. Bar chart: decoder comparison
# ======================================================================

def plot_decoder_comparison(
    df: pd.DataFrame,
    metric: str = "logical_error_rate",
    title: str = "Decoder Comparison",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Horizontal bar chart comparing decoders on a single metric.

    Parameters
    ----------
    df : pd.DataFrame — output of BenchmarkRunner.run()
    metric : column name to plot
    """
    fig, ax = _get_ax(ax)
    df_sorted = df.sort_values(metric, ascending=True)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df_sorted))]

    bars = ax.barh(df_sorted["decoder"], df_sorted[metric], color=colors, height=0.6)
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(title, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ======================================================================
# 2. LER vs noise rate
# ======================================================================

def plot_ler_vs_noise(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    threshold: Optional[float] = None,
    title: str = "Logical Error Rate vs Physical Error Rate",
    ax: Optional[plt.Axes] = None,
    log_scale: bool = True,
) -> plt.Figure:
    """
    Plot LER-vs-p curves for multiple decoders.

    Parameters
    ----------
    curves : {decoder_name: (noise_rates, lers)}
    threshold : float, optional — draws a vertical dashed line at p*
    """
    fig, ax = _get_ax(ax, figsize=(7, 5))

    for i, (name, (ps, lers)) in enumerate(curves.items()):
        ax.plot(
            ps, lers,
            marker=MARKERS[i % len(MARKERS)],
            color=PALETTE[i % len(PALETTE)],
            label=name,
            linewidth=2,
            markersize=6,
        )

    if threshold is not None:
        ax.axvline(threshold, color="gray", linestyle="--", linewidth=1.5,
                   label=f"Threshold p*={threshold:.3f}")

    ax.set_xlabel("Physical Error Rate p")
    ax.set_ylabel("Logical Error Rate")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
    if log_scale:
        ax.set_yscale("log")
    fig.tight_layout()
    return fig


# ======================================================================
# 3. LER vs code distance
# ======================================================================

def plot_ler_vs_distance(
    curves: Dict[str, Tuple[List[int], np.ndarray]],
    p: float,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot LER vs code distance d for multiple decoders."""
    fig, ax = _get_ax(ax, figsize=(6, 5))
    if title is None:
        title = f"LER vs Code Distance (p={p:.3f})"

    for i, (name, (dists, lers)) in enumerate(curves.items()):
        ax.plot(
            dists, lers,
            marker=MARKERS[i % len(MARKERS)],
            color=PALETTE[i % len(PALETTE)],
            label=name,
            linewidth=2,
            markersize=7,
        )

    ax.set_xlabel("Code Distance d")
    ax.set_ylabel("Logical Error Rate")
    ax.set_title(title, fontweight="bold")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()
    fig.tight_layout()
    return fig


# ======================================================================
# 4. Training curves
# ======================================================================

def plot_training_curves(
    history,
    model_name: str = "",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot train/val loss and accuracy over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history.train_loss) + 1)
    ax1.plot(epochs, history.train_loss, label="Train", color=PALETTE[0])
    ax1.plot(epochs, history.val_loss, label="Val", color=PALETTE[1], linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} — Loss", fontweight="bold")
    ax1.legend()

    ax2.plot(epochs, history.train_acc, label="Train", color=PALETTE[0])
    ax2.plot(epochs, history.val_acc, label="Val", color=PALETTE[1], linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{model_name} — Accuracy", fontweight="bold")
    ax2.legend()

    fig.tight_layout()
    return fig


# ======================================================================
# 5. IQ scatter plot
# ======================================================================

def plot_iq_scatter(
    iq_points: np.ndarray,
    labels: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    title: str = "IQ Plane",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Scatter plot of IQ readout points coloured by qubit state.

    Parameters
    ----------
    iq_points : (N, 2) array — [I, Q]
    labels : (N,) int array — 0 or 1
    predictions : (N,) int array, optional — mark misclassified points
    """
    fig, ax = _get_ax(ax, figsize=(6, 6))

    for state, color, label_str in zip([0, 1], [PALETTE[0], PALETTE[1]], ["|0⟩", "|1⟩"]):
        mask = labels == state
        ax.scatter(
            iq_points[mask, 0], iq_points[mask, 1],
            c=color, label=label_str, alpha=0.4, s=15, linewidths=0,
        )

    if predictions is not None:
        wrong = predictions != labels
        ax.scatter(
            iq_points[wrong, 0], iq_points[wrong, 1],
            c="red", marker="x", s=40, label="Misclassified", zorder=5,
        )

    ax.set_xlabel("I (In-phase)")
    ax.set_ylabel("Q (Quadrature)")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


# ======================================================================
# 6. Syndrome heatmap
# ======================================================================

def plot_syndrome_heatmap(
    syndromes: np.ndarray,
    distance: int,
    n_show: int = 4,
    title: str = "Syndrome Patterns",
) -> plt.Figure:
    """
    Show a grid of syndrome patterns as 2D heatmaps.

    Parameters
    ----------
    syndromes : (N, L) array
    distance : int — code distance
    n_show : int — number of examples to show
    """
    side = distance - 1
    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3))
    if n_show == 1:
        axes = [axes]

    for ax, syn in zip(axes, syndromes[:n_show]):
        truncated = syn[:side * side].reshape(side, side)
        im = ax.imshow(truncated, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Σ={syn.sum()}", fontsize=10)

    plt.colorbar(im, ax=axes, shrink=0.8, label="Syndrome bit")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


# ======================================================================
# 7. Confusion matrix
# ======================================================================

def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None,
    title: str = "Confusion Matrix",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot a normalised confusion matrix."""
    from sklearn.metrics import confusion_matrix as sk_cm

    if labels is None:
        labels = ["No error", "Logical error"]

    cm = sk_cm(y_true, y_pred, normalize="true")
    fig, ax = _get_ax(ax, figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black")
    fig.tight_layout()
    return fig


# ======================================================================
# Helpers
# ======================================================================

def _get_ax(
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax
