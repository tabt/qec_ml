"""
qec_ml.utils.training
=======================
Generic training / evaluation loop for all ML decoders.

Supports:
  - Binary classification (syndrome → logical error)
  - Multi-class classification (IQ → qubit state)
  - Regression / MSE (autoencoder denoising)
  - Mixed-precision training (AMP)
  - Cosine LR schedule with linear warmup
  - Early stopping
  - Checkpoint save/load
"""

from __future__ import annotations

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, Any, Callable, Literal
from dataclasses import dataclass, field

from qec_ml.utils.config import TrainingConfig


@dataclass
class TrainingHistory:
    """Collects per-epoch metrics."""
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    train_acc: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)
    lr: list = field(default_factory=list)
    epoch_times: list = field(default_factory=list)

    def best_val_loss(self) -> float:
        return min(self.val_loss) if self.val_loss else float("inf")

    def best_val_acc(self) -> float:
        return max(self.val_acc) if self.val_acc else 0.0


class Trainer:
    """
    General-purpose trainer for binary / multi-class classification.

    Parameters
    ----------
    model : nn.Module
    config : TrainingConfig
    loss_fn : callable, optional
        Defaults to BCEWithLogitsLoss for binary, CrossEntropyLoss otherwise.
    n_classes : int
        1 → binary (logit), >1 → multi-class (logits over classes).

    Examples
    --------
    >>> trainer = Trainer(model, cfg)
    >>> history = trainer.fit(train_loader, val_loader)
    >>> trainer.load_best()
    >>> results = trainer.evaluate(test_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        loss_fn: Optional[Callable] = None,
        n_classes: int = 1,
    ):
        self.model = model
        self.cfg = config
        self.n_classes = n_classes
        self.device = torch.device(config.resolve_device())
        self.model.to(self.device)

        if loss_fn is not None:
            self.loss_fn = loss_fn
        elif n_classes == 1:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))
        self._best_val_loss = float("inf")
        self._patience_counter = 0
        self._best_state: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainingHistory:
        """Run the full training loop."""
        cfg = self.cfg
        history = TrainingHistory()
        scheduler = self._make_scheduler(len(train_loader))

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.perf_counter()
            train_loss, train_acc = self._train_epoch(train_loader, scheduler)
            val_loss, val_acc = self._eval_epoch(val_loader)
            elapsed = time.perf_counter() - t0

            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            history.train_acc.append(train_acc)
            history.val_acc.append(val_acc)
            history.lr.append(self.optimizer.param_groups[0]["lr"])
            history.epoch_times.append(elapsed)

            print(
                f"Epoch {epoch:3d}/{cfg.epochs} | "
                f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
                f"{elapsed:.1f}s"
            )

            # Save best model
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self._patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
            else:
                self._patience_counter += 1

            if self._patience_counter >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        return history

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate on a DataLoader, return loss and accuracy."""
        loss, acc = self._eval_epoch(loader)
        return {"loss": loss, "accuracy": acc, "logical_error_rate": 1.0 - acc}

    def load_best(self) -> None:
        """Restore the best checkpoint weights."""
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            self.model.to(self.device)

    def predict_proba(self, loader: DataLoader) -> np.ndarray:
        """Return sigmoid probabilities for binary or softmax for multi-class."""
        self.model.eval()
        probs = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                logits = self.model(x)
                if self.n_classes == 1:
                    p = torch.sigmoid(logits).cpu().numpy()
                else:
                    p = torch.softmax(logits, dim=-1).cpu().numpy()
                probs.append(p)
        return np.concatenate(probs, axis=0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self, loader: DataLoader, scheduler
    ):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(self.device.type == "cuda")):
                logits = self.model(x)
                if self.n_classes == 1:
                    loss = self.loss_fn(logits, y.float())
                    preds = (logits > 0).long()
                else:
                    loss = self.loss_fn(logits, y)
                    preds = logits.argmax(dim=-1)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.gradient_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scheduler.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_correct += (preds == y).sum().item()
            total += bs

        return total_loss / total, total_correct / total

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total = 0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            logits = self.model(x)

            if self.n_classes == 1:
                loss = self.loss_fn(logits, y.float())
                preds = (logits > 0).long()
            else:
                loss = self.loss_fn(logits, y)
                preds = logits.argmax(dim=-1)

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_correct += (preds == y).sum().item()
            total += bs

        return total_loss / total, total_correct / total

    def _make_scheduler(self, steps_per_epoch: int):
        cfg = self.cfg
        total_steps = cfg.epochs * steps_per_epoch
        warmup_steps = cfg.warmup_epochs * steps_per_epoch

        if cfg.scheduler == "cosine":
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(warmup_steps, 1)
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return 0.5 * (1 + math.cos(math.pi * progress))
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif cfg.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=steps_per_epoch * 10, gamma=0.5
            )
        else:
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda s: 1.0)

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        path = os.path.join(
            self.cfg.checkpoint_dir,
            f"best_{self.cfg.model_type}.pt"
        )
        torch.save({
            "epoch": epoch,
            "val_loss": val_loss,
            "state_dict": self._best_state,
            "config": self.cfg,
        }, path)
