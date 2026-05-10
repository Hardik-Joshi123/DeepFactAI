"""
Model Trainer
--------------
Reusable training loop for PyTorch models with:
  - train / validation split
  - early stopping
  - model checkpointing
  - learning rate scheduling
  - per-epoch accuracy and loss logging
  - matplotlib training curves

Usage:
  trainer = Trainer(model, optimizer, criterion, device, save_dir="saved_models")
  history = trainer.train(train_loader, val_loader, epochs=10)
  trainer.plot_history(history)
"""

from __future__ import annotations

import copy
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when val_loss has not improved for `patience` epochs."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, restore_best: bool = True) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self._best_loss = float("inf")
        self._counter = 0
        self.best_weights: dict | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """
        Returns True if training should stop.
        Saves best model weights internally if restore_best is True.
        """
        if val_loss < self._best_loss - self.min_delta:
            self._best_loss = val_loss
            self._counter = 0
            if self.restore_best:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self._counter += 1

        return self._counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        """Restore model to the best observed weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info("Restored best model weights.")


class Trainer:
    """
    Generic PyTorch training loop.

    Args:
        model:      nn.Module to train
        optimizer:  torch.optim optimizer
        criterion:  loss function (e.g. BCEWithLogitsLoss)
        device:     'cpu' | 'cuda' | 'mps'
        save_dir:   Directory to save checkpoints
        clip_grad:  Gradient clipping max norm (None to disable)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cpu",
        save_dir: str = "saved_models",
        clip_grad: Optional[float] = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.clip_grad = clip_grad

    # ── training loop ─────────────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        patience: int = 5,
        scheduler=None,
        checkpoint_name: str = "best_model.pt",
    ) -> dict:
        """
        Train the model and return the history dict.

        Returns:
            history: {
                'train_loss': [...], 'train_acc': [...],
                'val_loss': [...],   'val_acc': [...]
            }
        """
        early_stop = EarlyStopping(patience=patience)
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._run_epoch(train_loader, training=True)
            val_loss, val_acc = self._run_epoch(val_loader, training=False)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            elapsed = time.time() - t0
            logger.info(
                "Epoch %3d/%d | %.1fs | train_loss %.4f train_acc %.4f | "
                "val_loss %.4f val_acc %.4f",
                epoch, epochs, elapsed, train_loss, train_acc, val_loss, val_acc,
            )

            if scheduler:
                scheduler.step(val_loss)

            if early_stop.step(val_loss, self.model):
                logger.info("Early stopping triggered at epoch %d.", epoch)
                early_stop.restore(self.model)
                break

        # Save final checkpoint
        self.save_checkpoint(checkpoint_name)
        return history

    def _run_epoch(self, loader: DataLoader, training: bool) -> tuple[float, float]:
        self.model.train(training)
        total_loss = 0.0
        correct = 0
        total = 0

        ctx = torch.enable_grad if training else torch.no_grad
        with ctx():
            for batch in loader:
                inputs, labels = self._unpack_batch(batch)
                if training:
                    self.optimizer.zero_grad()

                outputs = self._forward(inputs)
                loss = self.criterion(outputs, labels.float())

                if training:
                    loss.backward()
                    if self.clip_grad:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                    self.optimizer.step()

                total_loss += loss.item() * len(labels)
                preds = (torch.sigmoid(outputs) >= 0.5).long()
                correct += (preds == labels).sum().item()
                total += len(labels)

        return total_loss / total, correct / total

    def _unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            inputs = batch[0].to(self.device)
            labels = batch[1].to(self.device)
        elif isinstance(batch, dict):
            labels = batch.pop("labels").to(self.device)
            inputs = {k: v.to(self.device) for k, v in batch.items()}
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
        return inputs, labels

    def _forward(self, inputs):
        if isinstance(inputs, dict):
            out = self.model(**inputs)
        else:
            out = self.model(inputs)
        # Handle tuple output (e.g. LSTM with attention)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.squeeze(-1)

    # ── checkpoint ────────────────────────────────────────────────────────────

    def save_checkpoint(self, name: str) -> None:
        path = self.save_dir / name
        torch.save(self.model.state_dict(), path)
        logger.info("Checkpoint saved → %s", path)

    def load_checkpoint(self, name: str) -> None:
        path = self.save_dir / name
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info("Checkpoint loaded ← %s", path)

    # ── visualisation ─────────────────────────────────────────────────────────

    @staticmethod
    def plot_history(history: dict, save_path: str = "training_curves.png") -> None:
        """Plot and save accuracy + loss curves."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available — skipping plot.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        epochs = range(1, len(history["train_loss"]) + 1)

        ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
        ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
        ax1.set_title("Training & Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy", markersize=4)
        ax2.plot(epochs, history["val_acc"], "r-o", label="Val Accuracy", markersize=4)
        ax2.set_title("Training & Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Training curves saved → %s", save_path)
