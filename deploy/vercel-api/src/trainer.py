"""
Unified training pipeline for News Topic Classification models.

Consolidates all duplicate training loops (6+ copies from the notebook)
into a single, configurable ``Trainer`` class with:

- Early stopping (monitor val_f1 / val_loss / val_acc)
- LR scheduling (ReduceLROnPlateau, CosineAnnealing)
- Gradient clipping
- Mixed-precision training (AMP)
- Class-weighted CrossEntropyLoss
- Best / last checkpoint saving with full state
- Per-epoch history tracking for downstream plotting
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR, LambdaLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from configs.config import TrainingConfig
from src.utils import clear_gpu_memory, format_time, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Unified training & validation driver for all PyTorch models.

    Replaces every hand-written training loop in the project with a single,
    tested, configurable class.

    Args:
        model: PyTorch model to train.
        config: ``TrainingConfig`` dataclass with all hyper-parameters.
        device: Target device (``torch.device``).
        class_weights: Optional 1-D tensor of per-class weights for the loss.
        experiment_name: Human-readable tag used in log messages and
            checkpoint filenames.

    Example::

        trainer = Trainer(model, config, device, class_weights=weights,
                          experiment_name='bilstm_tfidf')
        history = trainer.train(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
        class_weights: Optional[torch.Tensor] = None,
        experiment_name: str = "experiment",
        experiment_tracker: Optional[Any] = None,
        experiment_run_id: Optional[str] = None,
    ) -> None:
        self._model = model.to(device)
        self._config = config
        self._device = device
        self._experiment_name = experiment_name
        self._model_name: Optional[str] = None
        self._model_kwargs: Dict[str, Any] = {}
        self._experiment_tracker = experiment_tracker
        self._experiment_run_id = experiment_run_id
        if self._experiment_tracker is not None and not self._experiment_run_id:
            self._experiment_run_id = self._experiment_tracker.start_run(
                experiment_name, {"training": asdict(config)}
            )

        # ---- loss function ------------------------------------------------
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self._criterion = nn.CrossEntropyLoss(weight=class_weights)

        # ---- optimizer ----------------------------------------------------
        self._optimizer = self._build_optimizer()

        # ---- lr scheduler -------------------------------------------------
        self._scheduler = self._build_scheduler()
        self._pending_scheduler_state = None
        self.best_epoch = 0
        self.best_metrics = {}

        # ---- mixed precision ----------------------------------------------
        self._use_amp = config.mixed_precision and device.type == "cuda"
        # PyTorch 2.x moved AMP from ``torch.cuda.amp`` to ``torch.amp``.
        # The CUDA device type is explicit in the new API.
        self._scaler = (
            torch.amp.GradScaler("cuda", enabled=True) if self._use_amp else None
        )

        # ---- early stopping state -----------------------------------------
        self._best_metric: float = -float("inf") if self._higher_is_better else float("inf")
        self._best_model_state: Optional[Dict[str, Any]] = None
        self._patience_counter: int = 0

        # ---- history tracking ---------------------------------------------
        self._history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_f1": [],
            "val_f1": [],
            "lr": [],
        }

        # ---- epoch counter (supports checkpoint resume) -------------------
        self._start_epoch: int = 0
        self._current_epoch: int = 0

        logger.info(
            "Trainer initialised  |  experiment=%s  |  optimizer=%s  |  "
            "scheduler=%s  |  amp=%s  |  class_weights=%s",
            experiment_name,
            config.optimizer,
            config.scheduler,
            self._use_amp,
            class_weights is not None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Run the full training loop.

        Args:
            train_loader: Training ``DataLoader``.
            val_loader: Validation ``DataLoader``.
            num_epochs: Override ``config.epochs`` if provided.

        Returns:
            Training history dictionary with keys
            ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``,
            ``train_f1``, ``val_f1``, ``lr``.
        """
        epochs = num_epochs or self._config.epochs
        logger.info(
            "Starting training  |  epochs=%d  |  batches/epoch=%d  |  device=%s",
            epochs,
            len(train_loader),
            self._device,
        )

        if self._config.scheduler == "linear_warmup":
            import math
            total = max(1, math.ceil(len(train_loader) / self._config.accumulation_steps) * epochs)
            warmup = max(1, int(total * self._config.warmup_ratio))
            def schedule(step):
                return step / warmup if step < warmup else max(0., (total - step) / max(1, total - warmup))
            self._scheduler = LambdaLR(self._optimizer, schedule)
            if self._pending_scheduler_state is not None:
                self._scheduler.load_state_dict(self._pending_scheduler_state)
        for epoch in range(self._start_epoch, epochs):
            self._current_epoch = epoch
            epoch_start = time.perf_counter()

            # --- train -----
            train_metrics = self._train_epoch(train_loader)

            # --- validate --
            val_metrics = self._validate(val_loader)

            # --- lr --------
            current_lr = self._optimizer.param_groups[0]["lr"]
            self._step_scheduler(val_metrics)
            new_lr = self._optimizer.param_groups[0]["lr"]
            if new_lr != current_lr:
                logger.info(
                    "Learning rate changed: %.6f -> %.6f", current_lr, new_lr
                )

            # --- history ---
            self._history["train_loss"].append(train_metrics["loss"])
            self._history["val_loss"].append(val_metrics["loss"])
            self._history["train_acc"].append(train_metrics["acc"])
            self._history["val_acc"].append(val_metrics["acc"])
            self._history["train_f1"].append(train_metrics["f1"])
            self._history["val_f1"].append(val_metrics["f1"])
            self._history["lr"].append(current_lr)

            elapsed = time.perf_counter() - epoch_start

            # --- logging ---
            logger.info(
                "Epoch %d/%d | Train Loss: %.4f | Train Acc: %.4f | Train F1: %.4f",
                epoch + 1,
                epochs,
                train_metrics["loss"],
                train_metrics["acc"],
                train_metrics["f1"],
            )
            logger.info(
                "           | Val Loss:   %.4f | Val Acc:   %.4f | Val F1:   %.4f",
                val_metrics["loss"],
                val_metrics["acc"],
                val_metrics["f1"],
            )
            logger.info(
                "           | LR: %.6f | Time: %s",
                current_lr,
                format_time(elapsed),
            )
            if self._experiment_tracker is not None and self._experiment_run_id:
                self._experiment_tracker.log_epoch(
                    self._experiment_run_id,
                    epoch + 1,
                    {
                        "train_loss": train_metrics["loss"],
                        "train_acc": train_metrics["acc"],
                        "train_f1": train_metrics["f1"],
                        "val_loss": val_metrics["loss"],
                        "val_acc": val_metrics["acc"],
                        "val_f1": val_metrics["f1"],
                        "learning_rate": current_lr,
                        "epoch_time_seconds": elapsed,
                    },
                )

            # --- checkpointing ---
            val_metric_value = self._get_monitor_value(val_metrics)
            is_best = self._is_improvement(val_metric_value)

            if is_best:
                self._best_metric = val_metric_value
                self.best_epoch = epoch + 1
                self.best_metrics = {"best_val_f1": val_metrics["f1"], "best_val_acc": val_metrics["acc"], "best_epoch": self.best_epoch}
                self._best_model_state = copy.deepcopy(self._model.state_dict())
                self._patience_counter = 0
                logger.info(
                    "  >> New best %s: %.4f", self._config.monitor, val_metric_value
                )
                if self._config.save_best:
                    self._auto_save_checkpoint(is_best=True)
            else:
                self._patience_counter += 1

            if self._config.save_last:
                self._auto_save_checkpoint(is_best=False)

            # --- early stopping ---
            if self._config.early_stopping and self._should_stop_early(val_metric_value):
                logger.info(
                    "Early stopping triggered after %d epochs (patience=%d)",
                    epoch + 1,
                    self._config.patience,
                )
                break

        # Restore best model weights
        if self._best_model_state is not None:
            self._model.load_state_dict(self._best_model_state)
            logger.info(
                "Restored best model weights (best %s=%.4f)",
                self._config.monitor,
                self._best_metric,
            )

        if self._experiment_tracker is not None and self._experiment_run_id:
            self._experiment_tracker.finish_run(
                self._experiment_run_id,
                {
                    "best_metric": self._best_metric,
                    "best_epoch": self.best_epoch,
                    "best_metrics": self.best_metrics,
                    "monitor": self._config.monitor,
                    "epochs_completed": len(self._history["val_f1"]),
                    "history": self._history,
                },
            )

        return self._history

    @property
    def history(self) -> Dict[str, List[float]]:
        """Return the full training history dictionary.

        Keys:
            ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``,
            ``train_f1``, ``val_f1``, ``lr``.
        """
        return self._history

    @property
    def model(self) -> nn.Module:
        """Return the underlying model (with best weights after training)."""
        return self._model

    def set_model_metadata(self, registry_name: str, **model_kwargs: Any) -> None:
        """Store reconstruction metadata in subsequently saved checkpoints."""
        self._model_name = registry_name
        self._model_kwargs = model_kwargs

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, is_best: bool = False) -> None:
        """Persist a full training checkpoint.

        Saves model state, optimizer state, scheduler state, epoch counter,
        best metric value, and training history.

        Args:
            path: Destination filepath (e.g. ``checkpoints/bilstm_best.pt``).
            is_best: Whether this checkpoint is the current best.
        """
        checkpoint_dir = Path(path).parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint: Dict[str, Any] = {
            "epoch": self._current_epoch,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "best_metric": self._best_metric,
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
            "history": self._history,
            "config": {
                "experiment_name": self._experiment_name,
                "monitor": self._config.monitor,
                "patience": self._config.patience,
            },
            "model_name": self._model_name,
            "model_kwargs": self._model_kwargs,
            "is_best": is_best,
        }
        if self._scheduler is not None:
            checkpoint["scheduler_state_dict"] = self._scheduler.state_dict()
        if self._scaler is not None:
            checkpoint["scaler_state_dict"] = self._scaler.state_dict()

        try:
            torch.save(checkpoint, path)
            logger.debug("Checkpoint saved -> %s", path)
        except Exception:
            logger.exception("Failed to save checkpoint to %s", path)
            raise

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Restore training state from a checkpoint.

        Loads model weights, optimizer state, scheduler state, epoch counter,
        best metric, and history so that training can seamlessly resume.

        Args:
            path: Checkpoint filepath to load.

        Returns:
            The raw checkpoint dictionary for inspection.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        try:
            checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        except Exception:
            logger.exception("Failed to load checkpoint from %s", path)
            raise

        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            if self._scheduler is not None:
                self._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                self._pending_scheduler_state = checkpoint["scheduler_state_dict"]
        if "scaler_state_dict" in checkpoint and self._scaler is not None:
            self._scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self._start_epoch = checkpoint.get("epoch", 0) + 1
        self._best_metric = checkpoint.get("best_metric", self._best_metric)
        self._history = checkpoint.get("history", self._history)
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.best_metrics = checkpoint.get("best_metrics", {})
        if checkpoint.get("is_best"):
            self._best_model_state = copy.deepcopy(self._model.state_dict())

        logger.info(
            "Checkpoint loaded <- %s  |  resuming from epoch %d  |  best %s=%.4f",
            path,
            self._start_epoch,
            self._config.monitor,
            self._best_metric,
        )
        return checkpoint

    # ------------------------------------------------------------------
    # Core training / validation steps
    # ------------------------------------------------------------------

    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Execute one training epoch.

        Args:
            train_loader: Training ``DataLoader``.

        Returns:
            Dictionary with ``loss``, ``acc``, ``f1``.
        """
        self._model.train()

        running_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []
        num_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            inputs, targets, model_kwargs = self._unpack_batch(batch)

            accumulation = self._config.accumulation_steps
            if batch_idx % accumulation == 0:
                self._optimizer.zero_grad(set_to_none=True)
            group_start = (batch_idx // accumulation) * accumulation
            group_size = min(accumulation, len(train_loader) - group_start)
            with torch.amp.autocast(device_type=self._device.type, enabled=self._use_amp):
                outputs = self._model(inputs, **model_kwargs)
                loss = self._criterion(outputs, targets)
            backward_loss = loss / group_size
            if self._use_amp:
                self._scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()
            if (batch_idx + 1) % accumulation == 0 or batch_idx + 1 == len(train_loader):
                if self._use_amp:
                    self._scaler.unscale_(self._optimizer)
                if self._config.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self._model.parameters(), self._config.gradient_clip)
                stepped = True
                if self._use_amp:
                    old_scale = self._scaler.get_scale()
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                    stepped = self._scaler.get_scale() >= old_scale
                else:
                    self._optimizer.step()
                if stepped and self._config.scheduler == "linear_warmup":
                    self._scheduler.step()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().tolist())

            # Periodic batch-level logging
            if (
                self._config.log_interval > 0
                and (batch_idx + 1) % self._config.log_interval == 0
            ):
                logger.debug(
                    "  batch %d/%d  |  loss=%.4f",
                    batch_idx + 1,
                    len(train_loader),
                    loss.item(),
                )

        epoch_loss = running_loss / max(num_samples, 1)
        epoch_acc = np.mean(
            np.array(all_preds) == np.array(all_targets)
        ).item()
        epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

        return {"loss": epoch_loss, "acc": epoch_acc, "f1": epoch_f1}

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Dict[str, Any]:
        """Run a validation pass (no gradient computation).

        Args:
            val_loader: Validation ``DataLoader``.

        Returns:
            Dictionary with ``loss``, ``acc``, ``f1``, ``preds``, ``targets``.
        """
        self._model.eval()

        running_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []
        num_samples = 0

        for batch in val_loader:
            inputs, targets, model_kwargs = self._unpack_batch(batch)

            if self._use_amp:
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    outputs = self._model(inputs, **model_kwargs)
                    loss = self._criterion(outputs, targets)
            else:
                outputs = self._model(inputs, **model_kwargs)
                loss = self._criterion(outputs, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().tolist())

        epoch_loss = running_loss / max(num_samples, 1)
        epoch_acc = np.mean(
            np.array(all_preds) == np.array(all_targets)
        ).item()
        epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

        return {
            "loss": epoch_loss,
            "acc": epoch_acc,
            "f1": epoch_f1,
            "preds": all_preds,
            "targets": all_targets,
        }

    # ------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------

    def _should_stop_early(self, val_metric: float) -> bool:
        """Determine whether training should be stopped.

        The counter is incremented in :meth:`train` whenever the monitored
        metric does **not** improve.  This method only checks whether the
        counter exceeds ``patience``.

        Args:
            val_metric: Current value of the monitored metric.

        Returns:
            ``True`` if training should stop.
        """
        return self._patience_counter >= self._config.patience

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @property
    def _higher_is_better(self) -> bool:
        """Whether a higher monitored metric is better."""
        return self._config.monitor in ("val_f1", "val_acc")

    def _is_improvement(self, current: float) -> bool:
        """Check if *current* is better than *best* by at least ``min_delta``."""
        delta = self._config.min_delta
        if self._higher_is_better:
            return current > self._best_metric + delta
        return current < self._best_metric - delta

    def _get_monitor_value(self, val_metrics: Dict[str, float]) -> float:
        """Extract the monitored value from validation metrics."""
        mapping = {
            "val_f1": "f1",
            "val_acc": "acc",
            "val_loss": "loss",
        }
        key = mapping.get(self._config.monitor)
        if key is None:
            logger.warning(
                "Unknown monitor '%s', defaulting to 'f1'", self._config.monitor
            )
            key = "f1"
        return val_metrics[key]

    def _unpack_batch(
        self, batch: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Move batch data to the target device.

        Supports tuples/lists ``(inputs, targets[, padding_mask])`` as well as dicts with
        ``'input'``/``'inputs'`` and ``'target'``/``'targets'``/``'label'``/
        ``'labels'`` keys.

        Args:
            batch: A single batch from a ``DataLoader``.

        Returns:
            ``(inputs, targets, model_kwargs)`` tensors on ``self._device``.

        Raises:
            ValueError: If the batch format is not recognised.
        """
        if isinstance(batch, (tuple, list)):
            inputs, targets = batch[0], batch[1]
            padding_mask = batch[2] if len(batch) > 2 else None
        elif isinstance(batch, Mapping):
            input_key = next(
                (k for k in ("input", "inputs", "input_ids", "x", "X") if k in batch), None
            )
            target_key = next(
                (
                    k
                    for k in ("target", "targets", "label", "labels", "y", "Y")
                    if k in batch
                ),
                None,
            )
            if input_key is None or target_key is None:
                raise ValueError(
                    f"Dict batch must have input/target keys; got {list(batch.keys())}"
                )
            inputs, targets = batch[input_key], batch[target_key]
            padding_mask = batch.get("padding_mask")
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

        model_kwargs: Dict[str, torch.Tensor] = {}
        if padding_mask is not None:
            model_kwargs["padding_mask"] = padding_mask.to(self._device)
        if isinstance(batch, Mapping) and batch.get("attention_mask") is not None:
            model_kwargs["attention_mask"] = batch["attention_mask"].to(self._device)
        return inputs.to(self._device), targets.to(self._device), model_kwargs

    # ---- optimizer / scheduler factories ---------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Instantiate the optimizer from ``TrainingConfig``."""
        name = self._config.optimizer.lower()
        groups = {}
        for parameter_name, parameter in self._model.named_parameters():
            if not parameter.requires_grad:
                continue
            decay = self._config.weight_decay if parameter.ndim > 1 and not parameter_name.endswith("bias") else 0.0
            multiplier = self._config.head_lr_multiplier if parameter_name.startswith("classifier.") else 1.0
            groups.setdefault((decay, multiplier), []).append(parameter)
        params = [{"params": values, "weight_decay": decay,
                   "lr": self._config.learning_rate * multiplier}
                  for (decay, multiplier), values in groups.items()]

        if name == "adam":
            return Adam(
                params,
                lr=self._config.learning_rate,
                weight_decay=self._config.weight_decay,
            )
        if name == "adamw":
            return AdamW(
                params,
                lr=self._config.learning_rate,
                weight_decay=self._config.weight_decay,
            )
        if name == "sgd":
            return SGD(
                params,
                lr=self._config.learning_rate,
                momentum=0.9,
                weight_decay=self._config.weight_decay,
            )

        raise ValueError(
            f"Unknown optimizer '{name}'. Choose from: adam, adamw, sgd."
        )

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Instantiate the LR scheduler from ``TrainingConfig``."""
        name = self._config.scheduler.lower()

        if name in ("none", "", "linear_warmup"):
            return None

        if name == "reduce_on_plateau":
            mode = "max" if self._higher_is_better else "min"
            return ReduceLROnPlateau(
                self._optimizer,
                mode=mode,
                patience=self._config.scheduler_patience,
                factor=self._config.scheduler_factor,
                min_lr=self._config.scheduler_min_lr,
            )

        if name == "cosine":
            return CosineAnnealingLR(
                self._optimizer,
                T_max=self._config.epochs,
                eta_min=self._config.scheduler_min_lr,
            )

        if name == "step":
            return StepLR(
                self._optimizer,
                step_size=max(self._config.scheduler_patience, 1),
                gamma=self._config.scheduler_factor,
            )

        raise ValueError(
            f"Unknown scheduler '{name}'. "
            "Choose from: reduce_on_plateau, cosine, step, none."
        )

    def _step_scheduler(self, val_metrics: Dict[str, float]) -> None:
        """Advance the LR scheduler by one step."""
        if self._scheduler is None or self._config.scheduler == "linear_warmup":
            return

        if isinstance(self._scheduler, ReduceLROnPlateau):
            metric_value = self._get_monitor_value(val_metrics)
            self._scheduler.step(metric_value)
        else:
            self._scheduler.step()

    def _auto_save_checkpoint(self, is_best: bool) -> None:
        """Save a checkpoint to the default checkpoints directory."""
        ckpt_dir = Path(self._config.checkpoints_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        tag = "best" if is_best else "last"
        filename = f"{self._experiment_name}_{tag}.pt"
        path = str(ckpt_dir / filename)
        self.save_checkpoint(path, is_best=is_best)
