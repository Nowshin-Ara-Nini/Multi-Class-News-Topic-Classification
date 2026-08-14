"""
Unified evaluation and visualisation pipeline for News Topic Classification.

Consolidates 14+ copy-pasted plotting blocks from the original notebook into
a single ``Evaluator`` class providing:

- Comprehensive metric computation (accuracy, precision, recall, F1 at
  multiple averaging levels, MCC, Cohen's Kappa)
- Confusion matrix heatmaps
- Training-curve plots (loss, accuracy, F1, learning rate)
- Model-comparison bar charts
- Per-class ROC and Precision-Recall curves
- Formatted text reports and comparison DataFrames
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global Matplotlib / Seaborn defaults
# ---------------------------------------------------------------------------

_PLOT_PARAMS: Dict[str, Any] = {
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

# Professional colour palette shared across every plot
_PALETTE: List[str] = sns.color_palette("deep").as_hex()


def _apply_style() -> None:
    """Apply project-wide plot styling (idempotent)."""
    sns.set_style("darkgrid")
    matplotlib.rcParams.update(_PLOT_PARAMS)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Unified evaluation and plotting helper.

    Args:
        class_names: Human-readable class labels (e.g.
            ``['Business', 'Sci/Tech', 'Sports', 'World']``).
        results_dir: Default directory for saved figures / reports.

    Example::

        evaluator = Evaluator(class_names=['Business', 'Sci/Tech',
                                           'Sports', 'World'])
        metrics = evaluator.evaluate(y_true, y_pred, y_proba)
        evaluator.plot_confusion_matrix(y_true, y_pred, title='BiLSTM')
    """

    _DEFAULT_CLASS_NAMES: List[str] = [
        "Business",
        "Science and Technology",
        "Sports",
        "World News",
    ]

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        results_dir: str = "results",
    ) -> None:
        self._class_names = class_names or self._DEFAULT_CLASS_NAMES
        self._results_dir = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)
        _apply_style()
        logger.info(
            "Evaluator ready  |  classes=%s  |  results_dir=%s",
            self._class_names,
            self._results_dir,
        )

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true: Union[List[int], np.ndarray],
        y_pred: Union[List[int], np.ndarray],
        y_proba: Optional[np.ndarray] = None,
        model: Any = None,
        inference_time_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute a comprehensive set of classification metrics.

        Args:
            y_true: Ground-truth integer labels.
            y_pred: Predicted integer labels.
            y_proba: Optional predicted class probabilities of shape
                ``(n_samples, n_classes)``.

        Returns:
            Dictionary containing:

            - ``accuracy``, ``precision_macro``, ``recall_macro``,
              ``f1_micro``, ``f1_macro``, ``f1_weighted``
            - ``mcc`` (Matthews correlation coefficient)
            - ``cohen_kappa``
            - ``confusion_matrix`` (np.ndarray)
            - ``classification_report`` (formatted string)
            - ``per_class_metrics`` (list of dicts)
            - ``roc_auc_macro``, ``roc_auc_per_class`` (if *y_proba*
              is provided)
        """
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        # ---- scalar metrics ----
        acc = accuracy_score(y_true_arr, y_pred_arr)
        prec_macro = precision_score(
            y_true_arr, y_pred_arr, average="macro", zero_division=0
        )
        rec_macro = recall_score(
            y_true_arr, y_pred_arr, average="macro", zero_division=0
        )
        prec_weighted = precision_score(
            y_true_arr, y_pred_arr, average="weighted", zero_division=0
        )
        rec_weighted = recall_score(
            y_true_arr, y_pred_arr, average="weighted", zero_division=0
        )
        f1_micro = f1_score(
            y_true_arr, y_pred_arr, average="micro", zero_division=0
        )
        f1_macro = f1_score(
            y_true_arr, y_pred_arr, average="macro", zero_division=0
        )
        f1_weighted = f1_score(
            y_true_arr, y_pred_arr, average="weighted", zero_division=0
        )
        mcc = matthews_corrcoef(y_true_arr, y_pred_arr)
        kappa = cohen_kappa_score(y_true_arr, y_pred_arr)

        cm = confusion_matrix(y_true_arr, y_pred_arr)
        report_str = classification_report(
            y_true_arr,
            y_pred_arr,
            target_names=self._class_names,
            digits=4,
            zero_division=0,
        )

        # ---- per-class breakdown ----
        per_class: List[Dict[str, Any]] = []
        prec_per = precision_score(
            y_true_arr, y_pred_arr, average=None, zero_division=0
        )
        rec_per = recall_score(
            y_true_arr, y_pred_arr, average=None, zero_division=0
        )
        f1_per = f1_score(
            y_true_arr, y_pred_arr, average=None, zero_division=0
        )
        for idx, name in enumerate(self._class_names):
            per_class.append(
                {
                    "class": name,
                    "precision": float(prec_per[idx]),
                    "recall": float(rec_per[idx]),
                    "f1": float(f1_per[idx]),
                    "support": int((y_true_arr == idx).sum()),
                }
            )

        results: Dict[str, Any] = {
            "accuracy": float(acc),
            "precision_macro": float(prec_macro),
            "precision_weighted": float(prec_weighted),
            "recall_macro": float(rec_macro),
            "recall_weighted": float(rec_weighted),
            "f1_micro": float(f1_micro),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "mcc": float(mcc),
            "matthews_corrcoef": float(mcc),
            "cohen_kappa": float(kappa),
            "confusion_matrix": cm,
            "classification_report": report_str,
            "per_class_metrics": per_class,
        }

        if inference_time_ms is not None:
            results["inference_time_ms"] = float(inference_time_ms)
        if model is not None:
            if hasattr(model, "parameters"):
                results["parameter_count"] = int(sum(p.numel() for p in model.parameters()))
            model_path = getattr(model, "model_path", None)
            if model_path and Path(model_path).is_file():
                results["model_size_mb"] = Path(model_path).stat().st_size / (1024 ** 2)
        try:
            import torch
            results["gpu_memory_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
        except ImportError:
            results["gpu_memory_mb"] = 0.0

        # ---- ROC-AUC (requires probabilities) ----
        if y_proba is not None:
            try:
                num_classes = len(self._class_names)
                y_proba = np.asarray(y_proba)
                if y_proba.ndim != 2 or y_proba.shape[1] != num_classes:
                    raise ValueError(
                        "y_proba must have shape (n_samples, num_classes); "
                        f"received {y_proba.shape} for {num_classes} classes"
                    )
                y_bin = label_binarize(y_true_arr, classes=list(range(num_classes)))
                # sklearn represents binary labels as a single positive-class
                # column.  The per-class loop below deliberately reports both
                # classes, so restore the complementary negative-class column.
                if num_classes == 2:
                    y_bin = np.column_stack((1 - y_bin.ravel(), y_bin.ravel()))
                roc_auc_per: List[float] = []
                for i in range(num_classes):
                    auc_i = roc_auc_score(y_bin[:, i], y_proba[:, i])
                    roc_auc_per.append(float(auc_i))
                roc_auc_mac = float(np.mean(roc_auc_per))
                # ``roc_auc`` is the concise public metric name promised by
                # the CLI/API plan.  Keep the explicit macro name as well for
                # backwards-compatible comparison reports.
                results["roc_auc"] = roc_auc_mac
                results["roc_auc_macro"] = roc_auc_mac
                results["roc_auc_per_class"] = roc_auc_per
            except Exception:
                logger.warning("ROC-AUC computation failed", exc_info=True)

        logger.info(
            "Evaluation complete  |  acc=%.4f  |  f1_macro=%.4f  |  mcc=%.4f",
            acc,
            f1_macro,
            mcc,
        )
        return results

    # ------------------------------------------------------------------
    # Plotting — Confusion Matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        y_true: Union[List[int], np.ndarray],
        y_pred: Union[List[int], np.ndarray],
        title: str = "Confusion Matrix",
        normalize: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot a confusion matrix heatmap.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            title: Plot title.
            normalize: If ``True``, show row-normalised percentages.
            save_path: Optional path to save the figure.
        """
        _apply_style()
        cm = confusion_matrix(np.asarray(y_true), np.asarray(y_pred))

        fig, ax = plt.subplots(figsize=(10, 8))

        if normalize:
            cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
            fmt = ".1f"
            # Build annotation strings with count + percentage
            annot = np.empty_like(cm, dtype=object)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    annot[i, j] = f"{cm[i, j]}\n({cm_display[i, j]:.1f}%)"
            sns.heatmap(
                cm_display,
                annot=annot,
                fmt="",
                cmap="Blues",
                xticklabels=self._class_names,
                yticklabels=self._class_names,
                ax=ax,
                cbar_kws={"label": "Percentage (%)"},
            )
        else:
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self._class_names,
                yticklabels=self._class_names,
                ax=ax,
                cbar_kws={"label": "Count"},
            )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(title)
        plt.tight_layout()

        self._save_figure(fig, save_path, default_name="confusion_matrix")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Plotting — Training Curves
    # ------------------------------------------------------------------

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        title: str = "Training Curves",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot 2×2 training curves: loss, accuracy, F1, learning rate.

        Args:
            history: Dictionary returned by ``Trainer.train()``.  Expected keys:
                ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``,
                ``train_f1``, ``val_f1``, ``lr``.
            title: Suptitle for the figure.
            save_path: Optional path to save the figure.
        """
        _apply_style()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        epochs = range(1, len(history.get("train_loss", [])) + 1)

        # ---- Loss ----
        ax = axes[0, 0]
        ax.plot(epochs, history["train_loss"], label="Train", color=_PALETTE[0], linewidth=2)
        ax.plot(epochs, history["val_loss"], label="Validation", color=_PALETTE[3], linewidth=2)
        ax.set_title("Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

        # ---- Accuracy ----
        ax = axes[0, 1]
        ax.plot(epochs, history["train_acc"], label="Train", color=_PALETTE[0], linewidth=2)
        ax.plot(epochs, history["val_acc"], label="Validation", color=_PALETTE[3], linewidth=2)
        ax.set_title("Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()

        # ---- F1 Score ----
        ax = axes[1, 0]
        ax.plot(epochs, history["train_f1"], label="Train", color=_PALETTE[0], linewidth=2)
        ax.plot(epochs, history["val_f1"], label="Validation", color=_PALETTE[3], linewidth=2)
        ax.set_title("F1 Score (Macro)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1")
        ax.legend()

        # ---- Learning Rate ----
        ax = axes[1, 1]
        ax.plot(epochs, history["lr"], color=_PALETTE[2], linewidth=2)
        ax.set_title("Learning Rate")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        self._save_figure(fig, save_path, default_name="training_curves")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Plotting — Model Comparison
    # ------------------------------------------------------------------

    def plot_comparison_bar(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot a grouped bar chart comparing models across metrics.

        Args:
            results: Mapping ``{model_name: {metric_name: value, ...}, ...}``.
            metrics: Which metrics to include.  Defaults to
                ``['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']``.
            save_path: Optional path to save the figure.
        """
        _apply_style()
        if metrics is None:
            metrics = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]

        model_names = list(results.keys())
        n_models = len(model_names)
        n_metrics = len(metrics)

        x = np.arange(n_models)
        bar_width = 0.8 / n_metrics

        fig, ax = plt.subplots(figsize=(max(10, n_models * 2), 8))

        for i, metric in enumerate(metrics):
            values = [results[m].get(metric, 0.0) for m in model_names]
            offset = (i - n_metrics / 2 + 0.5) * bar_width
            bars = ax.bar(
                x + offset,
                values,
                bar_width,
                label=metric.replace("_", " ").title(),
                color=_PALETTE[i % len(_PALETTE)],
                edgecolor="white",
                linewidth=0.5,
            )
            # Value labels on top of each bar
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=30, ha="right")
        ax.set_ylim(0, 1.08)
        ax.legend(loc="lower right")
        plt.tight_layout()

        self._save_figure(fig, save_path, default_name="model_comparison")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Plotting — ROC Curves
    # ------------------------------------------------------------------

    def plot_roc_curves(
        self,
        y_true: Union[List[int], np.ndarray],
        y_proba: np.ndarray,
        title: str = "ROC Curves",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot per-class ROC curves and a macro-average curve.

        Args:
            y_true: Ground-truth integer labels.
            y_proba: Predicted probabilities ``(n_samples, n_classes)``.
            title: Plot title.
            save_path: Optional path to save the figure.
        """
        _apply_style()
        y_true_arr = np.asarray(y_true)
        n_classes = len(self._class_names)
        y_bin = label_binarize(y_true_arr, classes=list(range(n_classes)))

        fig, ax = plt.subplots(figsize=(10, 8))

        fpr_dict: Dict[int, np.ndarray] = {}
        tpr_dict: Dict[int, np.ndarray] = {}
        roc_auc_dict: Dict[int, float] = {}

        for i in range(n_classes):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
            ax.plot(
                fpr_dict[i],
                tpr_dict[i],
                label=f"{self._class_names[i]} (AUC = {roc_auc_dict[i]:.3f})",
                linewidth=2,
                color=_PALETTE[i % len(_PALETTE)],
            )

        # Macro-average ROC
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= n_classes
        macro_auc = auc(all_fpr, mean_tpr)

        ax.plot(
            all_fpr,
            mean_tpr,
            label=f"Macro Avg (AUC = {macro_auc:.3f})",
            linewidth=2.5,
            linestyle="--",
            color="black",
        )

        # Diagonal
        ax.plot([0, 1], [0, 1], "k:", linewidth=1, alpha=0.5)

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        plt.tight_layout()

        self._save_figure(fig, save_path, default_name="roc_curves")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Plotting — Precision-Recall Curves
    # ------------------------------------------------------------------

    def plot_precision_recall_curves(
        self,
        y_true: Union[List[int], np.ndarray],
        y_proba: np.ndarray,
        title: str = "Precision-Recall Curves",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot per-class Precision-Recall curves.

        Args:
            y_true: Ground-truth integer labels.
            y_proba: Predicted probabilities ``(n_samples, n_classes)``.
            title: Plot title.
            save_path: Optional path to save the figure.
        """
        _apply_style()
        y_true_arr = np.asarray(y_true)
        n_classes = len(self._class_names)
        y_bin = label_binarize(y_true_arr, classes=list(range(n_classes)))

        fig, ax = plt.subplots(figsize=(10, 8))

        for i in range(n_classes):
            prec_vals, rec_vals, _ = precision_recall_curve(
                y_bin[:, i], y_proba[:, i]
            )
            pr_auc = auc(rec_vals, prec_vals)
            ax.plot(
                rec_vals,
                prec_vals,
                label=f"{self._class_names[i]} (AUC = {pr_auc:.3f})",
                linewidth=2,
                color=_PALETTE[i % len(_PALETTE)],
            )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        plt.tight_layout()

        self._save_figure(fig, save_path, default_name="precision_recall_curves")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        results: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> str:
        """Generate a formatted text report comparing all models.

        Args:
            results: ``{model_name: metrics_dict, ...}`` where each
                *metrics_dict* was returned by :meth:`evaluate`.
            save_path: Optional filepath to write the report.

        Returns:
            The report as a string.
        """
        lines: List[str] = [
            "=" * 80,
            "NEWS TOPIC CLASSIFICATION — EVALUATION REPORT",
            "=" * 80,
            "",
        ]

        for model_name, metrics in results.items():
            lines.append(f"Model: {model_name}")
            lines.append("-" * 60)
            lines.append(f"  Accuracy:        {metrics.get('accuracy', 0):.4f}")
            lines.append(f"  F1 (micro):      {metrics.get('f1_micro', 0):.4f}")
            lines.append(f"  F1 (macro):      {metrics.get('f1_macro', 0):.4f}")
            lines.append(f"  F1 (weighted):   {metrics.get('f1_weighted', 0):.4f}")
            lines.append(f"  Precision:       {metrics.get('precision_macro', 0):.4f}")
            lines.append(f"  Recall:          {metrics.get('recall_macro', 0):.4f}")
            lines.append(f"  MCC:             {metrics.get('mcc', 0):.4f}")
            lines.append(f"  Cohen's Kappa:   {metrics.get('cohen_kappa', 0):.4f}")

            if "roc_auc_macro" in metrics:
                lines.append(f"  ROC-AUC (macro): {metrics['roc_auc_macro']:.4f}")

            report_str = metrics.get("classification_report")
            if report_str:
                lines.append("")
                lines.append("  Classification Report:")
                for line in report_str.split("\n"):
                    lines.append(f"    {line}")

            lines.append("")

        lines.append("=" * 80)
        report = "\n".join(lines)

        if save_path is not None:
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            try:
                with open(save_path, "w", encoding="utf-8") as fh:
                    fh.write(report)
                logger.info("Report saved -> %s", save_path)
            except Exception:
                logger.exception("Failed to save report to %s", save_path)

        return report

    # ------------------------------------------------------------------
    # Model comparison table
    # ------------------------------------------------------------------

    def compare_models(
        self,
        all_results: Dict[str, Dict[str, Any]],
        sort_by: str = "f1_macro",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Build a comparison DataFrame from multiple model results.

        Args:
            all_results: ``{model_name: metrics_dict, ...}``.
            sort_by: Column to sort by.
            ascending: Sort order.

        Returns:
            ``pd.DataFrame`` with one row per model and columns for each
            scalar metric.
        """
        scalar_keys = [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_micro",
            "f1_macro",
            "f1_weighted",
            "mcc",
            "cohen_kappa",
            "roc_auc_macro",
        ]

        rows: List[Dict[str, Any]] = []
        for name, metrics in all_results.items():
            row: Dict[str, Any] = {"model": name}
            for key in scalar_keys:
                row[key] = metrics.get(key, np.nan)
            rows.append(row)

        df = pd.DataFrame(rows).set_index("model")

        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)

        logger.info("Model comparison table (%d models):\n%s", len(df), df.to_string())
        return df

    def generate_markdown_table(
        self, all_results: Dict[str, Dict[str, Any]], sort_by: str = "f1_macro"
    ) -> str:
        """Return a compact Markdown table suitable for a README."""
        df = self.compare_models(all_results, sort_by=sort_by)
        columns = [c for c in ("accuracy", "f1_macro", "f1_weighted", "mcc", "cohen_kappa") if c in df]
        table = df[columns].copy()
        for column in columns:
            table[column] = table[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "—")
        return table.to_markdown()

    def generate_latex_table(
        self, all_results: Dict[str, Dict[str, Any]], sort_by: str = "f1_macro"
    ) -> str:
        """Return a publication-ready LaTeX comparison table."""
        df = self.compare_models(all_results, sort_by=sort_by)
        columns = [c for c in ("accuracy", "f1_macro", "f1_weighted", "mcc", "cohen_kappa") if c in df]
        return df[columns].to_latex(float_format="%.4f", caption="Model comparison", label="tab:model-comparison")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_figure(
        self,
        fig: plt.Figure,
        save_path: Optional[str],
        default_name: str,
    ) -> None:
        """Save a figure to disk with fallback naming.

        Args:
            fig: Matplotlib figure to save.
            save_path: Explicit path; if ``None`` uses *default_name*
                inside ``self._results_dir``.
            default_name: Filename stem used when *save_path* is ``None``.
        """
        if save_path is None:
            save_path = str(self._results_dir / f"{default_name}.png")
        else:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Figure saved -> %s", save_path)
        except Exception:
            logger.exception("Failed to save figure to %s", save_path)
