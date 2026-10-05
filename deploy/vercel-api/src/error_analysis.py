"""
Error analysis utilities for News Topic Classification.

Provides detailed analysis of model errors including false positive/negative
breakdown, most confused class pairs, and hardest examples analysis.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """
    Analyze model prediction errors to identify systematic failure patterns.

    Provides:
    - False positive/negative analysis per class
    - Most confused class pairs
    - Hardest examples (highest confidence wrong, lowest confidence correct)
    - Error distribution by text length
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        results_dir: str = "results",
    ):
        """
        Initialize ErrorAnalyzer.

        Args:
            class_names: Human-readable class labels.
            results_dir: Directory to save analysis outputs.
        """
        self.class_names = class_names or [
            "Business", "Science and Technology", "Sports", "World News"
        ]
        self.results_dir = results_dir

    def analyze(
        self,
        texts: List[str],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Run full error analysis.

        Args:
            texts: Original text inputs.
            y_true: Ground truth labels (integer-encoded).
            y_pred: Predicted labels (integer-encoded).
            y_proba: Prediction probabilities (N, num_classes). Optional.

        Returns:
            Dictionary containing all analysis results.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        results = {
            "total_samples": len(y_true),
            "total_errors": int((y_true != y_pred).sum()),
            "error_rate": float((y_true != y_pred).mean()),
        }

        # Per-class analysis
        results["per_class"] = self._per_class_errors(y_true, y_pred)

        # Confused pairs
        results["confused_pairs"] = self._most_confused_pairs(y_true, y_pred)

        # Error examples
        if texts is not None:
            error_mask = y_true != y_pred
            results["error_examples"] = self._get_error_examples(
                texts, y_true, y_pred, error_mask, y_proba
            )

        # Confidence analysis
        if y_proba is not None:
            results["confidence"] = self._confidence_analysis(
                y_true, y_pred, y_proba
            )

        # Error distribution by text length
        if texts is not None:
            results["length_analysis"] = self._length_analysis(
                texts, y_true, y_pred
            )

        logger.info(
            f"Error analysis complete: {results['total_errors']}/{results['total_samples']} "
            f"errors ({results['error_rate']:.4f} error rate)"
        )

        return results

    # Public, composable analysis helpers ---------------------------------

    def most_confused_pairs(self, y_true, y_pred, top_k: int = 5) -> List[Dict]:
        """Return the most common true-class → predicted-class mistakes."""
        return self._most_confused_pairs(np.asarray(y_true), np.asarray(y_pred), top_k)

    def misclassified_examples(self, texts, y_true, y_pred, n: int = 20) -> List[Dict]:
        """Return up to ``n`` misclassified examples in source order."""
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        return self._get_error_examples(texts, y_true, y_pred, y_true != y_pred, max_examples=n)

    def confidence_distribution(self, y_proba: np.ndarray, bins: int = 10) -> Dict:
        """Summarise maximum prediction confidence in histogram bins."""
        confidence = np.asarray(y_proba).max(axis=1)
        counts, edges = np.histogram(confidence, bins=bins, range=(0.0, 1.0))
        return {"mean": float(confidence.mean()), "median": float(np.median(confidence)),
                "counts": counts.tolist(), "bin_edges": edges.tolist()}

    def per_class_report(self, y_true, y_pred) -> Dict[str, Dict[str, float]]:
        """Return precision, recall, F1 and support for each configured class."""
        from sklearn.metrics import precision_recall_fscore_support
        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(self.class_names)), zero_division=0
        )
        return {name: {"precision": float(p[i]), "recall": float(r[i]),
                       "f1": float(f[i]), "support": int(s[i])}
                for i, name in enumerate(self.class_names)}

    def class_imbalance_report(self, y_true) -> Dict[str, Dict[str, float]]:
        """Return per-class counts and proportions."""
        y_true = np.asarray(y_true)
        total = max(len(y_true), 1)
        return {name: {"count": int((y_true == i).sum()),
                       "proportion": float((y_true == i).sum() / total)}
                for i, name in enumerate(self.class_names)}

    def failure_cases(self, texts, y_true, y_pred, y_proba, n: int = 10) -> Dict[str, List[Dict]]:
        """Return high-confidence wrong and low-confidence correct examples."""
        y_true, y_pred, y_proba = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_proba)
        confidence = y_proba.max(axis=1)
        def make(indices):
            return [{"index": int(i), "text": texts[i], "true_label": self.class_names[y_true[i]],
                     "predicted_label": self.class_names[y_pred[i]], "confidence": float(confidence[i])}
                    for i in indices]
        wrong = np.where(y_true != y_pred)[0]
        correct = np.where(y_true == y_pred)[0]
        return {"highest_confidence_wrong": make(wrong[np.argsort(confidence[wrong])[::-1][:n]]),
                "lowest_confidence_correct": make(correct[np.argsort(confidence[correct])[:n]])}

    def _per_class_errors(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict:
        """Compute false positive and false negative counts per class."""
        num_classes = len(self.class_names)
        per_class = {}

        for i in range(num_classes):
            class_name = self.class_names[i]
            tp = int(((y_true == i) & (y_pred == i)).sum())
            fp = int(((y_true != i) & (y_pred == i)).sum())
            fn = int(((y_true == i) & (y_pred != i)).sum())
            tn = int(((y_true != i) & (y_pred != i)).sum())
            total = int((y_true == i).sum())

            per_class[class_name] = {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "total_samples": total,
                "error_rate": fn / total if total > 0 else 0.0,
            }

        return per_class

    def _most_confused_pairs(
        self, y_true: np.ndarray, y_pred: np.ndarray, top_k: int = 10
    ) -> List[Dict]:
        """Find the most frequently confused class pairs."""
        cm = confusion_matrix(y_true, y_pred)
        pairs = []

        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm[i, j] > 0:
                    pairs.append({
                        "true_class": self.class_names[i],
                        "predicted_class": self.class_names[j],
                        "count": int(cm[i, j]),
                        "percentage": float(
                            cm[i, j] / cm[i].sum() * 100
                        ) if cm[i].sum() > 0 else 0.0,
                    })

        # Sort by count descending
        pairs.sort(key=lambda x: x["count"], reverse=True)
        return pairs[:top_k]

    def _get_error_examples(
        self,
        texts: List[str],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        error_mask: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        max_examples: int = 20,
    ) -> List[Dict]:
        """Get example texts that were misclassified."""
        error_indices = np.where(error_mask)[0]
        examples = []

        for idx in error_indices[:max_examples]:
            example = {
                "text": texts[idx] if idx < len(texts) else "",
                "true_label": self.class_names[y_true[idx]],
                "predicted_label": self.class_names[y_pred[idx]],
                "index": int(idx),
            }
            if y_proba is not None:
                example["confidence"] = float(y_proba[idx].max())
                example["true_class_prob"] = float(y_proba[idx, y_true[idx]])
            examples.append(example)

        # Sort by confidence (highest confidence wrong first)
        if y_proba is not None:
            examples.sort(key=lambda x: x["confidence"], reverse=True)

        return examples

    def _confidence_analysis(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
    ) -> Dict:
        """Analyze prediction confidence for correct vs incorrect predictions."""
        correct_mask = y_true == y_pred
        incorrect_mask = ~correct_mask

        max_proba = y_proba.max(axis=1)

        result = {
            "correct_avg_confidence": float(max_proba[correct_mask].mean())
            if correct_mask.any() else 0.0,
            "incorrect_avg_confidence": float(max_proba[incorrect_mask].mean())
            if incorrect_mask.any() else 0.0,
            "correct_median_confidence": float(np.median(max_proba[correct_mask]))
            if correct_mask.any() else 0.0,
            "incorrect_median_confidence": float(np.median(max_proba[incorrect_mask]))
            if incorrect_mask.any() else 0.0,
        }

        # Confidence distribution buckets
        buckets = [(0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        for low, high in buckets:
            mask = (max_proba >= low) & (max_proba < high)
            bucket_correct = int((correct_mask & mask).sum())
            bucket_total = int(mask.sum())
            result[f"bucket_{low:.1f}_{high:.1f}"] = {
                "total": bucket_total,
                "correct": bucket_correct,
                "accuracy": bucket_correct / bucket_total if bucket_total > 0 else 0.0,
            }

        return result

    def _length_analysis(
        self, texts: List[str], y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict:
        """Analyze error rate by text length."""
        lengths = np.array([len(t.split()) for t in texts])
        error_mask = y_true != y_pred

        # Quartile-based analysis
        quartiles = np.percentile(lengths, [25, 50, 75])
        bins = [0, quartiles[0], quartiles[1], quartiles[2], lengths.max() + 1]
        labels = ["short", "medium-short", "medium-long", "long"]

        length_results = {}
        for i in range(len(bins) - 1):
            mask = (lengths >= bins[i]) & (lengths < bins[i + 1])
            if mask.any():
                length_results[labels[i]] = {
                    "count": int(mask.sum()),
                    "errors": int((error_mask & mask).sum()),
                    "error_rate": float((error_mask & mask).sum() / mask.sum()),
                    "word_range": f"{int(bins[i])}-{int(bins[i+1]-1)}",
                }

        return length_results

    # -----------------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------------

    def plot_error_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Error Distribution by Class",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot per-class error rates as a bar chart."""
        per_class = self._per_class_errors(np.asarray(y_true), np.asarray(y_pred))

        classes = list(per_class.keys())
        error_rates = [per_class[c]["error_rate"] for c in classes]
        fp_counts = [per_class[c]["false_positives"] for c in classes]
        fn_counts = [per_class[c]["false_negatives"] for c in classes]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Error rate bar chart
        colors = sns.color_palette("viridis", len(classes))
        axes[0].bar(classes, error_rates, color=colors, edgecolor="white", linewidth=0.5)
        axes[0].set_title("Error Rate by Class", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Error Rate")
        axes[0].set_ylim(0, max(error_rates) * 1.3 if error_rates else 1)
        for i, v in enumerate(error_rates):
            axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
        axes[0].tick_params(axis="x", rotation=15)

        # FP vs FN stacked bar
        x = np.arange(len(classes))
        width = 0.35
        axes[1].bar(x - width / 2, fp_counts, width, label="False Positives",
                     color="#e74c3c", alpha=0.8)
        axes[1].bar(x + width / 2, fn_counts, width, label="False Negatives",
                     color="#3498db", alpha=0.8)
        axes[1].set_title("False Positives vs False Negatives", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Count")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(classes, rotation=15)
        axes[1].legend()

        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Error distribution plot saved to {save_path}")

        plt.show()

    def plot_confidence_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        title: str = "Prediction Confidence Distribution",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot confidence distributions for correct vs incorrect predictions."""
        correct_mask = np.asarray(y_true) == np.asarray(y_pred)
        max_proba = y_proba.max(axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(
            max_proba[correct_mask], bins=50, alpha=0.6,
            label=f"Correct (n={correct_mask.sum()})", color="#2ecc71", density=True
        )
        ax.hist(
            max_proba[~correct_mask], bins=50, alpha=0.6,
            label=f"Incorrect (n={(~correct_mask).sum()})", color="#e74c3c", density=True
        )

        ax.set_xlabel("Prediction Confidence", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 threshold")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Confidence distribution plot saved to {save_path}")

        plt.show()

    def plot_confused_pairs_heatmap(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Pairs (Errors Only)",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot heatmap showing only the error cells of the confusion matrix."""
        cm = confusion_matrix(np.asarray(y_true), np.asarray(y_pred))
        # Zero out the diagonal (correct predictions)
        np.fill_diagonal(cm, 0)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax, linewidths=0.5, linecolor="white"
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Confused pairs heatmap saved to {save_path}")

        plt.show()

    def generate_report(
        self,
        texts: List[str],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> str:
        """
        Generate a formatted text report of the error analysis.

        Args:
            texts: Original text inputs.
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            y_proba: Prediction probabilities.

        Returns:
            Formatted string report.
        """
        analysis = self.analyze(texts, y_true, y_pred, y_proba)

        lines = [
            "=" * 70,
            "ERROR ANALYSIS REPORT",
            "=" * 70,
            f"Total samples:  {analysis['total_samples']}",
            f"Total errors:   {analysis['total_errors']}",
            f"Error rate:     {analysis['error_rate']:.4f} ({analysis['error_rate']*100:.2f}%)",
            "",
            "-" * 70,
            "PER-CLASS ERROR RATES",
            "-" * 70,
        ]

        for cls_name, cls_data in analysis["per_class"].items():
            lines.append(
                f"  {cls_name:30s} | "
                f"Errors: {cls_data['false_negatives']:5d}/{cls_data['total_samples']:5d} | "
                f"Rate: {cls_data['error_rate']:.4f} | "
                f"FP: {cls_data['false_positives']:5d}"
            )

        lines.extend(["", "-" * 70, "MOST CONFUSED PAIRS", "-" * 70])

        for pair in analysis.get("confused_pairs", [])[:5]:
            lines.append(
                f"  {pair['true_class']:25s} → {pair['predicted_class']:25s} | "
                f"Count: {pair['count']:5d} ({pair['percentage']:.1f}%)"
            )

        if "confidence" in analysis:
            conf = analysis["confidence"]
            lines.extend([
                "", "-" * 70, "CONFIDENCE ANALYSIS", "-" * 70,
                f"  Correct predictions   — Avg: {conf['correct_avg_confidence']:.4f}, "
                f"Median: {conf['correct_median_confidence']:.4f}",
                f"  Incorrect predictions — Avg: {conf['incorrect_avg_confidence']:.4f}, "
                f"Median: {conf['incorrect_median_confidence']:.4f}",
            ])

        if "length_analysis" in analysis:
            lines.extend(["", "-" * 70, "ERROR RATE BY TEXT LENGTH", "-" * 70])
            for length_group, data in analysis["length_analysis"].items():
                lines.append(
                    f"  {length_group:15s} (words {data['word_range']:>10s}) | "
                    f"Errors: {data['errors']:5d}/{data['count']:5d} | "
                    f"Rate: {data['error_rate']:.4f}"
                )

        if analysis.get("error_examples"):
            lines.extend(["", "-" * 70, "SAMPLE MISCLASSIFIED EXAMPLES", "-" * 70])
            for i, ex in enumerate(analysis["error_examples"][:10], 1):
                text_preview = ex["text"][:80] + "..." if len(ex["text"]) > 80 else ex["text"]
                conf_str = f" (conf: {ex['confidence']:.3f})" if "confidence" in ex else ""
                lines.append(
                    f"  {i:2d}. [{ex['true_label']} → {ex['predicted_label']}]{conf_str}"
                )
                lines.append(f"      \"{text_preview}\"")

        lines.append("=" * 70)
        return "\n".join(lines)
