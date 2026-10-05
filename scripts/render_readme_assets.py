"""Render documentation figures from saved reports; never loads datasets/models."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    # Existing EDA and final-artifact evaluation plots are committed under docs.
    shutil.copy2(ROOT / "notebooks" / "word_count_distribution.png", ASSETS / "headline_word_count.png")
    shutil.copy2(ROOT / "results" / "deployed_test" / "confusion_matrix.png", ASSETS / "deployed_confusion_matrix.png")

    manifest = json.loads((ROOT / "results" / "quick_improvement" / "split_manifest.json").read_text(encoding="utf-8"))
    counts = manifest["class_counts"]
    labels = list(counts)
    values = [counts[label] for label in labels]
    total = sum(values)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#344054", "text.color": "#172c30"})
    fig, ax = plt.subplots(figsize=(8.5, 4.3), constrained_layout=True)
    bars = ax.barh(labels[::-1], values[::-1], color=["#b7492c", "#527a8a", "#6b9b83", "#d69b4a"])
    ax.set_title("Label distribution after training-only cleaning", loc="left", weight="bold", pad=14)
    ax.set_xlabel("Headlines")
    ax.set_xlim(0, max(values) * 1.2)
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    for bar, label in zip(bars, labels[::-1]):
        value = counts[label]
        ax.text(value + max(values) * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{value:,}  ({value / total:.1%})", va="center", color="#344054", fontsize=9)
    fig.savefig(ASSETS / "training_class_distribution.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    comparison = json.loads((ROOT / "deploy" / "vercel-api" / "model" / "comparison.json").read_text(encoding="utf-8"))
    comparison.sort(key=lambda row: row["val_f1"])
    names = [row["family"].replace("_", " ").title() for row in comparison]
    validation = np.array([row["val_f1"] for row in comparison])
    test = np.array([row["test_f1"] for row in comparison])
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(9, 6.3), constrained_layout=True)
    ax.barh(y + 0.18, validation, height=0.34, color="#527a8a", label="Validation macro F1")
    ax.barh(y - 0.18, test, height=0.34, color="#d69b4a", label="Test macro F1 (post-selection)")
    ax.set_yticks(y, names)
    ax.set_xlim(0.88, 0.95)
    ax.set_xlabel("Macro F1 (axis starts at 0.88)")
    ax.set_title("Best checkpoint by model family", loc="left", weight="bold", pad=14)
    ax.axvline(0.95, color="#b7492c", linestyle="--", linewidth=1.2, label="95% target")
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(ASSETS / "family_macro_f1.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
