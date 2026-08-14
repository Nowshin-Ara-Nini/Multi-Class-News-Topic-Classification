from __future__ import annotations

import numpy as np
import pytest
import torch.nn as nn

from src.error_analysis import ErrorAnalyzer
from src.evaluation import Evaluator


def test_evaluator_returns_extended_metrics(tmp_path):
    evaluator = Evaluator(class_names=["Business", "Sports"], results_dir=tmp_path)
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    probabilities = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]])
    metrics = evaluator.evaluate(y_true, y_pred, probabilities, model=nn.Linear(3, 2), inference_time_ms=1.25)

    expected = {"precision_macro", "precision_weighted", "recall_macro", "recall_weighted", "roc_auc", "matthews_corrcoef", "cohen_kappa", "inference_time_ms", "parameter_count", "gpu_memory_mb"}
    assert expected.issubset(metrics)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["roc_auc"] == pytest.approx(1.0)

def test_error_analyzer_helpers_cover_requested_reports():
    analyzer = ErrorAnalyzer(class_names=["Business", "Sports"])
    texts = ["market", "football", "stocks", "tennis"]
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    proba = np.array([[.9, .1], [.8, .2], [.2, .8], [.1, .9]])

    assert analyzer.most_confused_pairs(y_true, y_pred, top_k=1)[0]["count"] == 1
    assert len(analyzer.misclassified_examples(texts, y_true, y_pred)) == 2
    assert sum(analyzer.confidence_distribution(proba)["counts"]) == 4
    assert analyzer.per_class_report(y_true, y_pred)["Business"]["support"] == 2
    assert analyzer.class_imbalance_report(y_true)["Sports"]["count"] == 2
    cases = analyzer.failure_cases(texts, y_true, y_pred, proba, n=1)
    assert len(cases["highest_confidence_wrong"]) == 1
    assert len(cases["lowest_confidence_correct"]) == 1
