"""
News Topic Classification -- CLI Entry Point

Usage:
    python main.py train --config configs/default.yaml
    python main.py train --config configs/default.yaml --quick-test
    python main.py train --model bilstm --preprocessing optimum
    python main.py evaluate --model saved_models/bilstm_optimum_word2vec/
    python main.py predict --text "Apple reports record quarterly revenue"
    python main.py compare --results-dir results/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np


# Each selectable architecture has exactly one compatible feature modality.
# Keep this mapping next to the CLI declaration so validation and dispatch
# cannot drift apart.
_MODEL_FEATURES = {
    "logistic_regression": "tfidf",
    "dnn": "tfidf",
    "rnn": "word2vec",
    "birnn": "word2vec",
    "gru": "word2vec",
    "bigru": "word2vec",
    "lstm": "word2vec",
    "bilstm": "word2vec",
    "attention": "word2vec",
    "transformer": "word2vec",
    "distilbert": "distilbert",
}


def _resolve_feature_types(model: str | None, features: str | None) -> list[str]:
    """Return the feature branches needed by a valid train command.

    A specific architecture never needs unrelated feature extractors.  With
    no model selected, the default is the complete 11-model suite.
    """
    if model is not None:
        return [_MODEL_FEATURES[model]]
    if features == "all" or features is None:
        return ["tfidf", "word2vec", "distilbert"]
    return [features]


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="News Topic Classification -- Train, evaluate, and predict",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick test (2 epochs, small subset):
    python main.py train --config configs/default.yaml --quick-test

  Train all models with optimum preprocessing:
    python main.py train --preprocessing optimum --features all

  Train only BiLSTM:
    python main.py train --model bilstm --preprocessing optimum --features word2vec

  Train all models x all preprocessing x all features:
    python main.py train --preprocessing all --features all

  Evaluate a saved model:
    python main.py evaluate --model saved_models/bilstm_optimum_word2vec/

  Predict on new text:
    python main.py predict --model saved_models/bilstm_optimum_word2vec/ --text "Apple reports record revenue"

  Compare all trained models:
    python main.py compare --results-dir results/
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    train_parser.add_argument(
        "--model", type=str, default=None,
        choices=[
            "logistic_regression",
            "dnn",
            "rnn", "birnn",
            "gru", "bigru",
            "lstm", "bilstm",
            "attention",
            "transformer",
            "distilbert",
        ],
        help="Specific model to train. If omitted, trains all models.",
    )
    train_parser.add_argument(
        "--preprocessing", type=str, default=None,
        choices=["raw", "extreme", "optimum", "all"],
        help="Preprocessing mode. 'all' runs raw+extreme+optimum. Default: from config.",
    )
    train_parser.add_argument(
        "--features", type=str, default=None,
        choices=["tfidf", "word2vec", "distilbert", "all"],
        help="Feature type. Default: all features for the full suite; the required feature for one selected model.",
    )
    train_parser.add_argument(
        "--quick-test", action="store_true",
        help="Smoke test: 2 epochs, 1000 train / 200 val samples.",
    )
    train_parser.add_argument(
        "--device", type=str, default=None,
        help="Device override: 'cpu', 'cuda', 'cuda:0'",
    )
    train_parser.add_argument(
        "--skip-test-eval", action="store_true",
        help="Deprecated compatibility option; training never evaluates or opens the test CSV.",
    )

    # -----------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--model", type=str, required=True,
                             help="Path to saved model directory.")
    eval_parser.add_argument("--data", type=str, default=None,
                             help="Path to test CSV. Default: from config.")
    eval_parser.add_argument("--output", type=str, default="results/",
                             help="Directory to save evaluation results.")
    eval_parser.add_argument("--device", default=None, help="Use cpu for the serving artifact")

    # -----------------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------------
    predict_parser = subparsers.add_parser("predict", help="Predict on new text")
    predict_parser.add_argument("--model", type=str, required=True,
                                help="Path to saved model directory.")
    predict_parser.add_argument("--text", type=str, default=None,
                                help="Single text to classify.")
    predict_parser.add_argument("--file", type=str, default=None,
                                help="File with texts (one per line).")
    predict_parser.add_argument("--output", type=str, default=None,
                                help="Save predictions to JSON file.")

    # -----------------------------------------------------------------------
    # Compare
    # -----------------------------------------------------------------------
    compare_parser = subparsers.add_parser("compare", help="Compare trained models")
    compare_parser.add_argument("--results-dir", type=str, default="results/",
                                help="Directory containing model results.")
    compare_parser.add_argument(
        "--metric", type=str, default="f1_macro",
        choices=["accuracy", "f1_macro", "f1_weighted"],
        help="Metric to sort by.",
    )

    tune_parser = subparsers.add_parser("tune", help="Validation-only architecture and hyperparameter search")
    tune_parser.add_argument("--config", default="configs/improve.yaml")
    tune_parser.add_argument("--output", default="results/improvement")
    tune_parser.add_argument("--budget-hours", type=float, default=24.)
    tune_parser.add_argument("--trials-per-family", type=int, default=6)
    selected_parser = subparsers.add_parser("evaluate-selected", help="Explicit final test evaluation of frozen selections")
    selected_parser.add_argument("--selection", default="results/improvement/selection.json")
    selected_parser.add_argument("--data", default="data/Test_data.csv")
    selected_parser.add_argument("--output", default="results/final_test")
    export_parser = subparsers.add_parser("export", help="Package one selected inference model without training")
    export_parser.add_argument("--model")
    export_parser.add_argument("--selection", default="results/improvement/selection.json")
    export_parser.add_argument("--output", default="deploy/vercel-api/model")
    export_parser.add_argument("--quantize", action="store_true")
    benchmark_parser = subparsers.add_parser("benchmark", help="CPU resource and validation parity checks; never reads test data")
    benchmark_parser.add_argument("--model", required=True)
    benchmark_parser.add_argument("--reference", required=True)
    benchmark_parser.add_argument("--output", default="results/deployment_benchmark.json")
    prepare_parser = subparsers.add_parser("prepare-vercel", help="Stage Python API deployment files")
    prepare_parser.add_argument("--output", default="deploy/vercel-api")
    prepare_parser.add_argument("--selection", default="results/improvement/selection.json")

    args = parser.parse_args()
    if args.command == "train" and args.model and args.features not in (None, "all"):
        required_feature = _MODEL_FEATURES[args.model]
        if args.features != required_feature:
            parser.error(
                f"--model {args.model!r} requires --features {required_feature!r}; "
                f"got {args.features!r}. Use --features all or omit --features."
            )
    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SparseFeatureDataset:
    """Convert sparse TF-IDF rows lazily, avoiding a full dense copy."""

    def __init__(self, features, labels):
        import torch
        self.features = features
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        import torch
        row = self.features.getrow(index).toarray().ravel()
        return torch.from_numpy(row.astype(np.float32, copy=False)), self.labels[index]


def cmd_train(args: argparse.Namespace) -> None:
    from src.training_workflow import train
    train(args)


def _evaluate_checkpoint(model_path: str | Path, test_df, output_dir: str | Path, device=None) -> dict:
    """Evaluate one complete checkpoint and persist all report artifacts."""
    import pandas as pd
    import time
    from sklearn.preprocessing import LabelEncoder

    from src.inference import NewsClassifier
    from src.evaluation import Evaluator
    from src.error_analysis import ErrorAnalyzer
    from src.utils import setup_logging

    logger = setup_logging(name="evaluate")

    classifier = NewsClassifier.from_checkpoint(model_path, device=device)
    config = classifier.config
    required_columns = {config.data.text_column, config.data.label_column}
    missing_columns = required_columns - set(test_df.columns)
    if missing_columns:
        raise ValueError(
            f"Evaluation data is missing required column(s): {sorted(missing_columns)}. "
            f"Expected text='{config.data.text_column}' and label='{config.data.label_column}'."
        )
    texts  = test_df[config.data.text_column].tolist()
    labels = test_df[config.data.label_column].tolist()

    le = classifier.label_encoder or LabelEncoder().fit(config.data.class_names)
    y_true = le.transform(labels)

    start = time.perf_counter()
    results = classifier.batch_predict(texts)
    inference_time_ms = (time.perf_counter() - start) * 1000 / max(len(texts), 1)
    y_pred  = np.array([r["class_id"] for r in results])
    y_proba = np.array([
        [r["probabilities"].get(c, 0.0) for c in le.classes_]
        for r in results
    ]) if results and results[0]["probabilities"] else None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(class_names=config.data.class_names, results_dir=output_dir)
    metrics = evaluator.evaluate(
        y_true, y_pred, y_proba, model=classifier.model,
        inference_time_ms=inference_time_ms,
    )
    from src.utils import get_model_size
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(42)
    boot = [f1_score(y_true[idx], y_pred[idx], labels=np.arange(len(le.classes_)), average="macro", zero_division=0)
            for idx in (rng.integers(0, len(y_true), len(y_true)) for _ in range(1000))]
    metrics["macro_f1_ci_low"] = float(np.quantile(boot, .025))
    metrics["macro_f1_ci_high"] = float(np.quantile(boot, .975))
    metrics["test_rows"] = len(y_true)
    metrics["evaluation_device"] = str(classifier.device)
    metrics["model_size_mb"] = get_model_size(model_path)

    logger.info("\nEvaluation Results:")
    logger.info("  Accuracy        : %.4f", metrics["accuracy"])
    logger.info("  Macro F1        : %.4f", metrics["f1_macro"])
    logger.info("  Weighted F1     : %.4f", metrics["f1_weighted"])
    logger.info("  Matthews CC     : %.4f", metrics["mcc"])
    logger.info("  Inference / row : %.2f ms", metrics["inference_time_ms"])

    serializable_metrics = {
        key: value for key, value in metrics.items()
        if isinstance(value, (str, int, float, bool))
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(serializable_metrics, fh, indent=2)

    evaluator.plot_confusion_matrix(
        y_true, y_pred,
        title="Confusion Matrix",
        save_path=str(output_dir / "confusion_matrix.png"),
    )
    if y_proba is not None:
        evaluator.plot_roc_curves(y_true, y_proba, save_path=str(output_dir / "roc_curves.png"))
        evaluator.plot_precision_recall_curves(y_true, y_proba, save_path=str(output_dir / "precision_recall_curves.png"))

    analyzer = ErrorAnalyzer(class_names=config.data.class_names, results_dir=str(output_dir))
    report   = analyzer.generate_report(texts, y_true, y_pred, y_proba)
    logger.info("\n%s", report)
    (output_dir / "error_analysis.txt").write_text(report, encoding="utf-8")
    analysis = analyzer.analyze(texts, y_true, y_pred, y_proba)
    with open(output_dir / "error_analysis.json", "w", encoding="utf-8") as fh:
        json.dump(analysis, fh, indent=2, ensure_ascii=False)
    return serializable_metrics


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a trained model on a labelled test CSV."""
    import pandas as pd
    from configs.config import get_config

    # Read the checkpoint config first so the default data path remains
    # model-specific, then delegate to the shared evaluator used by training.
    config = get_config(Path(args.model) / "config.yaml")
    data_path = args.data if args.data else str(config.paths.test_path)
    test_df = pd.read_csv(data_path)
    metrics = _evaluate_checkpoint(args.model, test_df, args.output, device=args.device)
    from src.release import attach_test_metrics
    attach_test_metrics(Path(args.model), Path(data_path), metrics)


# ---------------------------------------------------------------------------
# Predict command
# ---------------------------------------------------------------------------

def cmd_predict(args: argparse.Namespace) -> None:
    """Predict on new text(s)."""
    from src.inference import NewsClassifier

    if not args.text and not args.file:
        print("Error: provide --text or --file")
        sys.exit(1)

    classifier = NewsClassifier.from_checkpoint(args.model)

    texts = []
    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    results = classifier.batch_predict(texts)

    CLASS_ICONS = {
        "Business": "?",
        "Science and Technology": "?",
        "Sports": "?",
        "World News": "?",
    }

    for r in results:
        icon = CLASS_ICONS.get(r["class"], "?")
        print(f"\n? {r['text'][:100]}")
        print(f"   {icon} {r['class']}  (confidence: {r['confidence']:.1%})")
        if r["probabilities"]:
            for cls, prob in sorted(r["probabilities"].items(), key=lambda x: -x[1]):
                bar = "?" * int(prob * 25)
                print(f"      {cls:32s} {prob:.3f}  {bar}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved -> {args.output}")


# ---------------------------------------------------------------------------
# Compare command
# ---------------------------------------------------------------------------

def cmd_compare(args: argparse.Namespace) -> None:
    """Compare all trained models from results JSON."""
    results_path = Path(args.results_dir) / "all_results.json"
    if not results_path.exists():
        print(f"No results found at {results_path}. Run training first.")
        sys.exit(1)

    with open(results_path) as f:
        all_results = json.load(f)

    metric = args.metric
    sorted_models = sorted(all_results.items(), key=lambda x: -x[1].get(metric, 0))

    print(f"\n{'=' * 65}")
    print(f"  Model Comparison  (sorted by {metric})")
    print(f"{'=' * 65}")
    print(f"  {'Experiment':<40}  {'Val Acc':>8}  {'Val F1':>8}")
    print(f"  {'-' * 61}")
    for name, res in sorted_models:
        acc = res.get("best_val_acc", res.get("accuracy", 0))
        f1  = res.get("best_val_f1",  res.get("f1_macro", 0))
        print(f"  {name:<40}  {acc:>8.4f}  {f1:>8.4f}")
    print(f"{'=' * 65}\n")

    try:
        from src.evaluation import Evaluator
        ev = Evaluator(results_dir=args.results_dir)
        ev.plot_comparison_bar(
            all_results,
            metrics=[metric],
            save_path=str(Path(args.results_dir) / "model_comparison.png"),
        )
        print(f"Comparison chart saved -> {args.results_dir}/model_comparison.png")
    except Exception as e:
        print(f"Warning: could not generate comparison plot: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.command is None:
        print("No command specified. Use --help for usage information.")
        sys.exit(1)
    from src.tuning import tune
    from src.release import evaluate_selected, export_model, prepare_vercel, benchmark
    dispatch = {
        "tune": tune,
        "evaluate-selected": evaluate_selected,
        "export": export_model,
        "prepare-vercel": prepare_vercel,
        "benchmark": benchmark,
        "train":    cmd_train,
        "evaluate": cmd_evaluate,
        "predict":  cmd_predict,
        "compare":  cmd_compare,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
