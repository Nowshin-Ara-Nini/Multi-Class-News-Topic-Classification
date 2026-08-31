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
        help="Skip held-out test evaluation after each trained model.",
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


def _make_loaders(
    X_feat, y, batch_size: int, shuffle: bool = True, padding_mask=None
):
    """Create a DataLoader without densifying an entire sparse matrix.

    ``padding_mask`` follows PyTorch convention: ``True`` means a padded
    sequence position.  It is only supplied for sequence models.
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    if hasattr(X_feat, "getrow"):
        if padding_mask is not None:
            raise ValueError("sparse feature loaders cannot use padding masks")
        ds = _SparseFeatureDataset(X_feat, y)
    else:
        X_dense = np.asarray(X_feat)
        tensors = [torch.as_tensor(X_dense, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)]
        if padding_mask is not None:
            tensors.append(torch.as_tensor(padding_mask, dtype=torch.bool))
        ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _class_weights(y_train, device):
    """Compute balanced class weights as a CUDA/CPU tensor."""
    import torch
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    return torch.FloatTensor(cw).to(device)


def _record(all_results: dict, exp_name: str, history: dict, logger) -> None:
    """Log and store training results."""
    best_f1  = max(history.get("val_f1",  [0.0]))
    best_acc = max(history.get("val_acc", [0.0]))
    all_results[exp_name] = {
        "best_val_f1":  round(best_f1,  4),
        "best_val_acc": round(best_acc, 4),
    }
    logger.info(
        "  [OK] %s  |  Best Val Acc: %.4f  |  Best Val F1: %.4f",
        exp_name, best_acc, best_f1,
    )


def _save(
    trainer, exp_name: str, saved_models_dir: str,
    model_name: str | None = None, model_kwargs: dict | None = None,
) -> Path:
    """Save a reconstructible PyTorch checkpoint and return its directory."""
    save_dir = Path(saved_models_dir) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    if model_name:
        trainer.set_model_metadata(model_name, **(model_kwargs or {}))
    trainer.save_checkpoint(str(save_dir / "model.pt"), is_best=True)
    return save_dir


def _save_pipeline_artifacts(
    save_dir: Path,
    config,
    preprocessor,
    label_encoder,
    feature_type: str,
    model_name: str,
    model_kwargs: dict | None = None,
    vectorizer=None,
    model=None,
    training_summary: dict | None = None,
) -> None:
    """Persist every non-model component required for prediction."""
    import joblib

    config.save(save_dir / "config.yaml")
    joblib.dump(preprocessor, save_dir / "preprocessor.joblib")
    joblib.dump(label_encoder, save_dir / "label_encoder.joblib")
    if feature_type == "distilbert" and vectorizer is not None:
        # Hugging Face components must be saved in their native format; a
        # joblib tokenizer is not reliably portable across transformers
        # versions.
        vectorizer.save_pretrained(save_dir / "tokenizer")
        if model is not None and hasattr(model, "distilbert"):
            model.distilbert.save_pretrained(save_dir / "base_model")
    elif vectorizer is not None:
        # Store the sklearn vectorizer, not the wrapper, so inference can use
        # its standard ``transform`` method.
        joblib.dump(getattr(vectorizer, "vectorizer", vectorizer), save_dir / "vectorizer.joblib")
    with open(save_dir / "pipeline_config.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "feature_type": feature_type,
                "model_name": model_name,
                "model_kwargs": model_kwargs or {},
                "preprocessing_mode": preprocessor.mode,
                "class_names": list(label_encoder.classes_),
            },
            fh,
            indent=2,
        )
    from src.utils import capture_environment, get_model_size
    parameter_count = (
        int(sum(parameter.numel() for parameter in model.parameters()))
        if model is not None and hasattr(model, "parameters") else None
    )
    card = {
        "model_name": save_dir.name,
        "architecture": model_name,
        "feature_type": feature_type,
        "preprocessing_mode": preprocessor.mode,
        "classes": list(label_encoder.classes_),
        "parameter_count": parameter_count,
        "model_size_mb": round(get_model_size(save_dir), 3),
        "training_summary": training_summary or {},
        "environment": capture_environment(),
    }
    with open(save_dir / "model_card.json", "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2, ensure_ascii=False)


def _evaluate_after_training(args, save_dir: Path, test_df, results_dir: str, exp_name: str, all_results: dict, logger) -> None:
    """Run and retain a held-out test evaluation for one newly saved model."""
    if args.skip_test_eval:
        return
    metrics = _evaluate_checkpoint(save_dir, test_df, Path(results_dir) / exp_name)
    # Keep validation and test results together without pretending they are
    # the same measurement.  Evaluation failures intentionally propagate: a
    # completed training run must not masquerade as a fully evaluated one.
    all_results[exp_name].update({f"test_{key}": value for key, value in metrics.items()})
    logger.info("  [TEST] %s | Acc: %.4f | Macro F1: %.4f", exp_name, metrics["accuracy"], metrics["f1_macro"])


# ---------------------------------------------------------------------------
# Train command
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    """Execute the full training pipeline."""
    import torch
    from configs.config import get_config
    from src.utils import (
        clear_invalid_local_proxy, configure_project_huggingface_cache,
        set_seed, get_device, setup_logging, timer,
    )
    from src.preprocessing import TextPreprocessor
    from src.datasets import DataManager
    from src.experiment import ExperimentTracker

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    config = get_config(args.config)
    config.validate()

    if args.quick_test:
        config.training.epochs = 2
        config.training.early_stopping = False

    logger = setup_logging(
        name="train",
        log_file=Path(config.paths.logs_dir) / "train.log",
    )
    config.paths.ensure_dirs()
    experiment_tracker = ExperimentTracker(config.paths.results_dir)

    logger.info("=" * 60)
    logger.info("Experiment : %s", config.experiment_name)
    logger.info("Device     : %s", args.device or config.device or "auto")
    logger.info("Quick test : %s", args.quick_test)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_manager = DataManager(config)
    logger.info("Loading training data from %s", config.paths.train_path)
    train_df, test_df = data_manager.load_data()
    logger.info("Loaded %d training rows and %d test rows", len(train_df), len(test_df))

    # On Windows, initialising CUDA before pandas has read the CSV can cause
    # a native pyarrow/pandas crash with no Python traceback.  Initialise the
    # accelerator only after all source data is safely in memory.
    set_seed(config.seed)
    device = get_device(args.device or config.device)
    logger.info("Training device initialised: %s", device)

    # Import PyTorch model code only after pandas has completed CSV loading.
    # This avoids native-library conflicts on Windows and keeps the
    # DistilBERT-only command free of unrelated feature dependencies.
    from src.models import create_model
    from src.trainer import Trainer

    # ------------------------------------------------------------------
    # Decide what to train
    # ------------------------------------------------------------------
    if args.preprocessing == "all":
        prep_modes = ["raw", "extreme", "optimum"]
    elif args.preprocessing:
        prep_modes = [args.preprocessing]
    else:
        prep_modes = [config.preprocessing.mode]

    feat_types = _resolve_feature_types(args.model, args.features)

    all_results: dict = {}
    # ------------------------------------------------------------------
    # Main loop: preprocessing x feature type
    # ------------------------------------------------------------------
    for prep_mode in prep_modes:
        logger.info("\n%s\nPreprocessing: %s\n%s", "=" * 60, prep_mode, "=" * 60)

        logger.info("Preparing %s text split", prep_mode)
        preprocessor = TextPreprocessor(mode=prep_mode)
        # Persist the actual preprocessing mode with each model.  The base
        # config may say "optimum" while this run is intentionally raw.
        from copy import deepcopy
        run_config = deepcopy(config)
        run_config.preprocessing.mode = prep_mode
        splits = data_manager.prepare_splits(train_df, preprocessor)

        X_train_texts: list = splits["train_texts"]
        X_val_texts:   list = splits["val_texts"]
        y_train:  np.ndarray = splits["train_labels"]
        y_val:    np.ndarray = splits["val_labels"]

        # Quick-test subsample
        if args.quick_test:
            X_train_texts = X_train_texts[:1000]
            y_train       = y_train[:1000]
            X_val_texts   = X_val_texts[:200]
            y_val         = y_val[:200]

        logger.info(
            "Split sizes -- train: %d  val: %d",
            len(X_train_texts), len(X_val_texts),
        )

        # ==============================================================
        # TF-IDF branch  ->  Logistic Regression + DNN
        # ==============================================================
        if "tfidf" in feat_types:
            from src.embeddings import TfidfFeatureExtractor

            logger.info("\n--- Feature: TF-IDF | Preprocessing: %s ---", prep_mode)

            tfidf = TfidfFeatureExtractor(config.tfidf)
            X_tr_tfidf = tfidf.fit_transform(X_train_texts)
            X_vl_tfidf = tfidf.transform(X_val_texts)
            input_dim  = X_tr_tfidf.shape[1]

            # --- Logistic Regression ---
            if args.model in (None, "logistic_regression"):
                from src.models.classical import ClassicalModelWrapper

                exp_name = f"lr_{prep_mode}_tfidf"
                logger.info("Training Logistic Regression -> %s", exp_name)
                lr = ClassicalModelWrapper(model_type="logistic_regression", config=config)
                with timer(f"LR ({exp_name})", logger):
                    lr.fit(X_tr_tfidf, y_train)
                res = lr.evaluate(X_vl_tfidf, y_val)
                all_results[exp_name] = {
                    "best_val_f1":  round(res["f1_macro"], 4),
                    "best_val_acc": round(res["accuracy"], 4),
                }
                logger.info(
                    "  [OK] %s  |  Val Acc: %.4f  |  Val F1: %.4f",
                    exp_name, res["accuracy"], res["f1_macro"],
                )
                save_lr_dir = Path(config.paths.saved_models_dir) / exp_name
                save_lr_dir.mkdir(parents=True, exist_ok=True)
                lr.save(save_lr_dir / "model.joblib")
                _save_pipeline_artifacts(
                    save_lr_dir, run_config, preprocessor, data_manager.label_encoder,
                    "tfidf", "logistic_regression", vectorizer=tfidf, model=lr.model,
                    training_summary=all_results[exp_name],
                )
                _evaluate_after_training(args, save_lr_dir, test_df, config.paths.results_dir, exp_name, all_results, logger)

            # --- DNN ---
            if args.model in (None, "dnn"):
                exp_name = f"dnn_{prep_mode}_tfidf"
                logger.info("Training DNN -> %s", exp_name)

                train_loader = _make_loaders(X_tr_tfidf, y_train, config.training.batch_size)
                val_loader   = _make_loaders(X_vl_tfidf, y_val,   config.training.batch_size, shuffle=False)

                dnn = create_model(
                    "dnn",
                    input_dim=input_dim,
                    num_classes=config.data.num_classes,
                    hidden_layers=config.dnn.hidden_layers,
                    dropout=config.dnn.dropout,
                )
                cw = _class_weights(y_train, device) if config.training.use_class_weights else None
                trainer = Trainer(
                    model=dnn, config=config.training,
                    device=device, class_weights=cw, experiment_name=exp_name,
                    experiment_tracker=experiment_tracker,
                )
                with timer(f"DNN ({exp_name})", logger):
                    history = trainer.train(train_loader, val_loader)
                _record(all_results, exp_name, history, logger)
                save_dir = _save(
                    trainer, exp_name, config.paths.saved_models_dir, "dnn",
                    {"input_dim": input_dim, "num_classes": config.data.num_classes,
                     "hidden_layers": config.dnn.hidden_layers, "dropout": config.dnn.dropout},
                )
                _save_pipeline_artifacts(
                    save_dir, run_config, preprocessor, data_manager.label_encoder,
                    "tfidf", "dnn", vectorizer=tfidf, model=trainer.model,
                    training_summary=all_results[exp_name],
                )
                _evaluate_after_training(args, save_dir, test_df, config.paths.results_dir, exp_name, all_results, logger)

        # ==============================================================
        # DistilBERT branch -> optional pretrained transformer baseline
        # ==============================================================
        if "distilbert" in feat_types and args.model in (None, "distilbert"):
            # Do this before importing transformers: Hugging Face may create
            # its HTTP client at import time and otherwise retain the bad proxy.
            removed_proxies = clear_invalid_local_proxy()
            hf_cache = configure_project_huggingface_cache()
            if removed_proxies:
                logger.warning(
                    "Ignoring invalid local proxy setting(s): %s",
                    ", ".join(removed_proxies),
                )
            logger.info("Hugging Face cache: %s", hf_cache)
            try:
                from dataclasses import replace
                from torch.utils.data import DataLoader
                from transformers import AutoTokenizer
                from src.datasets import DistilBERTDataset
            except ImportError as exc:
                raise RuntimeError(
                    "DistilBERT was selected but its optional dependencies are unavailable. "
                    "Install the project's requirements before training."
                ) from exc
            else:
                exp_name = f"distilbert_{prep_mode}"
                logger.info(
                    "Loading DistilBERT tokenizer '%s'. The first run may download model files.",
                    config.distilbert.model_name,
                )
                try:
                    tokenizer = AutoTokenizer.from_pretrained(config.distilbert.model_name)
                except Exception as exc:
                    raise RuntimeError(
                        "Could not load the DistilBERT tokenizer. Connect to "
                        "https://huggingface.co once to download 'distilbert-base-uncased', "
                        "or set distilbert.model_name in your YAML config to a complete "
                        "local Hugging Face model directory."
                    ) from exc
                train_dataset = DistilBERTDataset(
                    X_train_texts, y_train, tokenizer, config.distilbert.max_length
                )
                val_dataset = DistilBERTDataset(
                    X_val_texts, y_val, tokenizer, config.distilbert.max_length
                )
                train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
                model_kwargs = {
                    "model_name": config.distilbert.model_name,
                    "num_classes": config.data.num_classes,
                    "dropout": config.distilbert.dropout,
                    "freeze_base": config.distilbert.freeze_base,
                }
                logger.info("Loading DistilBERT encoder (freeze_base=%s)", config.distilbert.freeze_base)
                try:
                    model = create_model("distilbert", **model_kwargs)
                except Exception as exc:
                    raise RuntimeError(
                        "Could not load the DistilBERT encoder. Ensure the tokenizer and model "
                        "weights are available from Hugging Face or in the configured local model directory."
                    ) from exc
                bert_training = replace(config.training, learning_rate=config.distilbert.learning_rate)
                cw = _class_weights(y_train, device) if config.training.use_class_weights else None
                trainer = Trainer(
                    model=model, config=bert_training, device=device,
                    class_weights=cw, experiment_name=exp_name,
                    experiment_tracker=experiment_tracker,
                )
                with timer(f"DistilBERT ({exp_name})", logger):
                    history = trainer.train(train_loader, val_loader)
                _record(all_results, exp_name, history, logger)
                save_dir = _save(trainer, exp_name, config.paths.saved_models_dir, "distilbert", model_kwargs)
                _save_pipeline_artifacts(
                    save_dir, run_config, preprocessor, data_manager.label_encoder,
                    "distilbert", "distilbert", model_kwargs, vectorizer=tokenizer,
                    model=trainer.model, training_summary=all_results[exp_name],
                )
                _evaluate_after_training(args, save_dir, test_df, config.paths.results_dir, exp_name, all_results, logger)

        # ==============================================================
        # Word2Vec branch  ->  RNN/GRU/LSTM x uni/bi, Attention, Transformer
        # ==============================================================
        if "word2vec" in feat_types:
            from src.embeddings import Word2VecFeatureExtractor

            logger.info("\n--- Feature: Word2Vec | Preprocessing: %s ---", prep_mode)

            w2v = Word2VecFeatureExtractor(config.word2vec)
            try:
                w2v.load_pretrained()
            except Exception as exc:
                raise RuntimeError(
                    "Word2Vec was selected but the configured pretrained embeddings could not be loaded. "
                    "Download/cache the configured model, or select a TF-IDF or DistilBERT model."
                ) from exc

            with timer("Building sequence arrays", logger):
                X_tr_seq = w2v.texts_to_sequences(X_train_texts, max_len=config.sequence.max_len)
                X_vl_seq = w2v.texts_to_sequences(X_val_texts,   max_len=config.sequence.max_len)

            # Word2Vec uses zero vectors for post-padding.  Keep that mask
            # with each batch so recurrent, attention, and Transformer models
            # do not treat padding as news content.
            train_padding_mask = np.all(np.isclose(X_tr_seq, 0.0), axis=-1)
            val_padding_mask = np.all(np.isclose(X_vl_seq, 0.0), axis=-1)
            train_loader = _make_loaders(
                X_tr_seq, y_train, config.training.batch_size,
                padding_mask=train_padding_mask,
            )
            val_loader = _make_loaders(
                X_vl_seq, y_val, config.training.batch_size, shuffle=False,
                padding_mask=val_padding_mask,
            )

            cw = _class_weights(y_train, device) if config.training.use_class_weights else None

            # ---- RNN / GRU / LSTM x unidirectional / bidirectional ----
            ALL_RNN_VARIANTS = [
                ("rnn",  False, "rnn"),
                ("rnn",  True,  "birnn"),
                ("gru",  False, "gru"),
                ("gru",  True,  "bigru"),
                ("lstm", False, "lstm"),
                ("lstm", True,  "bilstm"),
            ]

            for cell_type, bidirectional, model_key in ALL_RNN_VARIANTS:
                # Skip variants the user didn't ask for
                if args.model is not None and args.model != model_key:
                    continue

                exp_name = f"{model_key}_{prep_mode}_word2vec"
                logger.info("Training %s -> %s", model_key.upper(), exp_name)

                rnn_model = create_model(
                    "rnn",
                    input_size=config.word2vec.vector_size,
                    hidden_size=config.rnn.hidden_size,
                    num_layers=config.rnn.num_layers,
                    num_classes=config.data.num_classes,
                    cell_type=cell_type,
                    bidirectional=bidirectional,
                    dropout=config.rnn.dropout,
                    pooling=config.rnn.pooling,
                    use_layer_norm=config.rnn.use_layer_norm,
                )
                trainer = Trainer(
                    model=rnn_model, config=config.training,
                    device=device, class_weights=cw, experiment_name=exp_name,
                    experiment_tracker=experiment_tracker,
                )
                with timer(f"{model_key.upper()} ({exp_name})", logger):
                    history = trainer.train(train_loader, val_loader)
                _record(all_results, exp_name, history, logger)
                save_dir = _save(
                    trainer, exp_name, config.paths.saved_models_dir, "rnn",
                    {"input_size": config.word2vec.vector_size,
                     "hidden_size": config.rnn.hidden_size,
                     "num_layers": config.rnn.num_layers,
                     "num_classes": config.data.num_classes,
                     "cell_type": cell_type, "bidirectional": bidirectional,
                     "dropout": config.rnn.dropout, "pooling": config.rnn.pooling,
                     "use_layer_norm": config.rnn.use_layer_norm},
                )
                _save_pipeline_artifacts(
                    save_dir, run_config, preprocessor, data_manager.label_encoder,
                    "word2vec", "rnn", model=trainer.model,
                    training_summary=all_results[exp_name],
                )
                _evaluate_after_training(args, save_dir, test_df, config.paths.results_dir, exp_name, all_results, logger)

            # ---- Attention ----
            if args.model in (None, "attention"):
                exp_name = f"attention_{prep_mode}_word2vec"
                logger.info("Training Attention -> %s", exp_name)

                attn_model = create_model(
                    "attention",
                    input_size=config.word2vec.vector_size,
                    hidden_size=config.attention.hidden_size,
                    num_layers=config.attention.num_layers,
                    num_classes=config.data.num_classes,
                    cell_type=config.attention.cell_type,
                    bidirectional=config.attention.bidirectional,
                    dropout=config.attention.dropout,
                    attention_dim=config.attention.attention_dim,
                )
                trainer = Trainer(
                    model=attn_model, config=config.training,
                    device=device, class_weights=cw, experiment_name=exp_name,
                    experiment_tracker=experiment_tracker,
                )
                with timer(f"Attention ({exp_name})", logger):
                    history = trainer.train(train_loader, val_loader)
                _record(all_results, exp_name, history, logger)
                save_dir = _save(
                    trainer, exp_name, config.paths.saved_models_dir, "attention",
                    {"input_size": config.word2vec.vector_size,
                     "hidden_size": config.attention.hidden_size,
                     "num_layers": config.attention.num_layers,
                     "num_classes": config.data.num_classes,
                     "cell_type": config.attention.cell_type,
                     "bidirectional": config.attention.bidirectional,
                     "dropout": config.attention.dropout,
                     "attention_dim": config.attention.attention_dim},
                )
                _save_pipeline_artifacts(
                    save_dir, run_config, preprocessor, data_manager.label_encoder,
                    "word2vec", "attention", model=trainer.model,
                    training_summary=all_results[exp_name],
                )
                _evaluate_after_training(args, save_dir, test_df, config.paths.results_dir, exp_name, all_results, logger)

            # ---- Transformer ----
            if args.model in (None, "transformer"):
                exp_name = f"transformer_{prep_mode}_word2vec"
                logger.info("Training Transformer -> %s", exp_name)

                transformer_model = create_model(
                    "transformer",
                    d_model=config.transformer.d_model,
                    nhead=config.transformer.nhead,
                    num_encoder_layers=config.transformer.num_encoder_layers,
                    dim_feedforward=config.transformer.dim_feedforward,
                    num_classes=config.data.num_classes,
                    max_seq_len=config.transformer.max_seq_len,
                    dropout=config.transformer.dropout,
                    pooling=config.transformer.pooling,
                )
                trainer = Trainer(
                    model=transformer_model, config=config.training,
                    device=device, class_weights=cw, experiment_name=exp_name,
                    experiment_tracker=experiment_tracker,
                )
                with timer(f"Transformer ({exp_name})", logger):
                    history = trainer.train(train_loader, val_loader)
                _record(all_results, exp_name, history, logger)
                save_dir = _save(
                    trainer, exp_name, config.paths.saved_models_dir, "transformer",
                    {"d_model": config.transformer.d_model,
                     "nhead": config.transformer.nhead,
                     "num_encoder_layers": config.transformer.num_encoder_layers,
                     "dim_feedforward": config.transformer.dim_feedforward,
                     "num_classes": config.data.num_classes,
                     "max_seq_len": config.transformer.max_seq_len,
                     "dropout": config.transformer.dropout,
                     "pooling": config.transformer.pooling},
                )
                _save_pipeline_artifacts(
                    save_dir, run_config, preprocessor, data_manager.label_encoder,
                    "word2vec", "transformer", model=trainer.model,
                    training_summary=all_results[exp_name],
                )
                _evaluate_after_training(args, save_dir, test_df, config.paths.results_dir, exp_name, all_results, logger)

    # ------------------------------------------------------------------
    # Final summary table
    # ------------------------------------------------------------------
    if all_results:
        from src.evaluation import Evaluator

        evaluator = Evaluator(
            class_names=config.data.class_names,
            results_dir=config.paths.results_dir,
        )
        logger.info("\n%s\nFINAL RESULTS\n%s", "=" * 60, "=" * 60)
        logger.info("%-45s  %8s  %8s", "Experiment", "Val Acc", "Val F1")
        logger.info("-" * 65)
        for name, res in sorted(all_results.items(), key=lambda x: -x[1]["best_val_f1"]):
            logger.info(
                "%-45s  %8.4f  %8.4f",
                name, res["best_val_acc"], res["best_val_f1"],
            )

        # Save results JSON
        results_path = Path(config.paths.results_dir) / "all_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("\nResults saved -> %s", results_path)

        test_results = {
            name: {key.removeprefix("test_"): value for key, value in result.items() if key.startswith("test_")}
            for name, result in all_results.items()
            if "test_accuracy" in result
        }
        if test_results:
            test_path = Path(config.paths.results_dir) / "test_results.json"
            with open(test_path, "w", encoding="utf-8") as fh:
                json.dump(test_results, fh, indent=2)
            evaluator.compare_models(test_results).to_csv(
                Path(config.paths.results_dir) / "test_results.csv"
            )
            (Path(config.paths.results_dir) / "test_results.md").write_text(
                evaluator.generate_markdown_table(test_results), encoding="utf-8"
            )
            (Path(config.paths.results_dir) / "test_results.tex").write_text(
                evaluator.generate_latex_table(test_results), encoding="utf-8"
            )
            logger.info("Held-out test summary saved -> %s", test_path)

        # Comparison plot
        try:
            evaluator.plot_comparison_bar(
                test_results or all_results,
                metrics=["f1_macro", "accuracy"] if test_results else ["best_val_f1", "best_val_acc"],
                save_path=str(Path(config.paths.results_dir) / "model_comparison.png"),
            )
        except Exception as e:
            logger.warning("Could not save comparison plot: %s", e)


# ---------------------------------------------------------------------------
# Evaluate command
# ---------------------------------------------------------------------------

def _evaluate_checkpoint(model_path: str | Path, test_df, output_dir: str | Path) -> dict:
    """Evaluate one complete checkpoint and persist all report artifacts."""
    import pandas as pd
    import time
    from sklearn.preprocessing import LabelEncoder

    from src.inference import NewsClassifier
    from src.evaluation import Evaluator
    from src.error_analysis import ErrorAnalyzer
    from src.utils import setup_logging

    logger = setup_logging(name="evaluate")

    classifier = NewsClassifier.from_checkpoint(model_path)
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
    from src.inference import NewsClassifier

    # Read the checkpoint config first so the default data path remains
    # model-specific, then delegate to the shared evaluator used by training.
    classifier = NewsClassifier.from_checkpoint(args.model)
    data_path = args.data if args.data else str(classifier.config.paths.test_path)
    test_df = pd.read_csv(data_path)
    _evaluate_checkpoint(args.model, test_df, args.output)


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
    dispatch = {
        "train":    cmd_train,
        "evaluate": cmd_evaluate,
        "predict":  cmd_predict,
        "compare":  cmd_compare,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
