"""Train on training/validation partitions only. Evaluation is a separate CLI."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

FAMILIES = ["logistic_regression", "dnn", "rnn", "birnn", "gru", "bigru",
            "lstm", "bilstm", "attention", "transformer", "distilbert"]


def train(args):
    from configs.config import get_config
    from src.training_data import training_split
    from src.utils import set_seed, get_device, setup_logging
    from src.preprocessing import TextPreprocessor
    from sklearn.preprocessing import LabelEncoder

    from src.utils import configure_project_huggingface_cache
    configure_project_huggingface_cache()
    config = get_config(args.config)
    config.validate()
    config.paths.ensure_dirs()
    train_df, val_df, split_hash = training_split(config)
    # CSV loading precedes CUDA initialization on Windows.
    import torch
    set_seed(config.seed)
    device = get_device(args.device or config.device)
    logger = setup_logging(name="train", log_file=Path(config.paths.logs_dir) / "train.log")
    logger.info("Training-only protocol: %d train / %d validation. Test CSV is not opened.", len(train_df), len(val_df))
    modes = ["raw", "optimum", "extreme"] if args.preprocessing == "all" else [args.preprocessing or config.preprocessing.mode]
    families = [args.model] if args.model else list(FAMILIES)
    if args.features not in (None, "all"):
        families = [f for f in families if ("tfidf" if f in FAMILIES[:2] else "distilbert" if f == "distilbert" else "word2vec") == args.features]
    encoder = LabelEncoder().fit(config.data.class_names)
    y_train = encoder.transform(train_df[config.data.label_column])
    y_val = encoder.transform(val_df[config.data.label_column])
    summaries = {}
    for mode in modes:
        prep = TextPreprocessor(mode=mode)
        texts_train = prep.preprocess_batch(train_df[config.data.text_column].tolist())
        texts_val = prep.preprocess_batch(val_df[config.data.text_column].tolist())
        yt, yv = y_train, y_val
        if args.quick_test:
            texts_train, texts_val, yt, yv = texts_train[:1000], texts_val[:200], yt[:1000], yv[:200]
        for family in families:
            set_seed(config.seed)
            run = copy.deepcopy(config)
            run.preprocessing.mode = mode
            if args.quick_test:
                run.training.epochs = 2
            result, name = train_one(family, run, prep, encoder, texts_train, texts_val, yt, yv,
                                     device, split_hash)
            summaries[name] = result
            Path(config.paths.results_dir, "all_results.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
            logger.info("%s validation macro F1 %.6f", name, result["best_val_f1"])
            if device.type == "cuda":
                torch.cuda.empty_cache()


def train_one(family, config, prep, label_encoder, train_texts, val_texts, yt, yv, device, split_hash):
    import joblib
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.utils.class_weight import compute_class_weight
    from src.models import create_model
    from src.trainer import Trainer
    from src.utils import capture_environment
    from main import _SparseFeatureDataset

    feature = "tfidf" if family in FAMILIES[:2] else "distilbert" if family == "distilbert" else "word2vec_ids"
    name = f"{family}_{prep.mode}_{feature}"
    folder = Path(config.paths.saved_models_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    vectorizer = None
    kwargs = {}
    model_name = family
    training = replace(config.training, checkpoints_dir=config.paths.checkpoints_dir)
    if feature == "tfidf":
        word = TfidfVectorizer(max_features=config.tfidf.max_features, ngram_range=config.tfidf.ngram_range,
                              sublinear_tf=config.tfidf.sublinear_tf, min_df=config.tfidf.min_df,
                              max_df=config.tfidf.max_df, dtype=np.float32)
        vectorizer = (FeatureUnion([("word", word), ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), max_features=config.tfidf.char_max_features,
            sublinear_tf=True, min_df=2, dtype=np.float32))]) if config.tfidf.use_char else word)
        xt, xv = vectorizer.fit_transform(train_texts), vectorizer.transform(val_texts)
        if family == "logistic_regression":
            from src.models.classical import ClassicalModelWrapper
            wrapper = ClassicalModelWrapper(model_type=family, config=config)
            wrapper.fit(xt, yt)
            model = wrapper.model
            predict_started = time.perf_counter()
            pred = model.predict(xv)
            metrics = {"best_val_f1": float(f1_score(yv, pred, average="macro", zero_division=0)),
                       "best_val_acc": float(accuracy_score(yv, pred)), "best_epoch": 1,
                       "inference_time_ms": (time.perf_counter() - predict_started) * 1000 / len(yv)}
            joblib.dump(model, folder / "model.joblib")
        else:
            kwargs = dict(input_dim=xt.shape[1], num_classes=config.data.num_classes, **asdict(config.dnn))
            model = create_model("dnn", **kwargs)
            train_dataset, val_dataset = _SparseFeatureDataset(xt, yt), _SparseFeatureDataset(xv, yv)
    elif feature == "distilbert":
        from transformers import AutoTokenizer, DataCollatorWithPadding
        from src.datasets import DistilBERTDataset
        vectorizer = AutoTokenizer.from_pretrained(config.distilbert.model_name)
        train_dataset = DistilBERTDataset(train_texts, yt, vectorizer, config.distilbert.max_length)
        val_dataset = DistilBERTDataset(val_texts, yv, vectorizer, config.distilbert.max_length)
        kwargs = dict(model_name=config.distilbert.model_name, num_classes=config.data.num_classes,
                      dropout=config.distilbert.dropout, freeze_base=config.distilbert.freeze_base,
                      pooling=config.distilbert.pooling)
        model = create_model("distilbert", **kwargs)
        # Reconstruct from config + fine-tuned state, never download base weights at inference.
        kwargs["base_config"] = model.distilbert.config.to_dict()
        training = replace(training, learning_rate=config.distilbert.learning_rate,
                           head_lr_multiplier=config.distilbert.head_lr_multiplier)
    else:
        from src.sequence_features import WordVocabulary
        from src.embeddings import Word2VecFeatureExtractor
        vectorizer = WordVocabulary(config.sequence.vocab_size, config.sequence.max_len).fit(train_texts)
        pretrained = Word2VecFeatureExtractor(config.word2vec).load_pretrained()
        matrix = vectorizer.embedding_matrix(pretrained.model, config.word2vec.vector_size, config.seed)
        del pretrained
        xt, xv = vectorizer.transform(train_texts), vectorizer.transform(val_texts)
        train_dataset = TensorDataset(torch.from_numpy(xt), torch.as_tensor(yt))
        val_dataset = TensorDataset(torch.from_numpy(xv), torch.as_tensor(yv))
        if family == "transformer":
            base_name = "transformer"
            base = dict(asdict(config.transformer), num_classes=config.data.num_classes,
                        input_dim=config.word2vec.vector_size)
            base["max_seq_len"] = config.sequence.max_len
        elif family == "attention":
            base_name = "attention"
            base = dict(asdict(config.attention), num_classes=config.data.num_classes,
                        input_size=config.word2vec.vector_size)
        else:
            base_name = "rnn"
            base = dict(asdict(config.rnn), num_classes=config.data.num_classes,
                        input_size=config.word2vec.vector_size)
            base.update(cell_type=family.removeprefix("bi"), bidirectional=family.startswith("bi"))
        model_name = "embedded_sequence"
        kwargs = dict(encoder_name=base_name, encoder_kwargs=base, vocab_size=len(vectorizer.words),
                      embedding_dim=config.word2vec.vector_size, trainable=config.sequence.trainable_embeddings)
        model = create_model(model_name, **kwargs)
        with torch.no_grad():
            model.embedding.weight.copy_(torch.from_numpy(matrix))
        del matrix

    if family != "logistic_regression":
        collate = DataCollatorWithPadding(vectorizer) if feature == "distilbert" else None
        train_loader = DataLoader(train_dataset, batch_size=training.batch_size, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_dataset, batch_size=training.batch_size, collate_fn=collate)
        weights = torch.tensor(compute_class_weight("balanced", classes=np.arange(config.data.num_classes), y=yt), dtype=torch.float32) if training.use_class_weights else None
        trainer = Trainer(model, training, device, class_weights=weights, experiment_name=name)
        trainer.set_model_metadata(model_name, **kwargs)
        history = trainer.train(train_loader, val_loader)
        metrics = dict(trainer.best_metrics)
        model = trainer.model
        batch = next(iter(val_loader))
        inputs, _, forward_kwargs = trainer._unpack_batch(batch)
        with torch.inference_mode():
            model(inputs, **forward_kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            predict_started = time.perf_counter()
            for _ in range(5):
                model(inputs, **forward_kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize()
        metrics["inference_time_ms"] = (time.perf_counter() - predict_started) * 1000 / (5 * inputs.size(0))
        torch.save({"model_name": model_name, "model_kwargs": kwargs,
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, folder / "model.pt")
        (folder / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    metrics.update(training_seconds=time.perf_counter() - started, family=family, seed=config.seed,
                   split_sha256=split_hash, model_dir=str(folder.resolve()), preprocessing=prep.mode)
    config.training = training
    config.save(folder / "config.yaml")
    # Training/validation text must not be retained in deployed preprocessing caches.
    prep._cache.clear()
    joblib.dump(prep, folder / "preprocessor.joblib")
    joblib.dump(label_encoder, folder / "label_encoder.joblib")
    if feature == "distilbert":
        vectorizer.save_pretrained(folder / "tokenizer")
    else:
        joblib.dump(vectorizer, folder / "vectorizer.joblib")
    pipeline = dict(schema_version=2, feature_type=feature, model_name=model_name, model_kwargs=kwargs,
                    preprocessing_mode=prep.mode, class_names=list(label_encoder.classes_))
    (folder / "pipeline_config.json").write_text(json.dumps(pipeline, indent=2), encoding="utf-8")
    weight = folder / ("model.joblib" if family == "logistic_regression" else "model.pt")
    card = dict(model_name=name, architecture=family, feature_type=feature,
                preprocessing_mode=prep.mode, classes=list(label_encoder.classes_),
                training_summary=metrics, environment=capture_environment(),
                weights_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(), test_metrics=None,
                parameter_count=sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else None)
    (folder / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return metrics, name
