"""
Production inference pipeline for News Topic Classification.

Provides a unified interface for classifying new text inputs using
any trained model (classical or deep learning). Handles model loading,
preprocessing, feature extraction, and prediction in a single call.
"""

from __future__ import annotations

import logging
import os
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from configs.config import Config, get_config
from src.utils import clear_invalid_local_proxy, configure_project_huggingface_cache, get_device, setup_logging

logger = logging.getLogger(__name__)


class NewsClassifier:
    """
    Production-ready news topic classifier.

    Loads a trained model with its associated preprocessor, vectorizer,
    and label encoder, then provides simple predict/batch_predict methods.

    Usage:
        classifier = NewsClassifier.from_checkpoint('saved_models/best_bilstm/')
        result = classifier.predict("Apple reports record quarterly revenue")
        # {'class': 'Business', 'confidence': 0.94, 'probabilities': {...}}

        results = classifier.batch_predict([
            "Scientists discover new species",
            "Lakers win championship"
        ])
    """

    def __init__(
        self,
        model: Any,
        preprocessor: Any = None,
        vectorizer: Any = None,
        label_encoder: Any = None,
        vocab: Any = None,
        config: Optional[Config] = None,
        device: Optional[torch.device] = None,
        model_type: str = "deep_learning",
        feature_type: Optional[str] = None,
    ):
        """
        Initialize the classifier with loaded components.

        Args:
            model: Trained model (PyTorch nn.Module or sklearn model).
            preprocessor: TextPreprocessor instance.
            vectorizer: Feature extractor (TF-IDF or Word2Vec).
            label_encoder: sklearn LabelEncoder for decoding predictions.
            vocab: Vocabulary instance (for sequence models).
            config: Configuration object.
            device: Torch device for inference.
            model_type: 'deep_learning' or 'classical'.
        """
        self.model = model
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
        self.vocab = vocab
        self.config = config or Config.default()
        self.device = device or get_device()
        self.model_type = model_type
        self.feature_type = feature_type

        # Move DL model to device and set eval mode
        if model_type == "deep_learning" and isinstance(model, torch.nn.Module):
            self.model = model.to(self.device)
            self.model.eval()

        logger.info(f"NewsClassifier initialized (type={model_type}, device={self.device})")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        config_path: Optional[str | Path] = None,
        device: Optional[str] = None,
    ) -> "NewsClassifier":
        """
        Load a complete classifier from a checkpoint directory.

        Expected directory structure:
            checkpoint_dir/
            ├── model.pt          (PyTorch model state)  OR  model.joblib (sklearn)
            ├── preprocessor.joblib
            ├── vectorizer.joblib
            ├── label_encoder.joblib
            ├── vocab.joblib      (optional, for sequence models)
            └── config.yaml       (optional)

        Args:
            checkpoint_dir: Path to directory containing saved artifacts.
            config_path: Optional path to config YAML file.
            device: Device override (e.g., 'cpu', 'cuda').

        Returns:
            Initialized NewsClassifier ready for predictions.

        Raises:
            FileNotFoundError: If checkpoint directory or required files don't exist.
        """
        checkpoint_dir = Path(checkpoint_dir)

        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {checkpoint_dir}"
            )

        # Load config
        config = None
        config_file = checkpoint_dir / "config.yaml"
        if config_path:
            config = get_config(config_path)
        elif config_file.exists():
            config = get_config(config_file)
        else:
            config = Config.default()

        # Determine device
        torch_device = get_device(device)

        pipeline_path = checkpoint_dir / "pipeline_config.json"
        pipeline = {}
        if pipeline_path.exists():
            with open(pipeline_path, encoding="utf-8") as fh:
                pipeline = json.load(fh)
        if not pipeline:
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint_dir}' is incomplete: missing "
                "pipeline_config.json. It cannot be evaluated or used for "
                "prediction because its architecture, feature type, and "
                "preprocessing metadata are unknown. Retrain this model with "
                "the current training command."
            )

        feature_type = pipeline.get("feature_type")
        if feature_type not in {"tfidf", "word2vec", "distilbert"}:
            raise ValueError(
                f"Checkpoint '{checkpoint_dir}' has an unsupported or missing "
                f"feature_type: {feature_type!r}. Retrain it with the current pipeline."
            )

        # Load components
        preprocessor = cls._load_component(
            checkpoint_dir / "preprocessor.joblib", "preprocessor"
        )
        label_encoder = cls._load_component(
            checkpoint_dir / "label_encoder.joblib", "label_encoder"
        )
        vectorizer = cls._load_component(
            checkpoint_dir / "vectorizer.joblib", "vectorizer"
        )
        vocab = cls._load_component(
            checkpoint_dir / "vocab.joblib", "vocab"
        )

        missing_components = []
        if preprocessor is None:
            missing_components.append("preprocessor.joblib")
        if label_encoder is None:
            missing_components.append("label_encoder.joblib")
        if feature_type == "tfidf" and vectorizer is None:
            missing_components.append("vectorizer.joblib")
        if missing_components:
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint_dir}' is incomplete: missing "
                f"{', '.join(missing_components)}. Retrain this model with "
                "the current training command."
            )

        # Load model (PyTorch or sklearn)
        if feature_type == "word2vec" and vectorizer is None:
            from src.embeddings import Word2VecFeatureExtractor
            vectorizer = Word2VecFeatureExtractor(config.word2vec).load_pretrained()
        elif feature_type == "distilbert" and vectorizer is None:
            clear_invalid_local_proxy()
            configure_project_huggingface_cache()
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError("DistilBERT inference requires `transformers`.") from exc
            tokenizer_dir = checkpoint_dir / "tokenizer"
            tokenizer_name = str(tokenizer_dir) if tokenizer_dir.exists() else pipeline.get("model_kwargs", {}).get("model_name", config.distilbert.model_name)
            vectorizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # New checkpoints include a local DistilBERT base-model snapshot.  It
        # makes prediction portable to machines without a Hugging Face cache.
        if feature_type == "distilbert" and (checkpoint_dir / "base_model").exists():
            pipeline = dict(pipeline)
            model_kwargs = dict(pipeline.get("model_kwargs", {}))
            model_kwargs["model_name"] = str(checkpoint_dir / "base_model")
            pipeline["model_kwargs"] = model_kwargs

        model, model_type = cls._load_model(checkpoint_dir, torch_device, config, pipeline)

        return cls(
            model=model,
            preprocessor=preprocessor,
            vectorizer=vectorizer,
            label_encoder=label_encoder,
            vocab=vocab,
            config=config,
            device=torch_device,
            model_type=model_type,
            feature_type=feature_type,
        )

    @staticmethod
    def _load_component(path: Path, name: str) -> Any:
        """Load a joblib-serialized component, returning None if not found."""
        if path.exists():
            component = joblib.load(path)
            logger.debug(f"Loaded {name} from {path}")
            return component
        logger.debug(f"{name} not found at {path}, skipping")
        return None

    @staticmethod
    def _load_model(
        checkpoint_dir: Path,
        device: torch.device,
        config: Config,
        pipeline: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Load model from checkpoint directory."""
        # Try PyTorch model first
        pt_path = checkpoint_dir / "model.pt"
        if pt_path.exists():
            checkpoint = torch.load(pt_path, map_location=device, weights_only=False)

            # If checkpoint contains model class info, reconstruct
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model_name = checkpoint.get("model_name") or (pipeline or {}).get("model_name")
                model_kwargs = checkpoint.get("model_kwargs") or (pipeline or {}).get("model_kwargs", {})

                if not model_name:
                    raise ValueError(
                        "Checkpoint lacks model reconstruction metadata. "
                        "Retrain once with the current pipeline to create portable artifacts."
                    )

                try:
                    from src.models import create_model
                    model = create_model(model_name, **model_kwargs)
                    model.load_state_dict(checkpoint["model_state_dict"])
                except Exception as e:
                    logger.warning(
                        f"Could not reconstruct model '{model_name}': {e}. "
                        f"Loading state dict directly."
                    )
                    # Fall back: assume model object was saved directly
                    model = checkpoint
            else:
                # Whole model was saved with torch.save
                model = checkpoint

            if isinstance(model, torch.nn.Module):
                model = model.to(device)
                model.eval()

            logger.info(f"Loaded PyTorch model from {pt_path}")
            return model, "deep_learning"

        # Try sklearn model
        sklearn_path = checkpoint_dir / "model.joblib"
        if sklearn_path.exists():
            model = joblib.load(sklearn_path)
            # ``ClassicalModelWrapper.save`` stores a compact state dict.
            if isinstance(model, dict) and "model" in model:
                model = model["model"]
            logger.info(f"Loaded sklearn model from {sklearn_path}")
            return model, "classical"

        raise FileNotFoundError(
            f"No model file found in {checkpoint_dir}. "
            f"Expected 'model.pt' (PyTorch) or 'model.joblib' (sklearn)."
        )

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classify a single text string.

        Args:
            text: Raw news headline text.

        Returns:
            Dictionary with:
                - 'class': Predicted class name (str)
                - 'class_id': Predicted class index (int)
                - 'confidence': Prediction confidence (float)
                - 'probabilities': Dict of class name → probability
                - 'inference_time_ms': Inference time in milliseconds
        """
        results = self.batch_predict([text])
        return results[0]

    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Classify a batch of text strings.

        Args:
            texts: List of raw news headline texts.

        Returns:
            List of prediction dictionaries (same format as predict()).
        """
        start_time = time.perf_counter()

        try:
            # Step 1: Preprocess
            if self.preprocessor is not None:
                processed_texts = self.preprocessor.preprocess_batch(
                    texts, show_progress=False
                )
            else:
                processed_texts = texts

            # Step 2: Feature extraction
            features = self._extract_features(processed_texts)

            # Step 3: Predict
            if self.model_type == "classical":
                predictions, probabilities = self._predict_classical(features)
            else:
                predictions, probabilities = self._predict_deep_learning_batched(features)

            # Step 4: Decode labels
            if self.label_encoder is not None:
                class_names = self.label_encoder.inverse_transform(predictions)
            else:
                class_names = [
                    self.config.data.class_names[p]
                    if p < len(self.config.data.class_names)
                    else str(p)
                    for p in predictions
                ]

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Build results
            results = []
            for i in range(len(texts)):
                prob_dict = {}
                if probabilities is not None and self.label_encoder is not None:
                    for j, cls_name in enumerate(self.label_encoder.classes_):
                        prob_dict[cls_name] = float(probabilities[i, j])
                elif probabilities is not None:
                    for j, cls_name in enumerate(self.config.data.class_names):
                        if j < probabilities.shape[1]:
                            prob_dict[cls_name] = float(probabilities[i, j])

                results.append({
                    "text": texts[i],
                    "class": class_names[i] if isinstance(class_names[i], str) else str(class_names[i]),
                    "class_id": int(predictions[i]),
                    "confidence": float(probabilities[i].max()) if probabilities is not None else 1.0,
                    "probabilities": prob_dict,
                    "inference_time_ms": round(elapsed_ms / len(texts), 2),
                })

            return results

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def _extract_features(self, texts: List[str]) -> Any:
        """Extract features from preprocessed texts."""
        if self.feature_type == "distilbert" and self.vectorizer is not None:
            return self.vectorizer(
                texts, truncation=True, padding=True,
                max_length=self.config.distilbert.max_length, return_tensors="pt",
            )
        if self.vectorizer is not None:
            # RNN, attention, and Transformer checkpoints are trained on
            # fixed-length Word2Vec *sequences*.  The extractor also exposes
            # averaged document vectors, so select sequences explicitly
            # before its generic feature methods.
            if self.feature_type == "word2vec" and hasattr(self.vectorizer, "texts_to_sequences"):
                return self.vectorizer.texts_to_sequences(
                    texts, max_len=self.config.sequence.max_len
                )
            # Check if vectorizer has transform method (TF-IDF, etc.)
            if hasattr(self.vectorizer, "transform"):
                return self.vectorizer.transform(texts)
            # Check if it's a Word2Vec feature extractor
            elif hasattr(self.vectorizer, "get_document_vectors"):
                return self.vectorizer.get_document_vectors(texts)
            elif hasattr(self.vectorizer, "texts_to_sequences"):
                return self.vectorizer.texts_to_sequences(
                    texts, max_len=self.config.sequence.max_len
                )

        # If using vocabulary for sequence models
        if self.vocab is not None:
            sequences = self.vocab.texts_to_sequences(texts)
            max_len = self.config.sequence.max_len
            return self.vocab.pad_sequences(sequences, max_len)

        # Fallback: return raw texts
        logger.warning("No vectorizer or vocab found. Returning raw texts.")
        return texts

    def _predict_classical(
        self, features: Any
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Run prediction with a classical sklearn model."""
        predictions = self.model.predict(features)

        probabilities = None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)

        return np.asarray(predictions), probabilities

    def _predict_deep_learning_batched(
        self, features: Any
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Run deep-learning inference in bounded batches.

        Keeping all test rows in a single DistilBERT forward pass can require
        tens of gigabytes of VRAM.  This uses ``inference.batch_size`` and
        automatically halves a CUDA batch if the available GPU memory is
        temporarily smaller than expected.
        """
        if self.feature_type == "distilbert":
            total = int(features["input_ids"].shape[0])
        elif hasattr(features, "shape"):
            # scipy sparse matrices deliberately reject ``len(matrix)``
            # because it is ambiguous.  Rows are the batch dimension for
            # dense arrays and sparse TF-IDF matrices alike.
            total = int(features.shape[0])
        else:
            total = len(features)
        if total == 0:
            return np.array([], dtype=np.int64), np.empty((0, 0))

        batch_size = max(1, int(self.config.inference.batch_size))
        predictions_parts: List[np.ndarray] = []
        probability_parts: List[np.ndarray] = []
        offset = 0

        logger.info("Running deep-learning inference for %d rows (batch_size=%d)", total, batch_size)
        while offset < total:
            end = min(offset + batch_size, total)
            batch_features = self._slice_features(features, offset, end)
            try:
                predictions, probabilities = self._predict_deep_learning(batch_features)
            except torch.OutOfMemoryError as exc:
                if self.device.type != "cuda" or batch_size == 1:
                    raise RuntimeError(
                        "CUDA ran out of memory during inference even with batch_size=1. "
                        "Close other GPU workloads or run inference on CPU."
                    ) from exc
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                logger.warning("CUDA OOM at row %d; retrying with batch_size=%d", offset, batch_size)
                continue

            predictions_parts.append(predictions)
            if probabilities is not None:
                probability_parts.append(probabilities)
            offset = end

        all_predictions = np.concatenate(predictions_parts)
        all_probabilities = np.concatenate(probability_parts) if probability_parts else None
        return all_predictions, all_probabilities

    @staticmethod
    def _slice_features(features: Any, start: int, end: int) -> Any:
        """Slice NumPy, sparse, list, and Hugging Face feature batches."""
        if hasattr(features, "keys") and "input_ids" in features:
            return {key: value[start:end] for key, value in features.items()}
        return features[start:end]

    @torch.no_grad()
    def _predict_deep_learning(
        self, features: Any
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Run prediction with a PyTorch model."""
        # Hugging Face tokenizers return ``BatchEncoding`` (a mapping-like
        # object), not necessarily a plain dict.  Handle DistilBERT directly
        # so token IDs are never converted into a NumPy string array.
        if self.feature_type == "distilbert":
            inputs = features["input_ids"].to(self.device)
            attention_mask = features.get("attention_mask")
            logits = self.model(
                inputs, attention_mask=attention_mask.to(self.device) if attention_mask is not None else None
            )
            probabilities = F.softmax(logits, dim=-1).cpu().numpy()
            return probabilities.argmax(axis=1), probabilities

        if isinstance(features, dict):
            inputs = features["input_ids"].to(self.device)
            attention_mask = features.get("attention_mask")
            logits = self.model(
                inputs, attention_mask=attention_mask.to(self.device) if attention_mask is not None else None
            )
            probabilities = F.softmax(logits, dim=-1).cpu().numpy()
            return probabilities.argmax(axis=1), probabilities

        # Convert features to tensor
        if isinstance(features, np.ndarray):
            inputs = torch.FloatTensor(features).to(self.device)
        elif hasattr(features, "toarray"):
            # Sparse matrix (TF-IDF)
            inputs = torch.FloatTensor(features.toarray()).to(self.device)
        else:
            inputs = torch.FloatTensor(np.array(features)).to(self.device)

        # Forward pass.  Sequence models need the same padding mask used at
        # training time; zero Word2Vec vectors are reserved for padding.
        if self.feature_type == "word2vec" and inputs.ndim == 3:
            padding_mask = torch.isclose(inputs, torch.zeros(1, device=self.device)).all(dim=-1)
            logits = self.model(inputs, padding_mask=padding_mask)
        else:
            logits = self.model(inputs)
        probabilities = F.softmax(logits, dim=-1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)

        return predictions, probabilities

    def save(
        self,
        save_dir: str | Path,
        model_name: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save all classifier components to a directory.

        Args:
            save_dir: Directory to save all components.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model.  A state dict without architecture metadata cannot be
        # reconstructed, so derive supported built-in metadata when callers do
        # not pass it explicitly.
        if self.model_type == "deep_learning" and isinstance(
            self.model, torch.nn.Module
        ):
            model_name = model_name or self._infer_model_name()
            if model_name is None:
                raise ValueError(
                    "Deep-learning checkpoints require model_name/model_kwargs "
                    "for portable loading."
                )
            model_kwargs = model_kwargs or self._infer_model_kwargs(model_name)
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "model_name": model_name,
                    "model_kwargs": model_kwargs,
                },
                save_dir / "model.pt",
            )
        elif self.model_type == "classical":
            joblib.dump(self.model, save_dir / "model.joblib")

        # Save components
        if self.preprocessor is not None:
            joblib.dump(self.preprocessor, save_dir / "preprocessor.joblib")
        if self.vectorizer is not None:
            joblib.dump(self.vectorizer, save_dir / "vectorizer.joblib")
        if self.label_encoder is not None:
            joblib.dump(self.label_encoder, save_dir / "label_encoder.joblib")
        if self.vocab is not None:
            joblib.dump(self.vocab, save_dir / "vocab.joblib")

        # Save config
        if self.config is not None:
            self.config.save(save_dir / "config.yaml")

        if self.model_type == "deep_learning":
            pipeline = {
                "feature_type": self.feature_type,
                "model_name": model_name,
                "model_kwargs": model_kwargs or {},
                "class_names": list(self.label_encoder.classes_) if self.label_encoder is not None else list(self.config.data.class_names),
            }
            (save_dir / "pipeline_config.json").write_text(
                json.dumps(pipeline, indent=2), encoding="utf-8"
            )

        logger.info(f"Classifier saved to {save_dir}")

    def _infer_model_name(self) -> Optional[str]:
        """Map built-in class names to the model registry names."""
        return {
            "DNNClassifier": "dnn",
            "RecurrentClassifier": "rnn",
            "AttentionClassifier": "attention",
            "TransformerClassifier": "transformer",
            "DistilBERTClassifier": "distilbert",
        }.get(type(self.model).__name__)

    def _infer_model_kwargs(self, model_name: str) -> Dict[str, Any]:
        """Extract the reconstruction arguments exposed by built-in models."""
        model = self.model
        keys = {
            "dnn": ("input_dim", "num_classes", "hidden_layers", "dropout_rate", "activation_name", "use_batch_norm"),
            "rnn": ("input_size", "hidden_size", "num_layers", "num_classes", "cell_type", "bidirectional", "dropout_rate", "pooling", "use_layer_norm"),
            "attention": ("input_size", "hidden_size", "num_layers", "num_classes", "cell_type", "bidirectional", "dropout_rate", "attention_dim"),
            "transformer": ("d_model", "nhead", "num_encoder_layers", "dim_feedforward", "num_classes", "max_seq_len", "dropout_rate", "pooling"),
            "distilbert": ("model_name", "num_classes", "dropout_rate"),
        }[model_name]
        aliases = {"dropout_rate": "dropout", "activation_name": "activation", "use_batch_norm": "batch_norm"}
        result = {}
        for key in keys:
            if hasattr(model, key):
                result[aliases.get(key, key)] = getattr(model, key)
        if model_name == "distilbert":
            result["freeze_base"] = not any(parameter.requires_grad for parameter in model.distilbert.parameters())
        return result
