"""
Central configuration management for News Topic Classification project.

Provides dataclass-based configuration with YAML serialization support.
All hyperparameters, paths, and settings are centralized here to eliminate
magic numbers scattered throughout the codebase.
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Literal


# ---------------------------------------------------------------------------
# Project root (two levels up from configs/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PathConfig:
    """File and directory paths."""
    project_root: str = str(PROJECT_ROOT)
    data_dir: str = str(PROJECT_ROOT / "data")
    train_csv: str = "Training_data_9.csv"
    test_csv: str = "Test_data.csv"
    saved_models_dir: str = str(PROJECT_ROOT / "saved_models")
    results_dir: str = str(PROJECT_ROOT / "results")
    logs_dir: str = str(PROJECT_ROOT / "logs")
    checkpoints_dir: str = str(PROJECT_ROOT / "checkpoints")

    @property
    def train_path(self) -> Path:
        return Path(self.data_dir) / self.train_csv

    @property
    def test_path(self) -> Path:
        return Path(self.data_dir) / self.test_csv

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        for d in [self.data_dir, self.saved_models_dir, self.results_dir,
                  self.logs_dir, self.checkpoints_dir]:
            os.makedirs(d, exist_ok=True)


@dataclass
class DataConfig:
    """Dataset and splitting configuration."""
    text_column: str = "News Headline"
    label_column: str = "News Topic"
    num_classes: int = 4
    class_names: List[str] = field(
        default_factory=lambda: ["Business", "Science and Technology", "Sports", "World News"]
    )
    val_split: float = 0.2
    test_split: float = 0.0  # Test set is a separate file
    remove_duplicates: bool = True
    random_state: int = 42
    split_manifest: Optional[str] = None


@dataclass
class PreprocessingConfig:
    """Text preprocessing settings."""
    # Available modes: 'raw', 'extreme', 'optimum'
    mode: str = "optimum"
    remove_html: bool = True
    lowercase: bool = True
    remove_stopwords: bool = True
    lemmatize: bool = True
    remove_special_chars: bool = True
    min_word_length: int = 2
    # Stopwords language
    stopwords_language: str = "english"


@dataclass
class TfidfConfig:
    """TF-IDF feature extraction settings."""
    max_features: int = 20_000
    ngram_range: tuple = (1, 2)
    sublinear_tf: bool = True
    min_df: int = 2
    max_df: float = 0.95
    use_char: bool = False
    char_max_features: int = 50_000


@dataclass
class ClassicalConfig:
    C: float = 1.0
    max_iter: int = 2000
    class_weight: Optional[str] = None


@dataclass
class Word2VecConfig:
    """Word2Vec / Skip-gram embedding settings."""
    # Pretrained model name (gensim)
    pretrained_model: str = "word2vec-google-news-300"
    vector_size: int = 300
    # For training custom Word2Vec
    window: int = 5
    min_count: int = 2
    workers: int = 4
    sg: int = 1  # 1 = Skip-gram, 0 = CBOW
    epochs: int = 10


@dataclass
class SequenceConfig:
    """Sequence model input settings."""
    max_len: int = 35
    embedding_dim: int = 300
    vocab_size: int = 50_000
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    trainable_embeddings: bool = False


@dataclass
class DNNConfig:
    """DNN (feed-forward) model hyperparameters."""
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.4
    activation: str = "relu"
    batch_norm: bool = False
    residual: bool = False


@dataclass
class RNNConfig:
    """Recurrent model hyperparameters (RNN/GRU/LSTM)."""
    cell_type: Literal["rnn", "gru", "lstm"] = "lstm"
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    # Pooling strategy: 'last', 'mean', 'max', 'mean_max'
    pooling: str = "mean_max"
    use_layer_norm: bool = True
    compact_head: bool = False


@dataclass
class AttentionConfig:
    """Attention-based model hyperparameters."""
    cell_type: Literal["gru", "lstm"] = "lstm"
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    attention_dim: int = 128
    combined_pooling: bool = False


@dataclass
class TransformerConfig:
    """Transformer encoder model hyperparameters."""
    d_model: int = 300
    nhead: int = 6
    num_encoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.3
    max_seq_len: int = 35
    pooling: str = "cls"  # 'cls', 'mean', 'max'
    normalize_input: bool = False


@dataclass
class DistilBERTConfig:
    """Configuration for the optional Hugging Face DistilBERT baseline."""
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    # A frozen encoder is the practical default for a portfolio baseline:
    # training stays fast and works on modest hardware.  Set this to False
    # when a full fine-tuning run and the required GPU memory are available.
    freeze_base: bool = True
    dropout: float = 0.3
    learning_rate: float = 2e-5
    pooling: str = "cls"
    head_lr_multiplier: float = 1.0


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    epochs: int = 15
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: str = "adam"  # 'adam', 'adamw', 'sgd'
    accumulation_steps: int = 1
    warmup_ratio: float = 0.06
    head_lr_multiplier: float = 1.0
    checkpoints_dir: str = "checkpoints"

    # Early stopping
    early_stopping: bool = True
    patience: int = 4
    min_delta: float = 1e-4
    monitor: str = "val_f1"  # 'val_loss', 'val_f1', 'val_acc'

    # Learning rate scheduling
    scheduler: str = "reduce_on_plateau"  # 'reduce_on_plateau', 'cosine', 'step', 'none'
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6

    # Gradient clipping
    gradient_clip: float = 5.0

    # Mixed precision
    mixed_precision: bool = True

    # Class weights
    use_class_weights: bool = True

    # Checkpointing
    save_best: bool = True
    save_last: bool = True

    # Logging
    log_interval: int = 100  # Log every N batches
    use_tensorboard: bool = False
    use_wandb: bool = False
    wandb_project: str = "news-classification"


@dataclass
class InferenceConfig:
    """Inference pipeline configuration."""
    model_path: Optional[str] = None
    preprocessor_path: Optional[str] = None
    vectorizer_path: Optional[str] = None
    label_encoder_path: Optional[str] = None
    device: Optional[str] = None  # None = auto-detect
    batch_size: int = 128


@dataclass
class Config:
    """Master configuration that contains all sub-configurations."""
    # Experiment metadata
    experiment_name: str = "news_topic_classification"
    seed: int = 42
    device: Optional[str] = None  # None = auto-detect

    # Sub-configs
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    word2vec: Word2VecConfig = field(default_factory=Word2VecConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    dnn: DNNConfig = field(default_factory=DNNConfig)
    rnn: RNNConfig = field(default_factory=RNNConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    distilbert: DistilBERTConfig = field(default_factory=DistilBERTConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def validate(self) -> None:
        """Validate configuration values before allocating training resources.

        ``assert`` is deliberately avoided here because Python can disable it
        with ``-O``.  Invalid configurations should always fail loudly.
        """
        if not 1e-6 <= self.training.learning_rate <= 1.0:
            raise ValueError("training.learning_rate must be in [1e-6, 1.0]")
        if not 1 <= self.training.batch_size <= 2048:
            raise ValueError("training.batch_size must be in [1, 2048]")
        if self.training.epochs < 1:
            raise ValueError("training.epochs must be at least 1")
        if not 0.0 <= self.rnn.dropout < 1.0:
            raise ValueError("rnn.dropout must be in [0, 1)")
        if not 0.0 <= self.attention.dropout < 1.0:
            raise ValueError("attention.dropout must be in [0, 1)")
        if not 0.0 <= self.transformer.dropout < 1.0:
            raise ValueError("transformer.dropout must be in [0, 1)")
        if self.rnn.hidden_size < 16 or self.attention.hidden_size < 16:
            raise ValueError("recurrent hidden sizes must be at least 16")
        if self.sequence.max_len < 5:
            raise ValueError("sequence.max_len must be at least 5")
        if self.transformer.max_seq_len < 1:
            raise ValueError("transformer.max_seq_len must be at least 1")
        if self.transformer.d_model % self.transformer.nhead:
            raise ValueError("transformer.d_model must be divisible by transformer.nhead")
        if not 8 <= self.distilbert.max_length <= 512:
            raise ValueError("distilbert.max_length must be in [8, 512]")
        if not 0.0 <= self.distilbert.dropout < 1.0:
            raise ValueError("distilbert.dropout must be in [0, 1)")
        if not 1e-7 <= self.distilbert.learning_rate <= 1e-2:
            raise ValueError("distilbert.learning_rate must be in [1e-7, 1e-2]")
        if self.training.optimizer not in {"adam", "adamw", "sgd"}:
            raise ValueError(f"unknown optimizer: {self.training.optimizer}")
        if self.training.scheduler not in {"reduce_on_plateau", "cosine", "step", "linear_warmup", "none", ""}:
            raise ValueError(f"unknown scheduler: {self.training.scheduler}")
        if not 0.0 < self.data.val_split < 1.0:
            raise ValueError("data.val_split must be between 0 and 1")
        if self.training.accumulation_steps < 1 or not 0 <= self.training.warmup_ratio < 1:
            raise ValueError("Invalid accumulation_steps or warmup_ratio")
        if self.training.monitor != "val_f1":
            raise ValueError("This experiment selects checkpoints by validation macro F1; use monitor=val_f1")
        if self.distilbert.pooling not in {"cls", "cls_mean"}:
            raise ValueError("distilbert.pooling must be cls or cls_mean")
        if self.classical.C <= 0 or self.classical.class_weight not in {None, "balanced"}:
            raise ValueError("Invalid classical regularization or class weight")
        if self.training.head_lr_multiplier <= 0 or self.distilbert.head_lr_multiplier <= 0:
            raise ValueError("Head learning rate multipliers must be positive")

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            # ``safe_dump`` serialises tuples as ordinary YAML sequences, so
            # the file remains readable by the matching ``safe_load`` call.
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "Config":
        """Recursively construct Config from a dictionary."""
        sub_configs = {}
        field_map = {
            "paths": PathConfig,
            "data": DataConfig,
            "preprocessing": PreprocessingConfig,
            "tfidf": TfidfConfig,
            "classical": ClassicalConfig,
            "word2vec": Word2VecConfig,
            "sequence": SequenceConfig,
            "dnn": DNNConfig,
            "rnn": RNNConfig,
            "attention": AttentionConfig,
            "transformer": TransformerConfig,
            "distilbert": DistilBERTConfig,
            "training": TrainingConfig,
            "inference": InferenceConfig,
        }

        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unsupported configuration sections: {sorted(unknown)}")
        for key, config_cls in field_map.items():
            if key in data and isinstance(data[key], dict):
                # Handle tuple fields (YAML loads them as lists)
                if key == "tfidf" and "ngram_range" in data[key]:
                    data[key]["ngram_range"] = tuple(data[key]["ngram_range"])
                sub_configs[key] = config_cls(**data[key])

        # Top-level fields
        top_level = {
            k: v for k, v in data.items()
            if k not in field_map and k in {f.name for f in cls.__dataclass_fields__.values()}
        }

        return cls(**top_level, **sub_configs)

    @classmethod
    def default(cls) -> "Config":
        """Create default configuration."""
        return cls()


def get_config(config_path: Optional[str | Path] = None) -> Config:
    """
    Get configuration. Loads from YAML if path provided, otherwise returns defaults.

    Args:
        config_path: Optional path to YAML configuration file.

    Returns:
        Config object with all settings.
    """
    if config_path is not None:
        return Config.from_yaml(config_path)
    return Config.default()
