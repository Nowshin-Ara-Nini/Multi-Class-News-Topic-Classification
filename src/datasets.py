"""
Dataset management for News Topic Classification.

Provides:
- ``Vocabulary``: Pure-Python tokeniser that replaces TensorFlow's Tokenizer.
  Supports word→index mapping, sequence conversion, and padding.
- ``NewsDataset``: PyTorch ``Dataset`` for sequence models (padded index
  sequences + labels).
- ``DataManager``: End-to-end pipeline for CSV loading, preprocessing,
  stratified train/val splitting, label encoding, and ``DataLoader`` creation.
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from configs.config import Config, DataConfig, SequenceConfig
from src.utils import set_seed, setup_logging

logger = logging.getLogger(__name__)

# PyArrow is optional for this project.  Some Windows/PyTorch environments
# crash natively while pandas builds Arrow-backed string arrays, so keep CSV
# text columns on pandas' standard Python/object backend instead.
try:
    pd.options.future.infer_string = False
except (AttributeError, KeyError):
    pass


# =========================================================================
# Vocabulary
# =========================================================================


class Vocabulary:
    """Word-level vocabulary that converts text to padded index sequences.

    Replaces TensorFlow / Keras ``Tokenizer`` with a lightweight,
    serialisable implementation compatible with PyTorch workflows.

    Attributes:
        max_size: Maximum vocabulary size (including special tokens).
        pad_token: Token used for padding (index 0).
        unk_token: Token used for unknown / OOV words (index 1).
        word2idx: Mapping from word to integer index.
        idx2word: Mapping from integer index to word.

    Example:
        >>> vocab = Vocabulary(max_size=5000).build(train_texts)
        >>> seqs = vocab.texts_to_sequences(["market hits record high"])
        >>> padded = vocab.pad_sequences(seqs, max_len=35)
    """

    def __init__(
        self,
        max_size: int = 50_000,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
    ) -> None:
        """Initialise vocabulary with special tokens.

        Args:
            max_size: Maximum number of words to keep (most frequent).
            pad_token: Padding token string.
            unk_token: Unknown-word token string.
        """
        self.max_size = max_size
        self.pad_token = pad_token
        self.unk_token = unk_token

        # Reserve index 0 for padding, 1 for unknown
        self.word2idx: Dict[str, int] = {pad_token: 0, unk_token: 1}
        self.idx2word: Dict[int, str] = {0: pad_token, 1: unk_token}
        self._is_built: bool = False

    # -----------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------

    def build(self, texts: List[str]) -> "Vocabulary":
        """Build vocabulary from a corpus of texts.

        Words are ranked by frequency; only the top ``max_size - 2`` words
        are kept (two slots are reserved for PAD and UNK).

        Args:
            texts: List of **preprocessed** text strings (space-separated
                tokens).

        Returns:
            ``self`` for method chaining.
        """
        logger.info("Building vocabulary from %d texts (max_size=%d)", len(texts), self.max_size)

        counter: Counter[str] = Counter()
        for text in texts:
            if isinstance(text, str):
                counter.update(text.split())

        # Keep top (max_size - 2) words; 2 reserved for PAD and UNK
        most_common = counter.most_common(self.max_size - 2)

        # Reset mappings
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx2word = {0: self.pad_token, 1: self.unk_token}

        for idx_offset, (word, _count) in enumerate(most_common, start=2):
            self.word2idx[word] = idx_offset
            self.idx2word[idx_offset] = word

        self._is_built = True

        logger.info(
            "Vocabulary built — %d unique tokens, %d retained (+ PAD, UNK)",
            len(counter),
            len(self.word2idx) - 2,
        )
        return self

    # -----------------------------------------------------------------
    # Conversion
    # -----------------------------------------------------------------

    def text_to_indices(self, text: str) -> List[int]:
        """Convert a single text to a list of word indices.

        Args:
            text: Preprocessed text string.

        Returns:
            List of integer indices.
        """
        unk_idx = self.word2idx[self.unk_token]
        return [self.word2idx.get(w, unk_idx) for w in text.split()]

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Convert multiple texts to lists of word indices.

        Args:
            texts: List of preprocessed text strings.

        Returns:
            List of integer-index lists.
        """
        return [self.text_to_indices(t) for t in texts]

    def pad_sequences(
        self,
        sequences: List[List[int]],
        max_len: int,
        padding: str = "post",
        truncating: str = "post",
    ) -> np.ndarray:
        """Pad / truncate sequences to a fixed length.

        Args:
            sequences: List of integer-index lists.
            max_len: Target sequence length.
            padding: Where to add padding — ``'post'`` (right) or
                ``'pre'`` (left).
            truncating: Where to truncate — ``'post'`` (right) or
                ``'pre'`` (left).

        Returns:
            NumPy array of shape ``(len(sequences), max_len)`` with dtype
            ``int64``.
        """
        pad_idx = self.word2idx[self.pad_token]
        result = np.full((len(sequences), max_len), pad_idx, dtype=np.int64)

        for i, seq in enumerate(sequences):
            if len(seq) == 0:
                continue

            # Truncate
            if len(seq) > max_len:
                if truncating == "post":
                    seq = seq[:max_len]
                else:
                    seq = seq[-max_len:]

            # Place into array
            if padding == "post":
                result[i, : len(seq)] = seq
            else:
                result[i, -len(seq) :] = seq

        return result

    # -----------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save vocabulary to a JSON file.

        Args:
            path: Destination file path (should end with ``.json``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "max_size": self.max_size,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "word2idx": self.word2idx,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)

        logger.info("Vocabulary saved to %s (%d words)", path, len(self.word2idx))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Vocabulary":
        """Load vocabulary from a JSON file.

        Args:
            path: Path to a previously saved vocabulary file.

        Returns:
            Reconstructed ``Vocabulary`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        vocab = cls(
            max_size=data["max_size"],
            pad_token=data["pad_token"],
            unk_token=data["unk_token"],
        )
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        vocab._is_built = True

        logger.info("Vocabulary loaded from %s (%d words)", path, len(vocab.word2idx))
        return vocab

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def size(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self.word2idx)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"Vocabulary(size={self.size}, max_size={self.max_size}, "
            f"built={self._is_built})"
        )

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx


# =========================================================================
# PyTorch Dataset
# =========================================================================


class NewsDataset(Dataset):
    """PyTorch Dataset for news headline classification.

    Returns padded index sequences and integer labels, suitable for
    sequence models (RNN / GRU / LSTM / Transformer).

    Attributes:
        sequences: Padded integer sequences of shape ``(N, max_len)``.
        labels: Integer-encoded labels of shape ``(N,)``.
    """

    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        vocab: Vocabulary,
        max_len: int = 35,
    ) -> None:
        """Create a dataset from texts and labels.

        Args:
            texts: List of preprocessed text strings.
            labels: Integer-encoded label array.
            vocab: Built ``Vocabulary`` instance.
            max_len: Maximum sequence length for padding/truncation.
        """
        if len(texts) != len(labels):
            raise ValueError(
                f"texts ({len(texts)}) and labels ({len(labels)}) "
                "must have the same length"
            )

        sequences = vocab.texts_to_sequences(texts)
        self.sequences: np.ndarray = vocab.pad_sequences(sequences, max_len=max_len)
        self.labels: np.ndarray = np.asarray(labels, dtype=np.int64)

        logger.debug(
            "NewsDataset created — %d samples, max_len=%d", len(self), max_len
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a (sequence, label) pair as tensors.

        Args:
            idx: Sample index.

        Returns:
            Tuple of ``(sequence_tensor, label_tensor)``.
        """
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence, label


class DistilBERTDataset(Dataset):
    """Tokenized text dataset for Hugging Face sequence classifiers."""

    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer: Any, max_length: int = 128) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_length,
            return_attention_mask=True,
        )
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# =========================================================================
# DataManager
# =========================================================================


class DataManager:
    """End-to-end data pipeline manager.

    Handles:
    - CSV loading with validation.
    - Duplicate removal.
    - Stratified train / validation splitting.
    - Label encoding (sklearn ``LabelEncoder``).
    - Vocabulary building.
    - PyTorch ``DataLoader`` creation.

    Attributes:
        config: Master project ``Config``.
        label_encoder: Fitted ``LabelEncoder`` mapping topic strings to ints.
    """

    def __init__(self, config: Config) -> None:
        """Initialise DataManager.

        Args:
            config: Project-wide ``Config`` dataclass.
        """
        self.config = config
        self.label_encoder: LabelEncoder = LabelEncoder()
        self._data_config: DataConfig = config.data
        self._seq_config: SequenceConfig = config.sequence

        logger.info("DataManager initialised")

    # -----------------------------------------------------------------
    # CSV loading
    # -----------------------------------------------------------------

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test CSV files.

        Returns:
            Tuple of ``(train_df, test_df)`` DataFrames.

        Raises:
            FileNotFoundError: If either CSV is missing.
            ValueError: If required columns are absent.
        """
        train_path = self.config.paths.train_path
        test_path = self.config.paths.test_path

        # --- Validate paths ---
        if not train_path.exists():
            raise FileNotFoundError(
                f"Training CSV not found: {train_path}\n"
                f"Expected location: {train_path.resolve()}\n"
                f"Please place '{self.config.paths.train_csv}' in "
                f"'{self.config.paths.data_dir}/'."
            )
        if not test_path.exists():
            raise FileNotFoundError(
                f"Test CSV not found: {test_path}\n"
                f"Expected location: {test_path.resolve()}\n"
                f"Please place '{self.config.paths.test_csv}' in "
                f"'{self.config.paths.data_dir}/'."
            )

        # --- Load ---
        train_df = pd.read_csv(str(train_path))
        test_df = pd.read_csv(str(test_path))

        # --- Validate columns ---
        required = {self._data_config.text_column, self._data_config.label_column}
        for name, df in [("Training", train_df), ("Test", test_df)]:
            missing = required - set(df.columns)
            if missing:
                raise ValueError(
                    f"{name} CSV is missing required columns: {missing}. "
                    f"Available columns: {list(df.columns)}"
                )

        logger.info(
            "Data loaded — train: %d rows, test: %d rows",
            len(train_df),
            len(test_df),
        )

        # --- Drop duplicates ---
        if self._data_config.remove_duplicates:
            before_train = len(train_df)
            before_test = len(test_df)
            train_df = train_df.drop_duplicates(
                subset=[self._data_config.text_column]
            ).reset_index(drop=True)
            test_df = test_df.drop_duplicates(
                subset=[self._data_config.text_column]
            ).reset_index(drop=True)
            logger.info(
                "Duplicates removed — train: %d→%d, test: %d→%d",
                before_train, len(train_df),
                before_test, len(test_df),
            )

        # --- Drop NaN rows in text/label columns ---
        for name, df_ref in [("train", train_df), ("test", test_df)]:
            n_before = len(df_ref)
            df_ref.dropna(
                subset=[self._data_config.text_column, self._data_config.label_column],
                inplace=True,
            )
            n_dropped = n_before - len(df_ref)
            if n_dropped > 0:
                logger.warning(
                    "Dropped %d NaN rows from %s set", n_dropped, name
                )

        return train_df, test_df

    # -----------------------------------------------------------------
    # Preprocessing + splitting
    # -----------------------------------------------------------------

    def prepare_splits(
        self,
        df: pd.DataFrame,
        preprocessor: Any,
    ) -> Dict[str, Any]:
        """Preprocess text and create stratified train / validation splits.

        Args:
            df: Training DataFrame (must contain text and label columns).
            preprocessor: A ``TextPreprocessor`` instance with a
                ``preprocess_batch`` method.

        Returns:
            Dictionary with keys:
                - ``'train_texts'``: List[str]
                - ``'val_texts'``: List[str]
                - ``'train_labels'``: np.ndarray (int-encoded)
                - ``'val_labels'``: np.ndarray (int-encoded)
                - ``'label_encoder'``: fitted LabelEncoder
                - ``'class_names'``: List[str]
        """
        text_col = self._data_config.text_column
        label_col = self._data_config.label_column

        # --- Preprocess texts ---
        texts: List[str] = preprocessor.preprocess_batch(
            df[text_col].astype(str).tolist()
        )

        # --- Encode labels ---
        self.label_encoder.fit(df[label_col].values)
        labels: np.ndarray = self.label_encoder.transform(df[label_col].values)

        logger.info(
            "Labels encoded — classes: %s",
            dict(zip(
                self.label_encoder.classes_,
                range(len(self.label_encoder.classes_)),
            )),
        )

        # --- Stratified split ---
        set_seed(self._data_config.random_state)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts,
            labels,
            test_size=self._data_config.val_split,
            random_state=self._data_config.random_state,
            stratify=labels,
        )

        logger.info(
            "Stratified split — train: %d, val: %d (val_ratio=%.2f)",
            len(train_texts),
            len(val_texts),
            self._data_config.val_split,
        )

        return {
            "train_texts": train_texts,
            "val_texts": val_texts,
            "train_labels": train_labels,
            "val_labels": val_labels,
            "label_encoder": self.label_encoder,
            "class_names": list(self.label_encoder.classes_),
        }

    # -----------------------------------------------------------------
    # DataLoader creation
    # -----------------------------------------------------------------

    def create_dataloaders(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: List[str],
        val_labels: np.ndarray,
        vocab: Vocabulary,
        batch_size: Optional[int] = None,
        max_len: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        test_texts: Optional[List[str]] = None,
        test_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, DataLoader]:
        """Build PyTorch DataLoaders for train, validation, and optionally test.

        Args:
            train_texts: Preprocessed training texts.
            train_labels: Integer-encoded training labels.
            val_texts: Preprocessed validation texts.
            val_labels: Integer-encoded validation labels.
            vocab: Built ``Vocabulary`` instance.
            batch_size: Batch size (defaults to ``config.training.batch_size``).
            max_len: Sequence length (defaults to ``config.sequence.max_len``).
            num_workers: ``DataLoader`` worker processes.
            pin_memory: Pin tensors to CUDA-pinned memory.
            test_texts: Optional preprocessed test texts.
            test_labels: Optional integer-encoded test labels.

        Returns:
            Dictionary with ``'train'``, ``'val'``, and optionally ``'test'``
            ``DataLoader`` instances.
        """
        batch_size = batch_size or self.config.training.batch_size
        max_len = max_len or self._seq_config.max_len

        # --- Build datasets ---
        train_ds = NewsDataset(train_texts, train_labels, vocab, max_len)
        val_ds = NewsDataset(val_texts, val_labels, vocab, max_len)

        loaders: Dict[str, DataLoader] = {
            "train": DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
            ),
            "val": DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
            ),
        }

        if test_texts is not None and test_labels is not None:
            test_ds = NewsDataset(test_texts, test_labels, vocab, max_len)
            loaders["test"] = DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
            )

        for split_name, loader in loaders.items():
            logger.info(
                "DataLoader '%s' — %d samples, %d batches (batch_size=%d)",
                split_name,
                len(loader.dataset),
                len(loader),
                batch_size,
            )

        return loaders

    # -----------------------------------------------------------------
    # Persistence helpers
    # -----------------------------------------------------------------

    def save_label_encoder(self, path: Union[str, Path]) -> None:
        """Pickle the fitted LabelEncoder.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self.label_encoder, fh)
        logger.info("LabelEncoder saved to %s", path)

    def load_label_encoder(self, path: Union[str, Path]) -> LabelEncoder:
        """Load a previously saved LabelEncoder.

        Args:
            path: Path to the pickled ``LabelEncoder``.

        Returns:
            Loaded ``LabelEncoder`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LabelEncoder file not found: {path}")

        with open(path, "rb") as fh:
            self.label_encoder = pickle.load(fh)

        logger.info(
            "LabelEncoder loaded from %s — classes: %s",
            path,
            list(self.label_encoder.classes_),
        )
        return self.label_encoder

    # -----------------------------------------------------------------
    # Representation
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DataManager(val_split={self._data_config.val_split}, "
            f"num_classes={self._data_config.num_classes})"
        )
