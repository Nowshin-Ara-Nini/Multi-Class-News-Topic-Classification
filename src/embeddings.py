"""
Feature extraction for News Topic Classification.

Provides two feature extraction strategies:
- ``TfidfFeatureExtractor``: Wraps scikit-learn's ``TfidfVectorizer`` with
  project-specific defaults, persistence, and logging.
- ``Word2VecFeatureExtractor``: Loads pretrained Word2Vec embeddings via
  ``gensim.downloader`` and produces both *averaged* document vectors
  (for LR / DNN) and *sequential* word-matrix representations (for RNN).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from configs.config import TfidfConfig, Word2VecConfig

logger = logging.getLogger(__name__)


# =========================================================================
# TF-IDF Feature Extractor
# =========================================================================


class TfidfFeatureExtractor:
    """TF-IDF feature extractor backed by scikit-learn.

    Wraps ``TfidfVectorizer`` with project defaults from ``TfidfConfig``,
    adds structured logging, and supports model persistence.

    Attributes:
        config: TF-IDF configuration dataclass.
        vectorizer: Underlying ``TfidfVectorizer`` instance.

    Example:
        >>> extractor = TfidfFeatureExtractor(TfidfConfig(max_features=20000))
        >>> X_train = extractor.fit_transform(train_texts)
        >>> X_test = extractor.transform(test_texts)
    """

    def __init__(self, config: Optional[TfidfConfig] = None) -> None:
        """Initialise the TF-IDF extractor.

        Args:
            config: ``TfidfConfig`` with vectorizer parameters.  Defaults
                are used when *None*.
        """
        self.config = config or TfidfConfig()

        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            sublinear_tf=self.config.sublinear_tf,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            dtype=np.float32,
        )
        self._is_fitted: bool = False

        logger.info(
            "TfidfFeatureExtractor initialised — max_features=%d, "
            "ngram_range=%s, sublinear_tf=%s",
            self.config.max_features,
            self.config.ngram_range,
            self.config.sublinear_tf,
        )

    # -----------------------------------------------------------------
    # Fit / Transform
    # -----------------------------------------------------------------

    def fit_transform(self, texts: List[str]) -> sparse.csr_matrix:
        """Fit the vectorizer on *texts* and return the TF-IDF matrix.

        Args:
            texts: List of preprocessed text strings.

        Returns:
            Sparse CSR matrix of shape ``(len(texts), max_features)``.
        """
        logger.info("Fitting TF-IDF on %d documents …", len(texts))

        matrix = self.vectorizer.fit_transform(texts)
        self._is_fitted = True

        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info(
            "TF-IDF fitted — vocabulary: %d terms, matrix shape: %s",
            vocab_size,
            matrix.shape,
        )
        return matrix

    def transform(self, texts: List[str]) -> sparse.csr_matrix:
        """Transform *texts* using the already-fitted vectorizer.

        Args:
            texts: List of preprocessed text strings.

        Returns:
            Sparse CSR matrix.

        Raises:
            RuntimeError: If the vectorizer has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "TfidfFeatureExtractor has not been fitted. "
                "Call fit_transform() first."
            )

        matrix = self.vectorizer.transform(texts)
        logger.info("TF-IDF transformed %d documents → %s", len(texts), matrix.shape)
        return matrix

    # -----------------------------------------------------------------
    # Feature names
    # -----------------------------------------------------------------

    def get_feature_names(self) -> List[str]:
        """Return the learned feature (term) names.

        Returns:
            List of term strings.

        Raises:
            RuntimeError: If the vectorizer has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("TfidfFeatureExtractor has not been fitted.")
        return list(self.vectorizer.get_feature_names_out())

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Pickle the fitted vectorizer to disk.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self.vectorizer, fh)
        logger.info("TF-IDF vectorizer saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path], config: Optional[TfidfConfig] = None) -> "TfidfFeatureExtractor":
        """Load a previously saved TF-IDF vectorizer.

        Args:
            path: Path to the pickled vectorizer.
            config: Optional ``TfidfConfig`` to attach.

        Returns:
            Reconstructed ``TfidfFeatureExtractor`` with the fitted
            vectorizer.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"TF-IDF vectorizer file not found: {path}")

        extractor = cls(config=config)
        with open(path, "rb") as fh:
            extractor.vectorizer = pickle.load(fh)
        extractor._is_fitted = True

        logger.info(
            "TF-IDF vectorizer loaded from %s (vocab=%d)",
            path,
            len(extractor.vectorizer.vocabulary_),
        )
        return extractor

    # -----------------------------------------------------------------
    # Representation
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        vocab_size = len(self.vectorizer.vocabulary_) if self._is_fitted else 0
        return (
            f"TfidfFeatureExtractor(max_features={self.config.max_features}, "
            f"fitted={self._is_fitted}, vocab={vocab_size})"
        )


# =========================================================================
# Word2Vec Feature Extractor
# =========================================================================


class Word2VecFeatureExtractor:
    """Word2Vec feature extractor using pretrained embeddings.

    Loads a pretrained model (default: ``word2vec-google-news-300``) via
    ``gensim.downloader`` and provides two output modes:

    1. **Averaged document vectors** — each document is represented by the
       mean of its constituent word vectors.  Suitable for Logistic
       Regression and DNN classifiers.
    2. **Sequential word matrices** — each document is a
       ``(max_len, dim)`` matrix of word vectors in order.  Suitable for
       RNN / GRU / LSTM models.

    Attributes:
        config: Word2Vec configuration dataclass.
        model: Loaded ``KeyedVectors`` model (or *None* before loading).
        dim: Embedding dimensionality.

    Example:
        >>> w2v = Word2VecFeatureExtractor(Word2VecConfig())
        >>> w2v.load_pretrained()
        >>> X_avg = w2v.get_document_vectors(train_texts)   # (N, 300)
        >>> X_seq = w2v.texts_to_sequences(train_texts, 35) # (N, 35, 300)
    """

    def __init__(self, config: Optional[Word2VecConfig] = None) -> None:
        """Initialise the Word2Vec extractor.

        Args:
            config: ``Word2VecConfig`` with model settings.
        """
        self.config = config or Word2VecConfig()
        self.model = None  # gensim.models.KeyedVectors
        self.dim: int = self.config.vector_size

        # OOV tracking
        self._oov_words: set = set()
        self._total_words: int = 0
        self._oov_count: int = 0

        logger.info(
            "Word2VecFeatureExtractor initialised — model=%s, dim=%d",
            self.config.pretrained_model,
            self.dim,
        )

    # -----------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------

    def load_pretrained(
        self, model_name: Optional[str] = None
    ) -> "Word2VecFeatureExtractor":
        """Load a pretrained Word2Vec model via ``gensim.downloader``.

        The model is cached locally by gensim after the first download.

        Args:
            model_name: Gensim model identifier.  Defaults to
                ``config.pretrained_model``.

        Returns:
            ``self`` for method chaining.
        """
        import gensim.downloader as gensim_api

        model_name = model_name or self.config.pretrained_model
        logger.info("Loading pretrained Word2Vec model: %s …", model_name)

        try:
            self.model = gensim_api.load(model_name)
            self.dim = self.model.vector_size
            logger.info(
                "Word2Vec model loaded — vocab=%d, dim=%d",
                len(self.model),
                self.dim,
            )
        except Exception:
            logger.exception("Failed to load Word2Vec model '%s'", model_name)
            raise

        return self

    # -----------------------------------------------------------------
    # Document vectors (averaged) — for LR / DNN
    # -----------------------------------------------------------------

    def get_document_vector(
        self,
        text: str,
        dim: Optional[int] = None,
    ) -> np.ndarray:
        """Compute the averaged word-vector representation of a document.

        Words not present in the vocabulary are skipped.  If *all* words
        are OOV the result is a zero vector.

        Args:
            text: Preprocessed text string.
            dim: Embedding dimension override (defaults to ``self.dim``).

        Returns:
            1-D NumPy array of shape ``(dim,)``.

        Raises:
            RuntimeError: If no pretrained model is loaded.
        """
        if self.model is None:
            raise RuntimeError(
                "No pretrained model loaded. Call load_pretrained() first."
            )

        dim = dim or self.dim
        words = text.split()

        if not words:
            return np.zeros(dim, dtype=np.float32)

        vectors: List[np.ndarray] = []
        for word in words:
            self._total_words += 1
            if word in self.model:
                vectors.append(self.model[word])
            else:
                self._oov_count += 1
                self._oov_words.add(word)

        if not vectors:
            return np.zeros(dim, dtype=np.float32)

        return np.mean(vectors, axis=0).astype(np.float32)

    def get_document_vectors(
        self,
        texts: List[str],
        dim: Optional[int] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Compute averaged document vectors for a batch of texts.

        Args:
            texts: List of preprocessed text strings.
            dim: Embedding dimension override.
            show_progress: Whether to show a ``tqdm`` progress bar.

        Returns:
            2-D NumPy array of shape ``(len(texts), dim)``.
        """
        if self.model is None:
            raise RuntimeError(
                "No pretrained model loaded. Call load_pretrained() first."
            )

        dim = dim or self.dim
        # Reset OOV counters for this batch
        self._oov_words = set()
        self._total_words = 0
        self._oov_count = 0

        logger.info("Computing averaged document vectors for %d texts …", len(texts))

        result = np.zeros((len(texts), dim), dtype=np.float32)
        iterator = tqdm(
            enumerate(texts),
            total=len(texts),
            desc="Document vectors",
            disable=not show_progress,
            unit="doc",
        )

        for i, text in iterator:
            result[i] = self.get_document_vector(text, dim=dim)

        oov_rate = (self._oov_count / self._total_words * 100) if self._total_words > 0 else 0.0
        logger.info(
            "Document vectors computed — shape=%s, OOV rate=%.2f%% "
            "(%d/%d words, %d unique OOV)",
            result.shape,
            oov_rate,
            self._oov_count,
            self._total_words,
            len(self._oov_words),
        )

        return result

    # -----------------------------------------------------------------
    # Sequence vectors — for RNN / GRU / LSTM
    # -----------------------------------------------------------------

    def text_to_sequence(
        self,
        text: str,
        max_len: int = 35,
        dim: Optional[int] = None,
    ) -> np.ndarray:
        """Convert text to a fixed-length sequence of word vectors.

        The output is a 2-D matrix of shape ``(max_len, dim)`` where each
        row is the embedding of the corresponding token position.  Shorter
        texts are zero-padded at the end; longer texts are truncated.

        Args:
            text: Preprocessed text string.
            max_len: Maximum sequence length.
            dim: Embedding dimension override.

        Returns:
            2-D NumPy array of shape ``(max_len, dim)``.

        Raises:
            RuntimeError: If no pretrained model is loaded.
        """
        if self.model is None:
            raise RuntimeError(
                "No pretrained model loaded. Call load_pretrained() first."
            )

        dim = dim or self.dim
        words = text.split()
        sequence = np.zeros((max_len, dim), dtype=np.float32)

        for i, word in enumerate(words[:max_len]):
            self._total_words += 1
            if word in self.model:
                sequence[i] = self.model[word]
            else:
                self._oov_count += 1
                self._oov_words.add(word)
            # OOV words remain zero; downstream masks safely ignore them.

        return sequence

    def texts_to_sequences(
        self,
        texts: List[str],
        max_len: int = 35,
        dim: Optional[int] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Convert a batch of texts to sequential word-vector matrices.

        Args:
            texts: List of preprocessed text strings.
            max_len: Maximum sequence length.
            dim: Embedding dimension override.
            show_progress: Whether to show a ``tqdm`` progress bar.

        Returns:
            3-D NumPy array of shape ``(len(texts), max_len, dim)``.
        """
        if self.model is None:
            raise RuntimeError(
                "No pretrained model loaded. Call load_pretrained() first."
            )

        dim = dim or self.dim
        self._oov_words = set()
        self._total_words = 0
        self._oov_count = 0
        logger.info(
            "Computing sequence vectors for %d texts (max_len=%d, dim=%d) …",
            len(texts),
            max_len,
            dim,
        )

        result = np.zeros((len(texts), max_len, dim), dtype=np.float32)
        iterator = tqdm(
            enumerate(texts),
            total=len(texts),
            desc="Sequence vectors",
            disable=not show_progress,
            unit="doc",
        )

        for i, text in iterator:
            result[i] = self.text_to_sequence(text, max_len=max_len, dim=dim)

        oov_rate = self._oov_count / self._total_words if self._total_words else 0.0
        log = logger.warning if oov_rate > 0.10 else logger.info
        log(
            "Sequence vectors computed: shape=%s, OOV rate=%.2f%% (%d/%d words)",
            result.shape, oov_rate * 100, self._oov_count, self._total_words,
        )
        return result

    # -----------------------------------------------------------------
    # Embedding matrix (for nn.Embedding initialisation)
    # -----------------------------------------------------------------

    def build_embedding_matrix(
        self,
        vocab: "Vocabulary",
        dim: Optional[int] = None,
    ) -> np.ndarray:
        """Build an embedding matrix aligned with a ``Vocabulary``.

        Each row ``i`` of the resulting matrix contains the pretrained
        embedding for ``vocab.idx2word[i]``.  Words not found in the
        pretrained model are initialised with small random values.

        Args:
            vocab: A built ``Vocabulary`` instance (from ``src.datasets``).
            dim: Embedding dimension override.

        Returns:
            2-D NumPy array of shape ``(vocab.size, dim)``.

        Raises:
            RuntimeError: If no pretrained model is loaded.
        """
        if self.model is None:
            raise RuntimeError(
                "No pretrained model loaded. Call load_pretrained() first."
            )

        dim = dim or self.dim
        matrix = np.random.uniform(-0.05, 0.05, (vocab.size, dim)).astype(np.float32)

        # Index 0 (PAD) should be zeros
        matrix[0] = np.zeros(dim, dtype=np.float32)

        found = 0
        for word, idx in vocab.word2idx.items():
            if word in self.model:
                matrix[idx] = self.model[word]
                found += 1

        coverage = found / vocab.size * 100 if vocab.size > 0 else 0.0
        logger.info(
            "Embedding matrix built — shape=%s, coverage=%.1f%% (%d/%d)",
            matrix.shape,
            coverage,
            found,
            vocab.size,
        )

        return matrix

    # -----------------------------------------------------------------
    # OOV diagnostics
    # -----------------------------------------------------------------

    def get_oov_stats(self) -> dict:
        """Return OOV statistics from the most recent batch operation.

        Returns:
            Dictionary with ``total_words``, ``oov_count``, ``oov_rate``,
            and ``unique_oov`` counts.
        """
        oov_rate = (
            self._oov_count / self._total_words * 100
            if self._total_words > 0
            else 0.0
        )
        return {
            "total_words": self._total_words,
            "oov_count": self._oov_count,
            "oov_rate_pct": round(oov_rate, 2),
            "unique_oov": len(self._oov_words),
        }

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save extractor state (config + OOV stats, NOT the gensim model).

        The pretrained model is large and cached by gensim; only
        lightweight metadata is saved here.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "pretrained_model": self.config.pretrained_model,
            "vector_size": self.dim,
            "oov_stats": self.get_oov_stats(),
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh)

        logger.info("Word2Vec extractor state saved to %s", path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        config: Optional[Word2VecConfig] = None,
        load_model: bool = True,
    ) -> "Word2VecFeatureExtractor":
        """Load extractor state and optionally reload the pretrained model.

        Args:
            path: Path to the saved state file.
            config: Optional ``Word2VecConfig`` override.
            load_model: Whether to also load the pretrained gensim model.

        Returns:
            Reconstructed ``Word2VecFeatureExtractor``.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Word2Vec extractor state not found: {path}"
            )

        with open(path, "rb") as fh:
            state = pickle.load(fh)

        effective_config = config or Word2VecConfig(
            pretrained_model=state["pretrained_model"],
            vector_size=state["vector_size"],
        )
        extractor = cls(config=effective_config)

        if load_model:
            extractor.load_pretrained(state["pretrained_model"])

        logger.info("Word2Vec extractor state loaded from %s", path)
        return extractor

    # -----------------------------------------------------------------
    # Representation
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        model_loaded = self.model is not None
        model_vocab = len(self.model) if model_loaded else 0
        return (
            f"Word2VecFeatureExtractor(model={self.config.pretrained_model!r}, "
            f"dim={self.dim}, loaded={model_loaded}, model_vocab={model_vocab})"
        )
