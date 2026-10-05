"""
Text preprocessing pipeline for News Topic Classification.

Provides three preprocessing modes with increasing aggressiveness:
- **raw**: Minimal cleaning (whitespace strip only).
- **extreme**: Heavy NLP pipeline (stopwords, lemmatization, short-word removal).
- **optimum**: Balanced pipeline that preserves important negation words.

Critical fixes from the original notebook:
- WordNetLemmatizer is instantiated ONCE in ``__init__``, not per call
  (was being created ~88k times during batch processing).
- Unused ``PorterStemmer`` removed from the optimum path.
- ``getTextFromRaw`` renamed to ``clean_raw`` for consistent snake_case.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Dict, List, Optional

from tqdm import tqdm

from configs.config import PreprocessingConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Contraction map — apostrophe variants (#39; from HTML entities)
_CONTRACTION_MAP: Dict[str, str] = {
    "won&#39;t": "will not",
    "can&#39;t": "cannot",
    "don&#39;t": "do not",
    "doesn&#39;t": "does not",
    "didn&#39;t": "did not",
    "isn&#39;t": "is not",
    "wasn&#39;t": "was not",
    "aren&#39;t": "are not",
    "weren&#39;t": "were not",
    "hasn&#39;t": "has not",
    "haven&#39;t": "have not",
    "hadn&#39;t": "had not",
    "wouldn&#39;t": "would not",
    "shouldn&#39;t": "should not",
    "couldn&#39;t": "could not",
    "mustn&#39;t": "must not",
    "needn&#39;t": "need not",
    "shan&#39;t": "shall not",
    "it&#39;s": "it is",
    "i&#39;m": "i am",
    "he&#39;s": "he is",
    "she&#39;s": "she is",
    "that&#39;s": "that is",
    "what&#39;s": "what is",
    "there&#39;s": "there is",
    "here&#39;s": "here is",
    "who&#39;s": "who is",
    "how&#39;s": "how is",
    "let&#39;s": "let us",
    "i&#39;ve": "i have",
    "we&#39;ve": "we have",
    "they&#39;ve": "they have",
    "you&#39;ve": "you have",
    "i&#39;ll": "i will",
    "we&#39;ll": "we will",
    "they&#39;ll": "they will",
    "you&#39;ll": "you will",
    "he&#39;ll": "he will",
    "she&#39;ll": "she will",
    "it&#39;ll": "it will",
    "i&#39;d": "i would",
    "we&#39;d": "we would",
    "they&#39;d": "they would",
    "you&#39;d": "you would",
    "he&#39;d": "he would",
    "she&#39;d": "she would",
    "won't": "will not",
    "can't": "cannot",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "wasn't": "was not",
    "aren't": "are not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't": "shall not",
    "it's": "it is",
    "i'm": "i am",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "what's": "what is",
    "there's": "there is",
    "here's": "here is",
    "who's": "who is",
    "how's": "how is",
    "let's": "let us",
    "i've": "i have",
    "we've": "we have",
    "they've": "they have",
    "you've": "you have",
    "i'll": "i will",
    "we'll": "we will",
    "they'll": "they will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "i'd": "i would",
    "we'd": "we would",
    "they'd": "they would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
}

# Negation / important words to keep even when removing stopwords
_IMPORTANT_WORDS = frozenset({
    "not", "no", "nor", "neither", "never", "none",
    "nobody", "nothing", "nowhere", "hardly", "scarcely",
    "barely", "against", "without", "but", "however",
    "despite", "although", "yet", "very", "most",
    "more", "less", "few", "much", "many",
})

# Pre-compiled regex patterns for performance
_RE_HTML_TAGS = re.compile(r"<[^>]+>")
_RE_HTML_ENTITIES = re.compile(r"&[a-zA-Z]+;|&#\d+;")
_RE_NON_ALPHA = re.compile(r"[^a-zA-Z\s]")
_RE_NON_ALPHA_SPACE = re.compile(r"[^a-zA-Z ]")
_RE_MULTI_SPACE = re.compile(r"\s+")


def generate_padding_mask(sequences):
    """Return a boolean mask where ``True`` marks an all-zero padded vector.

    Word2Vec sequences are represented as ``(batch, tokens, dimensions)``
    arrays.  Keeping this small helper in preprocessing makes the convention
    reusable and explicit across training and inference.
    """
    import numpy as np
    array = np.asarray(sequences)
    if array.ndim != 3:
        raise ValueError("sequences must have shape (batch, sequence, embedding)")
    return np.all(np.isclose(array, 0.0), axis=-1)


class TextPreprocessor:
    """Text preprocessing pipeline with configurable modes.

    Attributes:
        mode: Active preprocessing mode ('raw', 'extreme', or 'optimum').
        config: Preprocessing configuration dataclass.

    Example:
        >>> preprocessor = TextPreprocessor(mode='optimum')
        >>> preprocessor.preprocess("Breaking: Market hits <b>record</b> high!")
        'breaking market hit record high'
    """

    # Map mode names to method names
    _MODE_MAP = {
        "raw": "clean_raw",
        "extreme": "extreme_preprocess",
        "optimum": "optimum_preprocess",
    }

    def __init__(
        self,
        mode: str = "optimum",
        config: Optional[PreprocessingConfig] = None,
    ) -> None:
        """Initialise the text preprocessor.

        Args:
            mode: Preprocessing mode — one of 'raw', 'extreme', 'optimum'.
            config: Optional ``PreprocessingConfig`` override.  When *None*,
                a default config is created with the given *mode*.

        Raises:
            ValueError: If *mode* is not one of the supported modes.
        """
        if mode not in self._MODE_MAP:
            raise ValueError(
                f"Unknown preprocessing mode '{mode}'. "
                f"Choose from {list(self._MODE_MAP.keys())}"
            )

        self.mode = mode
        self.config = config or PreprocessingConfig(mode=mode)

        # Raw cleaning needs neither NLTK data nor network access.  Avoiding
        # the bootstrap here keeps prediction and tests fast/offline-safe.
        self._lemmatizer = None
        self._stopwords: frozenset[str] = frozenset()
        if self.mode != "raw":
            self._download_nltk_resources()
            from nltk.corpus import stopwords as _sw_corpus
            from nltk.stem import WordNetLemmatizer

            self._lemmatizer = WordNetLemmatizer()
            self._stopwords = frozenset(_sw_corpus.words(
                self.config.stopwords_language
            ))
        # Optimum mode keeps important negation words
        self._optimum_stopwords: frozenset[str] = self._stopwords - _IMPORTANT_WORDS

        # Cache for repeated texts
        self._cache: Dict[str, str] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        logger.info(
            "TextPreprocessor initialised — mode=%s, stopwords=%d, "
            "optimum_stopwords=%d (kept %d important words)",
            self.mode,
            len(self._stopwords),
            len(self._optimum_stopwords),
            len(self._stopwords) - len(self._optimum_stopwords),
        )

    # ------------------------------------------------------------------
    # NLTK bootstrap
    # ------------------------------------------------------------------

    @staticmethod
    def _download_nltk_resources() -> None:
        """Download required NLTK data packages if missing."""
        import nltk

        resources = [
            ("corpora/stopwords", "stopwords"),
            ("corpora/wordnet", "wordnet"),
            ("tokenizers/punkt_tab", "punkt_tab"),
            ("corpora/omw-1.4", "omw-1.4"),
        ]
        for path, package in resources:
            try:
                nltk.data.find(path)
            except LookupError:
                logger.info("Downloading NLTK resource: %s", package)
                try:
                    nltk.download(package, quiet=True)
                except Exception:
                    logger.exception(
                        "Failed to download NLTK resource '%s'", package
                    )
                    raise

    # ------------------------------------------------------------------
    # Public API — single text
    # ------------------------------------------------------------------

    def preprocess(self, text: str) -> str:
        """Preprocess a single text using the active mode.

        Args:
            text: Raw input text.

        Returns:
            Cleaned text string.
        """
        if not isinstance(text, str):
            logger.warning(
                "Non-string input (type=%s) coerced to str", type(text).__name__
            )
            text = str(text)

        # Cache lookup
        cache_key = f"{self.mode}::{text}"
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        self._cache_misses += 1

        method_name = self._MODE_MAP[self.mode]
        result: str = getattr(self, method_name)(text)

        self._cache[cache_key] = result
        return result

    def preprocess_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[str]:
        """Preprocess a batch of texts.

        Args:
            texts: List of raw text strings.
            show_progress: Whether to display a ``tqdm`` progress bar.

        Returns:
            List of cleaned text strings, same length as *texts*.
        """
        logger.info(
            "Batch preprocessing %d texts (mode=%s)", len(texts), self.mode
        )

        iterator = tqdm(
            texts,
            desc=f"Preprocessing ({self.mode})",
            disable=not show_progress,
            unit="doc",
        )
        results = [self.preprocess(t) for t in iterator]

        logger.info(
            "Batch preprocessing complete — cache hits=%d, misses=%d",
            self._cache_hits,
            self._cache_misses,
        )
        return results

    # ------------------------------------------------------------------
    # Mode 1: Raw (minimal)
    # ------------------------------------------------------------------

    def clean_raw(self, text: str) -> str:
        """Mode **raw** — strip leading/trailing whitespace only.

        Args:
            text: Raw input text.

        Returns:
            Whitespace-stripped text.
        """
        return text.strip()

    # ------------------------------------------------------------------
    # Mode 2: Extreme (heavy)
    # ------------------------------------------------------------------

    def extreme_preprocess(self, text: str) -> str:
        """Mode **extreme** — aggressive NLP preprocessing.

        Pipeline:
            1. Remove HTML tags and entities.
            2. Lowercase.
            3. Expand contractions (``#39;`` → apostrophe forms).
            4. Remove all non-alphabetic characters.
            5. Tokenise → remove stopwords → lemmatise.
            6. Remove short words (< ``min_word_length`` chars).

        Args:
            text: Raw input text.

        Returns:
            Heavily cleaned text string.
        """
        # 1. HTML
        text = _RE_HTML_TAGS.sub(" ", text)
        text = _RE_HTML_ENTITIES.sub(" ", text)

        # 2. Lowercase
        text = text.lower()

        # 3. Expand contractions
        text = self._expand_contractions(text)

        # 4. Remove non-alpha
        text = _RE_NON_ALPHA.sub(" ", text)

        # 5. Collapse whitespace & tokenise
        text = _RE_MULTI_SPACE.sub(" ", text).strip()
        tokens = text.split()

        # 6. Stopwords + lemmatise + short-word filter
        min_len = self.config.min_word_length
        tokens = [
            self._lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok not in self._stopwords and len(tok) >= min_len
        ]

        return " ".join(tokens)

    # ------------------------------------------------------------------
    # Mode 3: Optimum (balanced)
    # ------------------------------------------------------------------

    def optimum_preprocess(self, text: str) -> str:
        """Mode **optimum** — balanced preprocessing that keeps negations.

        Pipeline:
            1. Remove HTML tags and entities.
            2. Lowercase.
            3. Expand contractions.
            4. Controlled cleaning — keep alphabetic characters and spaces.
            5. Remove stopwords **except** important negation/modifier words.
            6. Lemmatise remaining tokens.

        Args:
            text: Raw input text.

        Returns:
            Cleaned text string retaining semantic nuance.
        """
        # 1. HTML
        text = _RE_HTML_TAGS.sub(" ", text)
        text = _RE_HTML_ENTITIES.sub(" ", text)

        # 2. Lowercase
        text = text.lower()

        # 3. Expand contractions
        text = self._expand_contractions(text)

        # 4. Controlled cleaning — keep only alpha + spaces
        text = _RE_NON_ALPHA_SPACE.sub(" ", text)

        # 5. Collapse whitespace & tokenise
        text = _RE_MULTI_SPACE.sub(" ", text).strip()
        tokens = text.split()

        # 6. Remove stopwords (keep important ones) + lemmatise
        tokens = [
            self._lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok not in self._optimum_stopwords
        ]

        return " ".join(tokens)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self, texts: List[str]) -> dict:
        """Compute corpus-level statistics after preprocessing.

        Args:
            texts: List of **already preprocessed** text strings.

        Returns:
            Dictionary with vocabulary size, average / min / max document
            length, total tokens, and cache performance counters.
        """
        if not texts:
            return {
                "vocab_size": 0,
                "avg_length": 0.0,
                "min_length": 0,
                "max_length": 0,
                "total_tokens": 0,
                "num_documents": 0,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
            }

        all_tokens: List[List[str]] = [t.split() for t in texts]
        lengths = [len(toks) for toks in all_tokens]
        vocab = {tok for toks in all_tokens for tok in toks}

        stats = {
            "vocab_size": len(vocab),
            "avg_length": round(sum(lengths) / len(lengths), 2),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "total_tokens": sum(lengths),
            "num_documents": len(texts),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }

        logger.info(
            "Corpus stats — vocab=%d, avg_len=%.1f, docs=%d",
            stats["vocab_size"],
            stats["avg_length"],
            stats["num_documents"],
        )
        return stats

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear the internal preprocessing cache."""
        size = len(self._cache)
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.debug("Preprocessing cache cleared (%d entries)", size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_contractions(text: str) -> str:
        """Replace HTML-encoded and standard contractions with expansions.

        Args:
            text: Lowercased text string.

        Returns:
            Text with contractions expanded.
        """
        for contraction, expansion in _CONTRACTION_MAP.items():
            if contraction in text:
                text = text.replace(contraction, expansion)
        return text

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TextPreprocessor(mode='{self.mode}', "
            f"stopwords={len(self._stopwords)}, "
            f"cached={len(self._cache)})"
        )
