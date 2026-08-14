from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from configs.config import PreprocessingConfig
from src.preprocessing import TextPreprocessor, generate_padding_mask


class _IdentityLemmatizer:
    def lemmatize(self, token: str) -> str:
        return token


def _lightweight_preprocessor(mode: str) -> TextPreprocessor:
    """Build a mode instance without downloading NLTK corpora in unit tests."""
    processor = TextPreprocessor.__new__(TextPreprocessor)
    processor.mode = mode
    processor.config = PreprocessingConfig(mode=mode)
    processor._lemmatizer = _IdentityLemmatizer()
    processor._stopwords = frozenset({"this", "is", "the", "and", "not"})
    processor._optimum_stopwords = processor._stopwords - {"not"}
    return processor


def test_raw_mode_preserves_content_except_outer_whitespace():
    assert TextPreprocessor(mode="raw").preprocess("  Market UP!  ") == "Market UP!"


def test_extreme_and_optimum_modes_apply_their_distinct_stopword_rules():
    text = "This isn't <b>good</b> and not bad!"
    extreme = _lightweight_preprocessor("extreme").extreme_preprocess(text)
    optimum = _lightweight_preprocessor("optimum").optimum_preprocess(text)

    assert "good" in extreme and "bad" in extreme
    assert "not" not in extreme
    assert "not" in optimum


def test_preprocessor_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown preprocessing mode"):
        TextPreprocessor(mode="unsupported")


def test_padding_mask_marks_only_all_zero_vectors():
    sequences = np.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 2.0], [0.0, 0.0]]])
    mask = generate_padding_mask(sequences)
    assert mask.tolist() == [[False, True], [False, True]]

    with pytest.raises(ValueError, match="shape"):
        generate_padding_mask(np.zeros((2, 3)))
