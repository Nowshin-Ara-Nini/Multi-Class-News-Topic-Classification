from __future__ import annotations

import sys

import pytest

from main import _resolve_feature_types, parse_args
from src.inference import NewsClassifier


def test_full_suite_defaults_to_all_feature_modalities():
    assert _resolve_feature_types(None, None) == ["tfidf", "word2vec", "distilbert"]


@pytest.mark.parametrize(
    ("model", "feature"),
    [
        ("logistic_regression", "tfidf"),
        ("dnn", "tfidf"),
        ("rnn", "word2vec"),
        ("birnn", "word2vec"),
        ("gru", "word2vec"),
        ("bigru", "word2vec"),
        ("lstm", "word2vec"),
        ("bilstm", "word2vec"),
        ("attention", "word2vec"),
        ("transformer", "word2vec"),
        ("distilbert", "distilbert"),
    ],
)
def test_each_selected_model_uses_only_its_required_feature(model, feature):
    assert _resolve_feature_types(model, None) == [feature]
    assert _resolve_feature_types(model, "all") == [feature]


def test_cli_rejects_an_incompatible_model_feature_pair(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "train", "--model", "bilstm", "--features", "tfidf"],
    )
    with pytest.raises(SystemExit, match="2"):
        parse_args()


def test_legacy_checkpoint_has_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="pipeline_config.json"):
        NewsClassifier.from_checkpoint(tmp_path)
