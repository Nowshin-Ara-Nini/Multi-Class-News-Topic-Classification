from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix

from configs.config import Config
from src.inference import NewsClassifier
from src.models.rnn import RecurrentClassifier


class _Vectorizer:
    def get_document_vectors(self, texts):
        return np.asarray([[len(text), 1.0] for text in texts], dtype=np.float32)


class _CountingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, inputs):
        self.batch_sizes.append(inputs.shape[0])
        return torch.zeros((inputs.shape[0], 4), device=inputs.device)


class _Word2VecSequenceVectorizer:
    def __init__(self):
        self.sequence_calls = 0
        self.document_calls = 0

    def texts_to_sequences(self, texts, max_len):
        self.sequence_calls += 1
        sequences = np.zeros((len(texts), max_len, 6), dtype=np.float32)
        sequences[:, 0, :] = 1.0
        return sequences

    def get_document_vectors(self, texts):
        self.document_calls += 1
        return np.ones((len(texts), 6), dtype=np.float32)


def test_deep_learning_inference_honours_configured_batch_size(cpu_device):
    config = Config()
    config.inference.batch_size = 2
    model = _CountingModel()
    classifier = NewsClassifier(
        model=model,
        vectorizer=_Vectorizer(),
        config=config,
        device=cpu_device,
        feature_type="word2vec",
    )

    results = classifier.batch_predict(["one", "two", "three", "four", "five"])

    assert len(results) == 5
    assert model.batch_sizes == [2, 2, 1]


def test_deep_learning_inference_batches_sparse_tfidf_features(cpu_device):
    config = Config()
    config.inference.batch_size = 2
    model = _CountingModel()
    classifier = NewsClassifier(
        model=model,
        config=config,
        device=cpu_device,
        feature_type="tfidf",
    )
    features = csr_matrix(np.ones((5, 3), dtype=np.float32))

    predictions, probabilities = classifier._predict_deep_learning_batched(features)

    assert predictions.shape == (5,)
    assert probabilities.shape == (5, 4)
    assert model.batch_sizes == [2, 2, 1]


def test_word2vec_sequence_model_uses_sequence_features(cpu_device):
    config = Config()
    config.inference.batch_size = 2
    vectorizer = _Word2VecSequenceVectorizer()
    model = RecurrentClassifier(
        input_size=6,
        hidden_size=8,
        num_layers=1,
        num_classes=4,
        cell_type="rnn",
        bidirectional=True,
        dropout=0.0,
    )
    classifier = NewsClassifier(
        model=model,
        vectorizer=vectorizer,
        config=config,
        device=cpu_device,
        feature_type="word2vec",
    )

    results = classifier.batch_predict(["one", "two", "three"])

    assert len(results) == 3
    assert vectorizer.sequence_calls == 1
    assert vectorizer.document_calls == 0
