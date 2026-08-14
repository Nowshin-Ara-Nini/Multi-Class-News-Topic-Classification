from __future__ import annotations

import numpy as np
import torch

from configs.config import Config
from src.inference import NewsClassifier


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
