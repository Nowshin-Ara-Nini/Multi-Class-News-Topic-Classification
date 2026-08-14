from __future__ import annotations

import numpy as np
import torch

from src.models import create_model, list_models
from src.models.attention import AttentionClassifier
from src.models.classical import ClassicalModelWrapper
from src.models.dnn import DNNClassifier
from src.models.rnn import RecurrentClassifier
from src.models.transformer import TransformerClassifier


def test_registered_neural_models_produce_class_logits():
    dense_input = torch.randn(3, 8)
    sequence_input = torch.randn(3, 5, 6)
    padding_mask = torch.tensor([[False, False, False, True, True]] * 3)
    models_and_inputs = [
        (DNNClassifier(input_dim=8, hidden_layers=[6], dropout=0.0), dense_input),
        (RecurrentClassifier(input_size=6, hidden_size=8, num_layers=1, bidirectional=False, dropout=0.0), sequence_input),
        (AttentionClassifier(input_size=6, hidden_size=8, num_layers=1, bidirectional=False, attention_dim=4, dropout=0.0), sequence_input),
        (TransformerClassifier(d_model=6, nhead=2, num_encoder_layers=1, dim_feedforward=12, dropout=0.0), sequence_input),
    ]

    for model, inputs in models_and_inputs:
        model.eval()
        with torch.no_grad():
            output = model(inputs, padding_mask=padding_mask) if inputs.ndim == 3 else model(inputs)
        assert output.shape == (3, 4)


def test_attention_assigns_no_weight_to_padding():
    model = AttentionClassifier(input_size=4, hidden_size=6, num_layers=1, bidirectional=False, attention_dim=3, dropout=0.0)
    inputs = torch.randn(2, 4, 4)
    mask = torch.tensor([[False, False, True, True], [False, True, True, True]])
    _, weights = model(inputs, padding_mask=mask, return_attention=True)
    assert torch.allclose(weights[mask], torch.zeros_like(weights[mask]), atol=1e-7)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))


def test_model_registry_and_classical_wrapper_work_on_synthetic_data():
    assert {"dnn", "rnn", "attention", "transformer", "distilbert"}.issubset(list_models())
    assert create_model("dnn", input_dim=4, hidden_layers=[3]).num_classes == 4

    wrapper = ClassicalModelWrapper("logistic_regression")
    features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 2)
    labels = np.array([0, 1, 0, 1] * 2)
    wrapper.fit(features, labels)
    assert wrapper.predict(features).shape == labels.shape
