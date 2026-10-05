"""Regression coverage for the new protocol. Run explicitly with pytest."""
import json
from collections import UserDict

import numpy as np
import pandas as pd
import pytest
import torch

from configs.config import Config, TrainingConfig
from src.models import create_model
from src.sequence_features import WordVocabulary
from src.training_data import training_split
from src.trainer import Trainer


def test_training_split_never_opens_test_csv(tmp_path, monkeypatch):
    config = Config()
    config.paths.data_dir = str(tmp_path)
    config.paths.results_dir = str(tmp_path / "results")
    rows = [{"News Headline": f"{label} sample {i}", "News Topic": label}
            for label in config.data.class_names for i in range(10)]
    pd.DataFrame(rows).to_csv(config.paths.train_path, index=False)
    original_read_csv = pd.read_csv
    def guarded_read_csv(path, *args, **kwargs):
        assert str(path).endswith("Training_data_9.csv")
        return original_read_csv(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    train, val, checksum = training_split(config)
    assert len(train) + len(val) == 40
    assert not set(train["_identity"]) & set(val["_identity"])
    assert training_split(config)[2] == checksum
    assert not config.paths.test_path.exists()


def test_conflicting_training_labels_are_quarantined(tmp_path):
    config = Config()
    config.paths.data_dir = str(tmp_path)
    config.paths.results_dir = str(tmp_path / "results")
    rows = [{"News Headline": f"{label} sample {i}", "News Topic": label}
            for label in config.data.class_names for i in range(10)]
    rows += [{"News Headline": "Same Story!", "News Topic": "Sports"},
             {"News Headline": "same story", "News Topic": "Business"}]
    pd.DataFrame(rows).to_csv(config.paths.train_path, index=False)
    train, val, _ = training_split(config)
    assert "same story" not in set(train["_identity"]) | set(val["_identity"])


def test_unknown_words_are_content_and_padding_is_invariant():
    vocabulary = WordVocabulary(max_length=6).fit(["first middle last"])
    ids = vocabulary.transform(["first unknown last", ""])
    assert ids[0, 1] == 1 and ids[0, 2] == vocabulary.indices["last"]
    assert ids[1, 0] == 1
    model = create_model("embedded_sequence", encoder_name="rnn",
                         encoder_kwargs={"input_size": 8, "hidden_size": 16, "num_layers": 1,
                                         "cell_type": "gru", "bidirectional": True, "dropout": 0.},
                         vocab_size=len(vocabulary.words), embedding_dim=8).eval()
    tokens = torch.from_numpy(ids)
    assert torch.allclose(model(tokens), model(torch.cat([tokens, torch.zeros(2, 3, dtype=torch.long)], 1)), atol=1e-6)
    model(tokens).sum().backward()
    assert model.embedding.weight.grad[0].eq(0).all()
    assert model.embedding.weight.grad[1].abs().sum() > 0


def test_transformer_layers_have_independent_attention_initialization():
    model = create_model("transformer", d_model=12, nhead=3, num_encoder_layers=2)
    first, second = model.transformer_encoder.layers
    assert not torch.equal(first.self_attn.in_proj_weight, second.self_attn.in_proj_weight)


def test_unknown_configuration_section_is_rejected():
    with pytest.raises(ValueError, match="Unsupported configuration"):
        Config._from_dict({"trainging": {"epochs": 1}})


def test_huggingface_mapping_batch_retains_attention_mask():
    trainer = Trainer(torch.nn.Linear(3, 4), TrainingConfig(), torch.device("cpu"))
    batch = UserDict(input_ids=torch.ones(2, 3, dtype=torch.long), labels=torch.zeros(2, dtype=torch.long),
                     attention_mask=torch.tensor([[1, 1, 0], [1, 0, 0]]))
    _, _, kwargs = trainer._unpack_batch(batch)
    assert torch.equal(kwargs["attention_mask"], batch["attention_mask"])


def test_new_sequence_artifact_roundtrip(tmp_path):
    import joblib
    from sklearn.preprocessing import LabelEncoder
    from src.inference import NewsClassifier
    from src.preprocessing import TextPreprocessor
    config = Config()
    config.save(tmp_path / "config.yaml")
    vocab = WordVocabulary(max_length=6).fit(["a news story"])
    kwargs = dict(encoder_name="rnn", encoder_kwargs=dict(input_size=8, hidden_size=16, num_layers=1),
                  vocab_size=len(vocab.words), embedding_dim=8)
    model = create_model("embedded_sequence", **kwargs).eval()
    torch.save(dict(model_name="embedded_sequence", model_kwargs=kwargs, model_state_dict=model.state_dict()), tmp_path / "model.pt")
    (tmp_path / "pipeline_config.json").write_text(json.dumps(dict(feature_type="word2vec_ids", model_name="embedded_sequence", model_kwargs=kwargs)))
    joblib.dump(vocab, tmp_path / "vectorizer.joblib")
    joblib.dump(TextPreprocessor("raw"), tmp_path / "preprocessor.joblib")
    joblib.dump(LabelEncoder().fit(config.data.class_names), tmp_path / "label_encoder.joblib")
    classifier = NewsClassifier.from_checkpoint(tmp_path, device="cpu")
    with torch.no_grad():
        expected = model(torch.from_numpy(vocab.transform(["a unknown story"]))).softmax(-1).numpy()[0]
    prediction = classifier.predict("a unknown story")
    assert np.allclose(list(prediction["probabilities"].values()), expected, atol=1e-6)
