from __future__ import annotations

import pytest

from configs.config import Config, get_config


def test_default_yaml_loads_and_uses_frozen_distilbert_base():
    config = get_config("configs/default.yaml")
    config.validate()
    assert config.distilbert.freeze_base is True


def test_config_round_trip_and_validation(tmp_path):
    config = Config()
    path = tmp_path / "config.yaml"
    config.save(path)
    loaded = Config.from_yaml(path)
    assert loaded.tfidf.ngram_range == (1, 2)

    loaded.training.batch_size = 0
    with pytest.raises(ValueError, match="batch_size"):
        loaded.validate()
