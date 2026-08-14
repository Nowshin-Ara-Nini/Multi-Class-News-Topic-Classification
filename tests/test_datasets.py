from __future__ import annotations

from configs.config import Config
from src.datasets import DataManager, NewsDataset, Vocabulary


def test_vocabulary_builds_sequences_and_post_pads():
    vocab = Vocabulary(max_size=5).build(["market rises market", "team wins"])
    sequences = vocab.texts_to_sequences(["market unknown"])
    padded = vocab.pad_sequences(sequences, max_len=4)

    assert vocab.word2idx["<PAD>"] == 0
    assert vocab.word2idx["<UNK>"] == 1
    assert padded.shape == (1, 4)
    assert padded[0, 0] == vocab.word2idx["market"]
    assert padded[0, 1] == 1
    assert padded[0, 2:].tolist() == [0, 0]


def test_news_dataset_returns_integer_tensors():
    vocab = Vocabulary().build(["market rises", "team wins"])
    dataset = NewsDataset(["market rises", "team wins"], [0, 1], vocab, max_len=3)
    sequence, label = dataset[0]
    assert sequence.shape == (3,)
    assert sequence.dtype.name == "int64" if hasattr(sequence.dtype, "name") else str(sequence.dtype) == "torch.int64"
    assert label.item() == 0


def test_data_manager_creates_stratified_splits(sample_texts, sample_labels):
    class IdentityPreprocessor:
        def preprocess_batch(self, texts):
            return texts

    class FakeSeries:
        def __init__(self, values):
            self._values = list(values)

        def astype(self, _type):
            return self

        def tolist(self):
            return list(self._values)

        @property
        def values(self):
            return self._values

    class FakeFrame:
        def __init__(self, text, label):
            self._columns = {"text": FakeSeries(text), "label": FakeSeries(label)}

        def __getitem__(self, key):
            return self._columns[key]

    config = Config()
    config.data.text_column = "text"
    config.data.label_column = "label"
    config.data.val_split = 0.25
    frame = FakeFrame(sample_texts, [f"class_{i}" for i in sample_labels])
    splits = DataManager(config).prepare_splits(frame, IdentityPreprocessor())

    assert len(splits["train_texts"]) == 12
    assert len(splits["val_texts"]) == 4
    assert set(splits["train_labels"]) == {0, 1, 2, 3}
    assert set(splits["val_labels"]) == {0, 1, 2, 3}
