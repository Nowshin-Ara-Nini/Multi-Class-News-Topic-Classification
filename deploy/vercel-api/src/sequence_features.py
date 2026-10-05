"""Portable training-vocabulary tokenization; PAD=0 and UNK=1."""
from collections import Counter
import numpy as np


class WordVocabulary:
    def __init__(self, max_features=50000, max_length=96):
        self.max_features = max_features
        self.max_length = max_length
        self.words = ["<PAD>", "<UNK>"]
        self.indices = {word: i for i, word in enumerate(self.words)}

    def fit(self, texts):
        counts = Counter(word for text in texts for word in text.split())
        self.words = ["<PAD>", "<UNK>"] + [word for word, _ in
            sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            if word not in {"<PAD>", "<UNK>"}][:self.max_features - 2]
        self.indices = {word: i for i, word in enumerate(self.words)}
        return self

    def transform(self, texts):
        output = np.zeros((len(texts), self.max_length), dtype=np.int64)
        for row, text in enumerate(texts):
            ids = [self.indices.get(word, 1) for word in text.split()[:self.max_length]] or [1]
            output[row, :len(ids)] = ids
        return output

    def embedding_matrix(self, keyed_vectors, dim, seed):
        matrix = np.random.default_rng(seed).normal(0, 0.02, (len(self.words), dim)).astype(np.float32)
        for i, word in enumerate(self.words[2:], 2):
            if word in keyed_vectors:
                matrix[i] = keyed_vectors[word]
        matrix[0] = 0
        return matrix
