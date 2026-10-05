"""Versioned sequence wrapper, leaving legacy vector-input models loadable."""
import torch
from torch import nn
from src.models import register_model, create_model


@register_model("embedded_sequence")
class EmbeddedSequenceClassifier(nn.Module):
    def __init__(self, encoder_name, encoder_kwargs, vocab_size, embedding_dim=300,
                 trainable=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.requires_grad_(trainable)
        self.encoder = create_model(encoder_name, **encoder_kwargs)

    def forward(self, input_ids, padding_mask=None):
        ids = input_ids.long()
        # Empty inputs use UNK so attention never sees an entirely masked row.
        if ids.eq(0).all(dim=1).any():
            ids = ids.clone()
            ids[ids.eq(0).all(dim=1), 0] = 1
        return self.encoder(self.embedding(ids), padding_mask=ids.eq(0))
