"""
Unified Recurrent Neural Network classifier for News Topic Classification.

Replaces six separate model classes (RNN, GRU, LSTM, BiRNN, BiGRU, BiLSTM)
with a single parameterised ``RecurrentClassifier`` that supports all
combinations of cell type, directionality, pooling strategy, and layer
normalisation.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

from src.models import register_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cell-type mapping
# ---------------------------------------------------------------------------

_RNN_CELLS = {
    "rnn": nn.RNN,
    "gru": nn.GRU,
    "lstm": nn.LSTM,
}


# ---------------------------------------------------------------------------
# Recurrent Classifier
# ---------------------------------------------------------------------------


@register_model('rnn')
class RecurrentClassifier(nn.Module):
    """Unified recurrent classifier (RNN / GRU / LSTM, uni- or bi-directional).

    Architecture:
        1. Recurrent encoder (parameterised cell type)
        2. Pooling over the temporal dimension (``last`` / ``mean`` /
           ``max`` / ``mean_max``)
        3. Optional LayerNorm on the pooled representation
        4. Classifier head:
           ``Linear → ReLU → Dropout → Linear → ReLU → Dropout
           → Linear → ReLU → Linear(num_classes)``

    Weight initialisation:
        * **Recurrent layers** — orthogonal initialisation.
        * **Linear layers** — Xavier uniform initialisation.

    Note:
        When ``num_layers == 1``, the ``dropout`` parameter is **not** passed
        to the PyTorch RNN constructor (it only applies *between* stacked
        layers). A separate ``nn.Dropout`` is used after the encoder instead.

    Args:
        input_size: Embedding dimension (e.g. 300 for Word2Vec).
        hidden_size: Number of hidden units per direction.
        num_layers: Number of stacked recurrent layers.
        num_classes: Number of output classes.
        cell_type: Recurrent cell variant (``'rnn'``, ``'gru'``, ``'lstm'``).
        bidirectional: Use bidirectional encoding.
        dropout: Dropout probability.
        pooling: Temporal pooling strategy.
        use_layer_norm: Apply LayerNorm on the pooled output.

    Example:
        >>> model = RecurrentClassifier(input_size=300, cell_type='lstm',
        ...                             bidirectional=True, pooling='mean_max')
        >>> x = torch.randn(32, 35, 300)  # (batch, seq_len, embed_dim)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([32, 4])
    """

    _VALID_POOLING = {"last", "mean", "max", "mean_max"}

    def __init__(
        self,
        input_size: int = 300,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 4,
        cell_type: str = "lstm",
        bidirectional: bool = True,
        dropout: float = 0.3,
        pooling: str = "mean_max",
        use_layer_norm: bool = True,
        compact_head: bool = False,
    ) -> None:
        super().__init__()

        # ---- Validate arguments ----
        cell_type = cell_type.lower()
        pooling = pooling.lower()

        if cell_type not in _RNN_CELLS:
            raise ValueError(
                f"Unsupported cell_type '{cell_type}'. "
                f"Choose from: {sorted(_RNN_CELLS)}"
            )
        if pooling not in self._VALID_POOLING:
            raise ValueError(
                f"Unsupported pooling '{pooling}'. "
                f"Choose from: {sorted(self._VALID_POOLING)}"
            )

        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.num_classes: int = num_classes
        self.cell_type: str = cell_type
        self.bidirectional: bool = bidirectional
        self.dropout_rate: float = dropout
        self.pooling: str = pooling
        self.use_layer_norm: bool = use_layer_norm

        num_directions = 2 if bidirectional else 1

        # ---- Recurrent encoder ----
        # PyTorch warns when dropout > 0 and num_layers == 1 because dropout
        # is only applied *between* stacked layers. Avoid the warning.
        rnn_dropout = dropout if num_layers > 1 else 0.0

        rnn_cls = _RNN_CELLS[cell_type]
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=rnn_dropout,
        )

        # Separate dropout applied after the encoder when num_layers == 1
        self.encoder_dropout = nn.Dropout(dropout)

        # ---- Pooling dimension ----
        rnn_out_dim = hidden_size * num_directions
        if pooling == "mean_max":
            pooled_dim = rnn_out_dim * 2  # mean + max concatenated
        else:
            pooled_dim = rnn_out_dim

        # ---- Optional LayerNorm ----
        self.layer_norm: Optional[nn.LayerNorm] = None
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(pooled_dim)

        # ---- Classifier head ----
        # Mirrors the notebook's OptimizedBiRNN classifier architecture:
        # (pooled_dim, 256) → (256, 128) → (128, 64) → (64, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

        if compact_head:
            self.classifier = nn.Sequential(nn.Linear(pooled_dim, 128), nn.GELU(),
                                            nn.Dropout(dropout), nn.Linear(128, num_classes))

        # ---- Weight initialisation ----
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "RecurrentClassifier created — cell=%s, hidden=%d, layers=%d, "
            "bidir=%s, pooling=%s, layer_norm=%s, dropout=%.2f, params=%s",
            cell_type,
            hidden_size,
            num_layers,
            bidirectional,
            pooling,
            use_layer_norm,
            dropout,
            f"{total_params:,}",
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Apply orthogonal init to recurrent weights and Xavier to linear."""
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                # Set forget-gate bias to 1 for LSTM (improved gradient flow)
                if self.cell_type == "lstm":
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Pooling strategies
    # ------------------------------------------------------------------

    def _pool_output(
        self, rnn_output: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pool the recurrent output across the temporal dimension.

        Args:
            rnn_output: Tensor of shape ``(batch, seq_len, hidden * dirs)``.

        Returns:
            Pooled tensor of shape ``(batch, pooled_dim)``.
        """
        if padding_mask is None:
            if self.pooling == "last":
                return rnn_output[:, -1, :]
            if self.pooling == "mean":
                return rnn_output.mean(dim=1)
            if self.pooling == "max":
                return rnn_output.max(dim=1).values
            return torch.cat([rnn_output.mean(dim=1), rnn_output.max(dim=1).values], dim=-1)

        valid = (~padding_mask).unsqueeze(-1)
        lengths = valid.squeeze(-1).sum(dim=1).clamp_min(1)
        if self.pooling == "last":
            return rnn_output[torch.arange(rnn_output.size(0), device=rnn_output.device), lengths - 1]

        mean_pool = (rnn_output * valid).sum(dim=1) / lengths.unsqueeze(-1)
        if self.pooling == "mean":
            return mean_pool
        max_pool = rnn_output.masked_fill(~valid, float("-inf")).max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        return max_pool if self.pooling == "max" else torch.cat([mean_pool, max_pool], dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass producing raw logits (pre-softmax).

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, input_size)``.

        Returns:
            Logits tensor of shape ``(batch_size, num_classes)``.
        """
        # Packed sequences prevent padding vectors from changing recurrent states.
        if padding_mask is not None:
            lengths = (~padding_mask).sum(dim=1).clamp_min(1).to("cpu")
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.rnn(packed)
            rnn_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=x.size(1)
            )
        else:
            rnn_output, _ = self.rnn(x)

        # Apply dropout after encoder (always, but especially important
        # when num_layers == 1 since PyTorch's inter-layer dropout is unused)
        rnn_output = self.encoder_dropout(rnn_output)

        # Temporal pooling
        pooled = self._pool_output(rnn_output, padding_mask)

        # Optional layer normalisation
        if self.layer_norm is not None:
            pooled = self.layer_norm(pooled)

        # Classification head
        logits: torch.Tensor = self.classifier(pooled)
        return logits

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        num_dirs = 2 if self.bidirectional else 1
        return (
            f"RecurrentClassifier(cell={self.cell_type}, "
            f"input={self.input_size}, hidden={self.hidden_size}, "
            f"layers={self.num_layers}, dirs={num_dirs}, "
            f"pooling='{self.pooling}', "
            f"layer_norm={self.use_layer_norm}, "
            f"dropout={self.dropout_rate}, "
            f"num_classes={self.num_classes})"
        )
