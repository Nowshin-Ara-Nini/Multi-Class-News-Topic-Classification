"""
Attention-based recurrent classifier for News Topic Classification.

Extends the recurrent encoder with an additive (Bahdanau-style) self-attention
mechanism that learns to weight each time step before feeding the context
vector into a classification head.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import register_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cell-type mapping
# ---------------------------------------------------------------------------

_RNN_CELLS = {
    "gru": nn.GRU,
    "lstm": nn.LSTM,
}


# ---------------------------------------------------------------------------
# Self-Attention Layer
# ---------------------------------------------------------------------------


class SelfAttention(nn.Module):
    """Additive (Bahdanau-style) self-attention over a sequence.

    Computes attention weights as::

        e_t   = tanh(W_a @ h_t + b_a)
        score = v_a^T @ e_t
        alpha = softmax(score)
        context = sum(alpha_t * h_t)

    Args:
        hidden_dim: Dimensionality of the input hidden states (per step).
        attention_dim: Dimensionality of the internal attention projection.

    Example:
        >>> attn = SelfAttention(hidden_dim=512, attention_dim=128)
        >>> h = torch.randn(32, 35, 512)
        >>> context, weights = attn(h)
        >>> context.shape, weights.shape
        (torch.Size([32, 512]), torch.Size([32, 35]))
    """

    def __init__(self, hidden_dim: int, attention_dim: int) -> None:
        super().__init__()

        self.W_a = nn.Linear(hidden_dim, attention_dim, bias=True)
        self.v_a = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted context vector.

        Args:
            hidden_states: Encoder outputs of shape
                ``(batch, seq_len, hidden_dim)``.

        Returns:
            Tuple of:
                - ``context`` — weighted sum of shape ``(batch, hidden_dim)``.
                - ``attention_weights`` — normalised weights of shape
                  ``(batch, seq_len)``.
        """
        # (batch, seq_len, attention_dim)
        energy = torch.tanh(self.W_a(hidden_states))

        # (batch, seq_len, 1) → (batch, seq_len)
        scores = self.v_a(energy).squeeze(-1)

        # Padding must receive no probability mass.  ``True`` marks padding,
        # matching PyTorch's Transformer convention.
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, float("-inf"))
            # An all-padding input is not useful, but keep softmax finite.
            all_padding = padding_mask.all(dim=1, keepdim=True)
            scores = torch.where(all_padding, torch.zeros_like(scores), scores)

        # Softmax over the sequence dimension
        attention_weights = F.softmax(scores, dim=-1)

        # Weighted sum: (batch, seq_len, 1) * (batch, seq_len, hidden_dim)
        context = torch.bmm(
            attention_weights.unsqueeze(1), hidden_states
        ).squeeze(1)

        return context, attention_weights


# ---------------------------------------------------------------------------
# Attention Classifier
# ---------------------------------------------------------------------------


@register_model('attention')
class AttentionClassifier(nn.Module):
    """Recurrent encoder with self-attention for News Topic Classification.

    Architecture:
        1. BiLSTM / BiGRU encoder
        2. Additive self-attention — learns per-timestep importance weights
        3. LayerNorm on the context vector
        4. Classifier head: Dropout → Linear → ReLU → Linear(num_classes)

    Weight initialisation:
        * **Recurrent layers** — orthogonal (hidden-to-hidden),
          Xavier (input-to-hidden).
        * **Linear layers** — Xavier uniform.

    Args:
        input_size: Embedding dimension (e.g. 300).
        hidden_size: Number of hidden units per direction.
        num_layers: Number of stacked recurrent layers.
        num_classes: Number of output classes.
        cell_type: Recurrent cell (``'lstm'`` or ``'gru'``).
        bidirectional: Use bidirectional encoding.
        dropout: Dropout probability.
        attention_dim: Internal dimension of the attention projection.

    Example:
        >>> model = AttentionClassifier(input_size=300, cell_type='lstm')
        >>> x = torch.randn(32, 35, 300)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([32, 4])

        >>> # Retrieve attention weights for interpretability
        >>> logits, weights = model(x, return_attention=True)
        >>> weights.shape
        torch.Size([32, 35])
    """

    def __init__(
        self,
        input_size: int = 300,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 4,
        cell_type: str = "lstm",
        bidirectional: bool = True,
        dropout: float = 0.3,
        attention_dim: int = 128,
    ) -> None:
        super().__init__()

        cell_type = cell_type.lower()
        if cell_type not in _RNN_CELLS:
            raise ValueError(
                f"Unsupported cell_type '{cell_type}'. "
                f"Choose from: {sorted(_RNN_CELLS)}"
            )

        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.num_classes: int = num_classes
        self.cell_type: str = cell_type
        self.bidirectional: bool = bidirectional
        self.dropout_rate: float = dropout
        self.attention_dim: int = attention_dim

        num_directions = 2 if bidirectional else 1
        encoder_output_dim = hidden_size * num_directions

        # ---- Recurrent encoder ----
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

        self.encoder_dropout = nn.Dropout(dropout)

        # ---- Self-attention ----
        self.attention = SelfAttention(
            hidden_dim=encoder_output_dim,
            attention_dim=attention_dim,
        )

        # ---- LayerNorm + Classifier head ----
        self.layer_norm = nn.LayerNorm(encoder_output_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

        # ---- Weight initialisation ----
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "AttentionClassifier created — cell=%s, hidden=%d, layers=%d, "
            "bidir=%s, attn_dim=%d, dropout=%.2f, params=%s",
            cell_type,
            hidden_size,
            num_layers,
            bidirectional,
            attention_dim,
            dropout,
            f"{total_params:,}",
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Apply orthogonal init to recurrent weights and Xavier to linear."""
        # Recurrent weights
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                if self.cell_type == "lstm":
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)

        # All linear layers (attention + classifier)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass producing raw logits (pre-softmax).

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, input_size)``.
            return_attention: If ``True``, also return the attention weights
                for interpretability / visualisation.

        Returns:
            If ``return_attention`` is ``False``:
                Logits tensor of shape ``(batch_size, num_classes)``.
            If ``return_attention`` is ``True``:
                Tuple of (logits, attention_weights) where
                ``attention_weights`` has shape ``(batch_size, seq_len)``.
        """
        # Encode
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
        rnn_output = self.encoder_dropout(rnn_output)

        # Attention
        context, attention_weights = self.attention(rnn_output, padding_mask)

        # Normalise + classify
        context = self.layer_norm(context)
        logits: torch.Tensor = self.classifier(context)

        if return_attention:
            return logits, attention_weights
        return logits

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_attention_weights(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract attention weights without computing gradients.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, input_size)``.

        Returns:
            Attention weights of shape ``(batch_size, seq_len)``.
        """
        self.eval()
        with torch.no_grad():
            _, weights = self.forward(x, padding_mask=padding_mask, return_attention=True)
        return weights

    def __repr__(self) -> str:
        num_dirs = 2 if self.bidirectional else 1
        return (
            f"AttentionClassifier(cell={self.cell_type}, "
            f"input={self.input_size}, hidden={self.hidden_size}, "
            f"layers={self.num_layers}, dirs={num_dirs}, "
            f"attn_dim={self.attention_dim}, "
            f"dropout={self.dropout_rate}, "
            f"num_classes={self.num_classes})"
        )
