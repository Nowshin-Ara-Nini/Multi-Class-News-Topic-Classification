"""
Transformer Encoder classifier for News Topic Classification.

Uses PyTorch's native ``TransformerEncoder`` with sinusoidal positional
encoding and a configurable pooling + classification head.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn

from src.models import register_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Adds position-dependent sinusoidal signals to the input embeddings so
    the Transformer can attend to positional information.

    Args:
        d_model: Embedding / model dimensionality.
        max_len: Maximum sequence length to pre-compute encodings for.
        dropout: Dropout applied after adding positional encoding.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute the positional encoding table
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, max_len, d_model) — broadcastable over the batch dimension
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the input.

        Args:
            x: Input embeddings of shape ``(batch, seq_len, d_model)``.

        Returns:
            Position-encoded tensor of the same shape.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Classifier
# ---------------------------------------------------------------------------


@register_model('transformer')
class TransformerClassifier(nn.Module):
    """Transformer Encoder classifier for text classification.

    Architecture:
        1. Optional CLS token prepended to the sequence
        2. Sinusoidal positional encoding
        3. Input projection (if needed) to ``d_model``
        4. ``TransformerEncoder`` layers
        5. Pooling (CLS token / mean / max)
        6. Classifier head: LayerNorm → Dropout → Linear → ReLU → Linear

    Args:
        d_model: Model / embedding dimensionality. Must be divisible
            by ``nhead``.
        nhead: Number of attention heads.
        num_encoder_layers: Number of ``TransformerEncoderLayer`` blocks.
        dim_feedforward: Hidden size of the feed-forward sub-layers.
        num_classes: Number of output classes.
        max_seq_len: Maximum sequence length (used for positional encoding).
        dropout: Dropout probability throughout the model.
        pooling: Sequence pooling strategy (``'cls'``, ``'mean'``, ``'max'``).
        input_dim: If not equal to ``d_model``, a linear projection maps
            input features to ``d_model``.

    Example:
        >>> model = TransformerClassifier(d_model=300, nhead=6)
        >>> x = torch.randn(32, 35, 300)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([32, 4])
    """

    _VALID_POOLING = {"cls", "mean", "max"}

    def __init__(
        self,
        d_model: int = 300,
        nhead: int = 6,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        num_classes: int = 4,
        max_seq_len: int = 35,
        dropout: float = 0.3,
        pooling: str = "cls",
        input_dim: Optional[int] = None,
        normalize_input: bool = False,
    ) -> None:
        super().__init__()

        pooling = pooling.lower()
        if pooling not in self._VALID_POOLING:
            raise ValueError(
                f"Unsupported pooling '{pooling}'. "
                f"Choose from: {sorted(self._VALID_POOLING)}"
            )
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )

        self.d_model: int = d_model
        self.nhead: int = nhead
        self.num_encoder_layers: int = num_encoder_layers
        self.dim_feedforward: int = dim_feedforward
        self.num_classes: int = num_classes
        self.max_seq_len: int = max_seq_len
        self.dropout_rate: float = dropout
        self.pooling: str = pooling
        self.normalize_input = normalize_input
        self.input_norm = nn.LayerNorm(d_model) if normalize_input else nn.Identity()

        # ---- Optional input projection ----
        self.input_proj: Optional[nn.Linear] = None
        if input_dim is not None and input_dim != d_model:
            self.input_proj = nn.Linear(input_dim, d_model)

        # ---- CLS token (learnable) ----
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ---- Positional encoding ----
        # +1 for possible CLS token
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len + 1,
            dropout=dropout,
        )

        # ---- Transformer encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False,
        )

        # ---- Classifier head ----
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

        # ---- Weight initialisation ----
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "TransformerClassifier created — d_model=%d, nhead=%d, "
            "layers=%d, ff=%d, pooling=%s, dropout=%.2f, params=%s",
            d_model,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            pooling,
            dropout,
            f"{total_params:,}",
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Apply Xavier uniform initialisation to all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass producing raw logits (pre-softmax).

        Args:
            x: Input embeddings of shape ``(batch, seq_len, input_dim)``
                or ``(batch, seq_len, d_model)``.
            padding_mask: Optional boolean mask of shape
                ``(batch, seq_len)`` where ``True`` indicates padding
                positions to be ignored.
            src_key_padding_mask: Backwards-compatible alias for
                ``padding_mask``.

        Returns:
            Logits tensor of shape ``(batch_size, num_classes)``.
        """
        if padding_mask is not None and src_key_padding_mask is not None:
            raise ValueError("Pass only one of padding_mask or src_key_padding_mask")
        src_key_padding_mask = padding_mask if padding_mask is not None else src_key_padding_mask

        # Optional input projection
        if self.input_proj is not None:
            x = self.input_proj(x)

        batch_size = x.size(0)

        # Prepend CLS token if using CLS pooling
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

            # Extend padding mask for CLS token (never masked)
            if src_key_padding_mask is not None:
                cls_mask = torch.zeros(
                    batch_size, 1,
                    dtype=torch.bool,
                    device=src_key_padding_mask.device,
                )
                src_key_padding_mask = torch.cat(
                    [cls_mask, src_key_padding_mask], dim=1
                )

        # Scale embeddings (common Transformer practice)
        x = self.input_norm(x) if self.normalize_input else x * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        encoded = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )

        # Pooling
        pooled = self._pool_output(encoded, src_key_padding_mask)

        # Classification head
        logits: torch.Tensor = self.classifier(pooled)
        return logits

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------

    def _pool_output(
        self,
        encoded: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool the encoder output across the sequence dimension.

        Args:
            encoded: Encoder output of shape ``(batch, seq_len, d_model)``.
            mask: Optional padding mask of shape ``(batch, seq_len)``.

        Returns:
            Pooled tensor of shape ``(batch, d_model)``.
        """
        if self.pooling == "cls":
            # CLS token is at position 0
            return encoded[:, 0, :]

        elif self.pooling == "mean":
            if mask is not None:
                # Zero-out padding positions before averaging
                # mask: True = padded → invert for weighting
                weight = (~mask).unsqueeze(-1).float()
                summed = (encoded * weight).sum(dim=1)
                lengths = weight.sum(dim=1).clamp(min=1.0)
                return summed / lengths
            return encoded.mean(dim=1)

        else:  # max
            if mask is not None:
                encoded = encoded.masked_fill(
                    mask.unsqueeze(-1), float("-inf")
                )
            pooled = encoded.max(dim=1).values
            return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TransformerClassifier(d_model={self.d_model}, "
            f"nhead={self.nhead}, "
            f"layers={self.num_encoder_layers}, "
            f"ff={self.dim_feedforward}, "
            f"pooling='{self.pooling}', "
            f"dropout={self.dropout_rate}, "
            f"num_classes={self.num_classes})"
        )
