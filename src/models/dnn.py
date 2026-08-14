"""
Feed-forward Deep Neural Network (DNN) classifier for News Topic Classification.

Designed to consume TF-IDF feature vectors and produce class logits through
a configurable stack of Linear → Activation → (BatchNorm) → Dropout layers.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn

from src.models import register_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Activation factory
# ---------------------------------------------------------------------------

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
    "tanh": nn.Tanh,
}


def _get_activation(name: str) -> nn.Module:
    """Return an activation module by name.

    Args:
        name: Activation function identifier (case-insensitive).

    Returns:
        Instantiated activation module.

    Raises:
        ValueError: If the activation name is not supported.
    """
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation '{name}'. "
            f"Choose from: {sorted(_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[key]()


# ---------------------------------------------------------------------------
# DNN Classifier
# ---------------------------------------------------------------------------


@register_model('dnn')
class DNNClassifier(nn.Module):
    """Configurable feed-forward deep neural network classifier.

    Architecture (default):
        Linear(input_dim, 256) → ReLU → Dropout(0.4)
        → Linear(256, 128)    → ReLU → Dropout(0.4)
        → Linear(128, 64)     → ReLU → Dropout(0.4)
        → Linear(64, num_classes)

    All linear layers use Xavier uniform initialisation.  Optional
    ``BatchNorm1d`` can be inserted before each activation.

    Args:
        input_dim: Dimensionality of input feature vector (e.g. TF-IDF).
        num_classes: Number of output classes.
        hidden_layers: List of hidden-layer widths.
        dropout: Dropout probability applied after each activation.
        activation: Activation function name (``'relu'``, ``'gelu'``, etc.).
        batch_norm: If ``True``, insert BatchNorm1d before each activation.

    Example:
        >>> model = DNNClassifier(input_dim=20000, num_classes=4)
        >>> logits = model(torch.randn(32, 20000))
        >>> logits.shape
        torch.Size([32, 4])
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        hidden_layers: Optional[List[int]] = None,
        dropout: float = 0.4,
        activation: str = "relu",
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [256, 128, 64]

        self.input_dim: int = input_dim
        self.num_classes: int = num_classes
        self.hidden_layers: List[int] = hidden_layers
        self.dropout_rate: float = dropout
        self.activation_name: str = activation
        self.use_batch_norm: bool = batch_norm

        # ---- Build sequential layers ----
        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(_get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final classification head (no activation / dropout after)
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

        # ---- Weight initialisation ----
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "DNNClassifier created — layers=%s, dropout=%.2f, activation=%s, "
            "batch_norm=%s, params=%s",
            hidden_layers,
            dropout,
            activation,
            batch_norm,
            f"{total_params:,}",
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Apply Xavier uniform initialisation to all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing raw logits (pre-softmax).

        Args:
            x: Input tensor of shape ``(batch_size, input_dim)``.

        Returns:
            Logits tensor of shape ``(batch_size, num_classes)``.
        """
        return self.network(x)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DNNClassifier(input_dim={self.input_dim}, "
            f"num_classes={self.num_classes}, "
            f"hidden_layers={self.hidden_layers}, "
            f"dropout={self.dropout_rate}, "
            f"activation='{self.activation_name}', "
            f"batch_norm={self.use_batch_norm})"
        )
