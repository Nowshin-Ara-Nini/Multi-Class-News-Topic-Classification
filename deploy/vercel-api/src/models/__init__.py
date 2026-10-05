"""
Model registry and factory for News Topic Classification.

Provides a decorator-based registration system and a factory function to
instantiate models by name.  All model submodules are imported at package
load time so that their ``@register_model`` decorators execute and populate
the global :data:`MODEL_REGISTRY`.

Usage:
    >>> from src.models import create_model, list_models
    >>> model = create_model('rnn', input_size=300, cell_type='lstm')
    >>> list_models()
    ['attention', 'dnn', 'rnn', 'transformer']
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Type

import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


# ---------------------------------------------------------------------------
# Registration decorator
# ---------------------------------------------------------------------------


def register_model(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """Decorator that registers a model class under a given name.

    Args:
        name: Identifier string used to look up the model later via
            :func:`create_model`.

    Returns:
        The original class, unmodified.

    Raises:
        ValueError: If ``name`` is already registered.

    Example:
        >>> @register_model('my_model')
        ... class MyModel(nn.Module):
        ...     ...
    """

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' is already registered by "
                f"{MODEL_REGISTRY[name].__name__}. "
                f"Cannot register {cls.__name__} under the same name."
            )
        MODEL_REGISTRY[name] = cls
        logger.debug("Registered model '%s' → %s", name, cls.__name__)
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a registered model by name.

    Args:
        name: Registered model identifier (e.g. ``'dnn'``, ``'rnn'``,
            ``'attention'``, ``'transformer'``).
        **kwargs: Keyword arguments forwarded to the model constructor.

    Returns:
        Instantiated ``nn.Module``.

    Raises:
        KeyError: If ``name`` is not found in the registry.

    Example:
        >>> model = create_model('dnn', input_dim=20000, num_classes=4)
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY)) or "(none)"
        raise KeyError(
            f"Model '{name}' is not registered. Available models: {available}"
        )

    model_cls = MODEL_REGISTRY[name]
    logger.info("Creating model '%s' (%s) with kwargs=%s", name, model_cls.__name__, kwargs)

    try:
        model = model_cls(**kwargs)
    except Exception:
        logger.exception("Failed to create model '%s'", name)
        raise

    return model


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def list_models() -> List[str]:
    """Return a sorted list of all registered model names.

    Returns:
        List of model identifier strings.
    """
    return sorted(MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Import submodules to trigger @register_model decorators
# ---------------------------------------------------------------------------
# NOTE: These imports MUST come after the registry and decorator definitions
# above, since the submodules use ``from src.models import register_model``.

from src.models.dnn import DNNClassifier  # noqa: E402, F401
from src.models.rnn import RecurrentClassifier  # noqa: E402, F401
from src.models.attention import AttentionClassifier  # noqa: E402, F401
from src.models.transformer import TransformerClassifier  # noqa: E402, F401
from src.models.distilbert import DistilBERTClassifier  # noqa: E402, F401
from src.models.embedded_sequence import EmbeddedSequenceClassifier  # noqa: E402, F401

# Classical models are not nn.Module and are accessed directly
from src.models.classical import ClassicalModelWrapper  # noqa: E402, F401

__all__ = [
    # Registry API
    "MODEL_REGISTRY",
    "register_model",
    "create_model",
    "list_models",
    # Model classes
    "DNNClassifier",
    "RecurrentClassifier",
    "AttentionClassifier",
    "TransformerClassifier",
    "DistilBERTClassifier",
    "ClassicalModelWrapper",
]
