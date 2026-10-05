"""Optional DistilBERT classifier baseline.

The Transformers dependency is imported only when the model is constructed,
so users who only run the classical/PyTorch models do not need it installed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models import register_model


@register_model("distilbert")
class DistilBERTClassifier(nn.Module):
    """DistilBERT encoder with a small trainable classification head."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 4,
        dropout: float = 0.3,
        freeze_base: bool = False,
        pooling: str = "cls",
        base_config: dict | None = None,
    ) -> None:
        super().__init__()
        try:
            from transformers import DistilBertModel, DistilBertConfig
        except ImportError as exc:
            raise ImportError(
                "DistilBERT requires `transformers`. Run `pip install -r requirements.txt`."
            ) from exc

        self.model_name = model_name
        try:
            self.distilbert = (DistilBertModel(DistilBertConfig.from_dict(base_config))
                               if base_config else DistilBertModel.from_pretrained(model_name))
        except Exception as exc:
            raise RuntimeError(
                f"Could not load DistilBERT model '{model_name}'. Connect to Hugging Face "
                "for the initial download or provide a local directory containing the "
                "complete pretrained model files."
            ) from exc
        if freeze_base:
            for parameter in self.distilbert.parameters():
                parameter.requires_grad = False
        if pooling not in {"cls", "cls_mean"}:
            raise ValueError("pooling must be cls or cls_mean")
        self.pooling = pooling
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.distilbert.config.hidden_size * (2 if pooling == "cls_mean" else 1), 256),
            nn.GELU() if pooling == "cls_mean" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = hidden[:, 0, :]
        if self.pooling == "cls_mean":
            mask = torch.ones_like(input_ids).unsqueeze(-1) if attention_mask is None else attention_mask.unsqueeze(-1)
            mean = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled = torch.cat([pooled, mean], dim=-1)
        return self.classifier(pooled)
