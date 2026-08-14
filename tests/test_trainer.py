from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from configs.config import TrainingConfig
from src.models.dnn import DNNClassifier
from src.trainer import Trainer


def test_two_epoch_training_loop_completes(cpu_device):
    torch.manual_seed(7)
    inputs = torch.randn(12, 5)
    labels = torch.tensor([0, 1, 2, 3] * 3)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=4, shuffle=False)
    config = TrainingConfig(
        epochs=2,
        batch_size=4,
        scheduler="none",
        mixed_precision=False,
        early_stopping=False,
        save_best=False,
        save_last=False,
    )
    trainer = Trainer(DNNClassifier(input_dim=5, hidden_layers=[8], dropout=0.0), config, cpu_device)
    history = trainer.train(loader, loader)

    assert all(len(values) == 2 for values in history.values())
    assert all(0.0 <= score <= 1.0 for score in history["val_f1"])
