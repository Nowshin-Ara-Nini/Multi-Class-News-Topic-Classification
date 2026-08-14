"""Shared, fast fixtures for the project's automated test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_texts() -> list[str]:
    return [
        "market shares rise today",
        "new satellite launches successfully",
        "team wins championship match",
        "leaders meet for peace talks",
    ] * 4


@pytest.fixture
def sample_labels() -> np.ndarray:
    return np.tile(np.arange(4, dtype=np.int64), 4)


@pytest.fixture
def cpu_device() -> torch.device:
    return torch.device("cpu")
