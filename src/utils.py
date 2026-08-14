"""
Utility functions for the News Topic Classification project.

Provides reproducibility (seeding), device management, structured logging,
timing utilities, and memory tracking.
"""

from __future__ import annotations

import gc
import logging
import os
import random
import sys
import time
import platform
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def clear_invalid_local_proxy() -> list[str]:
    """Remove the known unusable localhost proxy injected by some shells.

    ``127.0.0.1:9`` has no listening proxy service, so HTTP clients fail with
    ``WinError 10061`` instead of connecting directly.  Only this exact,
    invalid value is removed; legitimate user proxy settings are preserved.

    Returns:
        Names of environment variables that were removed.
    """
    invalid_proxy = "http://127.0.0.1:9"
    proxy_variables = (
        "ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY",
        "all_proxy", "http_proxy", "https_proxy",
    )
    removed: list[str] = []
    for name in proxy_variables:
        value = os.environ.get(name, "").rstrip("/").lower()
        if value == invalid_proxy:
            os.environ.pop(name, None)
            removed.append(name)
    return removed


def configure_project_huggingface_cache() -> Path:
    """Use a writable, project-local cache for Hugging Face artifacts.

    Some Windows installations protect the default user cache directory.  A
    cache under the project keeps downloaded tokenizer/model files available
    across runs without requiring administrator permissions.
    """
    cache_dir = Path(__file__).resolve().parent.parent / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    return cache_dir

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for full reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    logging.getLogger(__name__).debug(f"All random seeds set to {seed}")


# ---------------------------------------------------------------------------
# Device Management
# ---------------------------------------------------------------------------

def get_device(preferred: Optional[str] = None) -> torch.device:
    """
    Auto-detect the best available compute device.

    Priority: preferred > CUDA > MPS (Apple Silicon) > CPU

    Args:
        preferred: Override device string (e.g., 'cuda', 'cpu', 'cuda:0').

    Returns:
        torch.device for computation.
    """
    logger = logging.getLogger(__name__)

    if preferred is not None:
        device = torch.device(preferred)
        logger.info(f"Using preferred device: {device}")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"Using CUDA device: {gpu_name} ({gpu_mem:.1f} GB)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    return device


def get_device_info() -> dict:
    """Get detailed device information for experiment logging."""
    info = {
        "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }

    if torch.cuda.is_available():
        info.update({
            "device": "cuda",
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "gpu_memory_gb": round(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
            ),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        })

    info["torch_version"] = torch.__version__
    info["numpy_version"] = np.__version__

    return info


def capture_environment() -> dict:
    """Return serialisable runtime information for experiment provenance.

    The result deliberately contains versions and hardware details only; it
    never captures environment variables, paths outside the project, or other
    potentially sensitive process state.
    """
    info = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    try:
        import sklearn
        info["scikit_learn_version"] = sklearn.__version__
    except ImportError:
        pass
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        pass
    info.update(get_device_info())
    return info


def get_model_size(path: str | Path) -> float:
    """Return the combined size of model files at *path* in MiB.

    ``path`` can be one file or a model directory.  Only model-weight files
    are counted, avoiding inflated numbers from cached preprocessors.
    """
    target = Path(path)
    if target.is_file():
        return target.stat().st_size / (1024 ** 2)
    if not target.exists():
        return 0.0
    suffixes = {".pt", ".pth", ".bin", ".safetensors", ".joblib", ".onnx"}
    return sum(
        file.stat().st_size for file in target.rglob("*")
        if file.is_file() and file.suffix.lower() in suffixes
    ) / (1024 ** 2)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
    name: Optional[str] = None,
) -> logging.Logger:
    """
    Configure structured logging with console and optional file output.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to write logs to file.
        name: Logger name. None returns the root logger.

    Returns:
        Configured logger instance.
    """
    # Windows PowerShell can default to cp1252, which cannot print report
    # symbols such as arrows.  Prefer UTF-8 while retaining a safe fallback
    # for hosts that do not expose ``reconfigure``.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, OSError):
        pass

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Timing Utilities
# ---------------------------------------------------------------------------

@contextmanager
def timer(description: str = "Operation", logger: Optional[logging.Logger] = None):
    """
    Context manager for timing code blocks.

    Usage:
        with timer("Training epoch"):
            train_one_epoch()

    Args:
        description: Human-readable description of the timed operation.
        logger: Logger to use. Falls back to print if None.
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start

    msg = f"{description} completed in {format_time(elapsed)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def format_time(seconds: float) -> str:
    """Format elapsed time into human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


# ---------------------------------------------------------------------------
# Memory Management
# ---------------------------------------------------------------------------

def get_gpu_memory_usage() -> dict:
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "max_allocated": 0}

    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / (1024 ** 2), 2),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
    }


def clear_gpu_memory() -> None:
    """Clear GPU memory cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


class MemoryTracker:
    """Track memory usage across training steps."""

    def __init__(self):
        self.snapshots = []

    def snapshot(self, label: str = "") -> dict:
        """Take a memory snapshot."""
        info = get_gpu_memory_usage()
        info["label"] = label
        info["timestamp"] = time.time()
        self.snapshots.append(info)
        return info

    def report(self) -> str:
        """Generate memory usage report."""
        if not self.snapshots:
            return "No memory snapshots recorded."

        lines = ["Memory Usage Report:", "-" * 50]
        for snap in self.snapshots:
            lines.append(
                f"  {snap['label']:30s} | "
                f"Alloc: {snap.get('allocated_mb', 0):8.1f} MB | "
                f"Reserved: {snap.get('reserved_mb', 0):8.1f} MB"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model Parameter Counting
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> dict:
    """
    Count model parameters (total and trainable).

    Args:
        model: PyTorch model.

    Returns:
        Dict with 'total', 'trainable', and 'non_trainable' parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
        "total_mb": round(total * 4 / (1024 ** 2), 2),  # Assuming float32
    }


def format_params(n: int) -> str:
    """Format parameter count with K/M/B suffixes."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)
