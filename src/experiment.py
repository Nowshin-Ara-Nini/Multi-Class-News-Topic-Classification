"""Lightweight, local experiment tracking with reproducible JSON artifacts.

This intentionally avoids a hosted service.  Every training run gets one
JSON record containing its configuration, environment, epoch metrics, final
summary, and artifact paths under ``results/experiments``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from src.utils import capture_environment


class ExperimentTracker:
    """Persist one self-contained metadata record for each training run."""

    def __init__(self, results_dir: str | Path) -> None:
        self.directory = Path(results_dir) / "experiments"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._runs: Dict[str, Dict[str, Any]] = {}

    def start_run(self, name: str, config: Dict[str, Any]) -> str:
        run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
        self._runs[run_id] = {
            "run_id": run_id,
            "name": name,
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "environment": capture_environment(),
            "epochs": [],
        }
        self._write(run_id)
        return run_id

    def log_epoch(self, run_id: str, epoch: int, metrics: Dict[str, Any]) -> None:
        run = self._runs.get(run_id)
        if run is None:
            return
        run["epochs"].append({"epoch": int(epoch), **_json_safe(metrics)})
        self._write(run_id)

    def finish_run(
        self,
        run_id: str,
        summary: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, str]] = None,
        status: str = "completed",
    ) -> None:
        run = self._runs.get(run_id)
        if run is None:
            return
        run["status"] = status
        run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if summary:
            run["summary"] = _json_safe(summary)
        if artifacts:
            run["artifacts"] = artifacts
        self._write(run_id)

    def fail_run(self, run_id: str, error: Exception) -> None:
        self.finish_run(run_id, {"error": str(error)}, status="failed")

    def _write(self, run_id: str) -> None:
        path = self.directory / f"{run_id}.json"
        path.write_text(json.dumps(self._runs[run_id], indent=2, ensure_ascii=False), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    """Convert common NumPy/PyTorch scalar values to standard JSON types."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value
