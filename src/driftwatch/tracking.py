"""Experiment tracking behind one small interface.

``FileTracker`` needs nothing but the filesystem and is the default. ``MlflowTracker`` is used
when a tracking URI is configured and the optional ``mlflow`` extra is installed.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


class Tracker(Protocol):
    def log_run(
        self,
        name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        artifacts: Sequence[Path] = (),
    ) -> str:
        """Record one run and return its identifier."""
        ...


def flatten(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dictionaries into dotted keys: ``{"a": {"b": 1}}`` -> ``{"a.b": 1}``."""
    flat: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


class FileTracker:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def log_run(
        self,
        name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        artifacts: Sequence[Path] = (),
    ) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "-", name) or "run"
        run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{safe}"
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "params.json").write_text(
            json.dumps(flatten(params), indent=2, default=str), encoding="utf-8"
        )
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        if artifacts:
            (run_dir / "artifacts").mkdir()
            for artifact in artifacts:
                shutil.copy2(artifact, run_dir / "artifacts" / Path(artifact).name)
        return run_id


class MlflowTracker:
    def __init__(self, tracking_uri: str, experiment: str = "driftwatch"):
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError("install the 'mlflow' extra to use an MLflow tracking URI") from exc
        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

    def log_run(
        self,
        name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        artifacts: Sequence[Path] = (),
    ) -> str:
        with self._mlflow.start_run(run_name=name) as run:
            self._mlflow.log_params(flatten(params))
            self._mlflow.log_metrics(metrics)
            for artifact in artifacts:
                self._mlflow.log_artifact(str(artifact))
            return str(run.info.run_id)


def get_tracker(tracking_uri: str | None, fallback_dir: Path | str) -> Tracker:
    """MLflow when a URI is given, otherwise the file-based tracker."""
    if tracking_uri:
        return MlflowTracker(tracking_uri)
    return FileTracker(fallback_dir)
