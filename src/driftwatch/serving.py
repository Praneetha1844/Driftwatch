"""FastAPI serving layer with Prometheus metrics and on-demand drift checks."""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Any, Protocol

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from driftwatch import __version__
from driftwatch.config import Settings, load_settings
from driftwatch.drift import detect_drift
from driftwatch.registry import ModelRegistry
from driftwatch.train import load_artifact


class ServingModel(Protocol):
    version: str
    feature_names: list[str]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the positive-class probability for each row."""
        ...


class PipelineModel:
    """Adapts a fitted scikit-learn pipeline to the ``ServingModel`` interface."""

    def __init__(self, pipeline: Any, version: str, feature_names: list[str]):
        self.pipeline, self.version, self.feature_names = pipeline, version, feature_names

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        scores = self.pipeline.predict_proba(frame[self.feature_names])[:, 1]
        return np.asarray(scores, dtype=float)


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    probability: float
    label: int


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


def create_app(
    model: ServingModel,
    reference: pd.DataFrame | None = None,
    settings: Settings | None = None,
    buffer_size: int = 5000,
    threshold: float = 0.5,
) -> FastAPI:
    """Build the API around one model. Each app owns its metrics registry, so several apps
    (for example in tests) can coexist in one process."""
    metrics = CollectorRegistry()
    scored = Counter(
        "driftwatch_predictions_total", "Rows scored", ["model_version"], registry=metrics
    )
    rejected = Counter("driftwatch_rejected_requests_total", "Rejected requests", registry=metrics)
    latency = Histogram("driftwatch_predict_seconds", "Latency of /predict", registry=metrics)
    info = Gauge(
        "driftwatch_model_info", "Serving model version", ["model_version"], registry=metrics
    )
    feature_psi = Gauge(
        "driftwatch_feature_psi", "Latest PSI per feature", ["feature"], registry=metrics
    )
    info.labels(model.version).set(1)

    lock = threading.Lock()
    rows: deque[dict[str, Any]] = deque(maxlen=buffer_size)
    scores: deque[float] = deque(maxlen=buffer_size)

    app = FastAPI(title="Driftwatch", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model_version": model.version}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        started = time.perf_counter()
        frame = pd.DataFrame(request.records)
        missing = [c for c in model.feature_names if c not in frame or frame[c].isna().any()]
        if missing:
            rejected.inc()
            raise HTTPException(status_code=422, detail={"missing_or_null_features": missing})
        try:
            probabilities = model.predict_proba(frame)
        except (ValueError, TypeError) as exc:
            rejected.inc()
            raise HTTPException(status_code=422, detail=f"could not score records: {exc}") from exc
        with lock:
            rows.extend(frame[model.feature_names].to_dict("records"))
            scores.extend(float(p) for p in probabilities)
        scored.labels(model.version).inc(len(probabilities))
        latency.observe(time.perf_counter() - started)
        return PredictResponse(
            model_version=model.version,
            predictions=[
                Prediction(probability=float(p), label=int(p >= threshold)) for p in probabilities
            ],
        )

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics), media_type=CONTENT_TYPE_LATEST)

    @app.get("/drift")
    def drift() -> dict[str, Any]:
        if reference is None or settings is None:
            raise HTTPException(status_code=503, detail="drift monitoring is not configured")
        with lock:
            current = pd.DataFrame(list(rows))
            current_scores = np.asarray(scores, dtype=float)
        if len(current) < settings.drift.min_samples:
            return {
                "status": "insufficient_data",
                "n_current": len(current),
                "min_samples": settings.drift.min_samples,
            }
        report = detect_drift(
            reference,
            current,
            settings.data,
            settings.drift,
            reference_scores=model.predict_proba(reference),
            current_scores=current_scores,
        )
        for item in report.features:
            feature_psi.labels(item.feature).set(item.psi)
        return report.to_dict()

    return app


def app_from_env(
    name: str | None = None, registry_dir: str | None = None, config: str | None = None
) -> FastAPI:
    """Serve the production version from the registry.

    Explicit arguments win, then the ``DRIFTWATCH_MODEL_NAME``, ``DRIFTWATCH_REGISTRY_DIR``
    and ``DRIFTWATCH_CONFIG`` environment variables, then the defaults. Also usable as a
    factory for ``uvicorn --factory``.
    """
    name = name or os.environ.get("DRIFTWATCH_MODEL_NAME", "churn-demo")
    registry = ModelRegistry(registry_dir or os.environ.get("DRIFTWATCH_REGISTRY_DIR", "registry"))
    version = registry.production_version(name)
    if version is None:
        raise RuntimeError(
            f"model {name!r} has no production version; train, register and promote one first"
        )
    artifact = load_artifact(registry.version_dir(name, version))
    settings = load_settings(config or os.environ.get("DRIFTWATCH_CONFIG", "config/default.yaml"))
    model = PipelineModel(artifact.pipeline, str(version), list(artifact.metadata["features"]))
    return create_app(model, artifact.reference, settings)
