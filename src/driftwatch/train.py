"""Config-driven training for a binary tabular classifier, and the artifact format."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from driftwatch.config import ModelConfig, Settings
from driftwatch.data import require_valid

MODEL_FILE = "model.joblib"
REFERENCE_FILE = "reference.csv"
METADATA_FILE = "metadata.json"


@dataclass
class TrainResult:
    pipeline: Pipeline
    metrics: dict[str, float]
    reference: pd.DataFrame  # training rows used as the drift baseline


@dataclass
class LoadedArtifact:
    pipeline: Pipeline
    reference: pd.DataFrame
    metadata: dict[str, Any]


def _estimator(config: ModelConfig, seed: int) -> Any:
    if config.type == "logistic_regression":
        return LogisticRegression(**{"max_iter": 1000, **config.params})
    return RandomForestClassifier(**{"random_state": seed, **config.params})


def build_pipeline(settings: Settings) -> Pipeline:
    schema = settings.data
    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler(), schema.numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), schema.categorical_features),
        ]
    )
    model = _estimator(settings.train.model, settings.train.seed)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def train(frame: pd.DataFrame, settings: Settings) -> TrainResult:
    """Validate, split (stratified), fit and score on the held-out split."""
    schema = settings.data
    require_valid(frame, schema)
    if set(frame[schema.target].unique()) != {0, 1}:
        raise ValueError("the target must contain both classes 0 and 1")

    features, target = frame[schema.features], frame[schema.target].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=settings.train.test_size,
        random_state=settings.train.seed,
        stratify=target,
    )
    pipeline = build_pipeline(settings).fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    reference = x_train.assign(**{schema.target: y_train})
    if len(reference) > settings.train.reference_rows:
        reference = reference.sample(
            settings.train.reference_rows, random_state=settings.train.seed
        )
    return TrainResult(pipeline, metrics, reference.reset_index(drop=True))


def save_artifact(result: TrainResult, directory: Path | str, settings: Settings) -> Path:
    """Write model, reference sample and metadata into ``directory``."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.pipeline, out / MODEL_FILE)
    result.reference.to_csv(out / REFERENCE_FILE, index=False)
    metadata = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "metrics": result.metrics,
        "features": settings.data.features,
        "target": settings.data.target,
        "model_type": settings.train.model.type,
        "seed": settings.train.seed,
    }
    (out / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out


def load_artifact(directory: Path | str) -> LoadedArtifact:
    """Load an artifact. Only load artifacts you produced: joblib uses pickle."""
    root = Path(directory)
    try:
        metadata = json.loads((root / METADATA_FILE).read_text(encoding="utf-8"))
        return LoadedArtifact(
            joblib.load(root / MODEL_FILE), pd.read_csv(root / REFERENCE_FILE), metadata
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{root} is not a valid model artifact: {exc}") from exc
