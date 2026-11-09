"""Typed settings loaded from YAML. Every threshold lives here, not in the code."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ColumnSpec(BaseModel):
    dtype: Literal["float", "int", "category"]
    nullable: bool = False
    min: float | None = None
    max: float | None = None


class DataSchema(BaseModel):
    target: str
    columns: dict[str, ColumnSpec]

    @model_validator(mode="after")
    def _target_is_declared(self) -> DataSchema:
        if self.target not in self.columns:
            raise ValueError(f"target column {self.target!r} must be declared in columns")
        return self

    @property
    def numeric_features(self) -> list[str]:
        return [c for c, s in self.columns.items() if s.dtype != "category" and c != self.target]

    @property
    def categorical_features(self) -> list[str]:
        return [c for c, s in self.columns.items() if s.dtype == "category" and c != self.target]

    @property
    def features(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]


class ModelConfig(BaseModel):
    type: Literal["logistic_regression", "random_forest"] = "logistic_regression"
    params: dict[str, Any] = Field(default_factory=dict)


class TrainConfig(BaseModel):
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    seed: int = 42
    reference_rows: int = Field(default=5000, ge=100)
    model: ModelConfig = Field(default_factory=ModelConfig)


class DriftThresholds(BaseModel):
    bins: int = Field(default=10, ge=2)
    psi_warn: float = Field(default=0.1, gt=0)
    psi_alert: float = Field(default=0.25, gt=0)
    ks_alpha: float = Field(default=0.05, gt=0, lt=1)
    min_samples: int = Field(default=200, ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> DriftThresholds:
        if self.psi_warn >= self.psi_alert:
            raise ValueError("psi_warn must be smaller than psi_alert")
        return self


class PromotionConfig(BaseModel):
    metric: Literal["accuracy", "f1", "roc_auc"] = "f1"
    margin: float = Field(default=0.01, ge=0.0)
    min_absolute: float = Field(default=0.0, ge=0.0, le=1.0)


class TriggerConfig(BaseModel):
    alert_features: int = Field(default=1, ge=1)
    warn_features: int = Field(default=3, ge=1)
    schedule_days: int = Field(default=30, ge=1)
    prediction_drift_alert: bool = True


class Settings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    data: DataSchema = Field(alias="schema")
    train: TrainConfig = Field(default_factory=TrainConfig)
    drift: DriftThresholds = Field(default_factory=DriftThresholds)
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    trigger: TriggerConfig = Field(default_factory=TriggerConfig)


def load_settings(path: Path | str) -> Settings:
    """Parse and validate a YAML settings file."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return Settings.model_validate(raw)
