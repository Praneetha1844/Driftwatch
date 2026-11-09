"""Drift statistics: PSI and the two-sample Kolmogorov-Smirnov test, plus a combined report."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from driftwatch.config import DataSchema, DriftThresholds

EPSILON = 1e-4  # floor for bin proportions so the logarithm stays finite


class Status(StrEnum):
    OK = "ok"
    WARN = "warn"
    ALERT = "alert"
    INSUFFICIENT_DATA = "insufficient_data"


_ORDER = {Status.OK: 0, Status.WARN: 1, Status.ALERT: 2}


def _psi(reference_p: np.ndarray, current_p: np.ndarray) -> float:
    ref = np.clip(reference_p, EPSILON, None)
    cur = np.clip(current_p, EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_numeric(reference: Any, current: Any, bins: int = 10) -> float:
    """Population Stability Index with bin edges taken from the reference quantiles.

    The outer bins are open-ended, so values outside the reference range are counted rather
    than dropped. A constant reference feature gets a single split at that constant.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        raise ValueError("PSI needs at least one non-null value on each side")
    inner = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    width = inner.size + 1
    ref_counts = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=width)
    cur_counts = np.bincount(np.searchsorted(inner, cur, side="right"), minlength=width)
    return _psi(ref_counts / ref.size, cur_counts / cur.size)


def psi_categorical(reference: Any, current: Any) -> float:
    """PSI over category proportions; categories seen on only one side count as zero."""
    ref = pd.Series(reference).dropna().astype(str)
    cur = pd.Series(current).dropna().astype(str)
    if ref.empty or cur.empty:
        raise ValueError("PSI needs at least one non-null value on each side")
    categories = sorted(set(ref) | set(cur))
    ref_p = ref.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy()
    cur_p = cur.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy()
    return _psi(ref_p, cur_p)


def ks_test(reference: Any, current: Any) -> tuple[float, float]:
    """Return the KS statistic and p-value (two-sample, two-sided)."""
    result = stats.ks_2samp(np.asarray(reference, float), np.asarray(current, float))
    return float(result.statistic), float(result.pvalue)


def classify(psi: float, ks_pvalue: float | None, thresholds: DriftThresholds) -> Status:
    """ALERT needs a large PSI. A significant KS result alone only raises a WARN, because
    with large samples KS flags differences too small to matter."""
    if psi >= thresholds.psi_alert:
        return Status.ALERT
    if psi >= thresholds.psi_warn or (ks_pvalue is not None and ks_pvalue < thresholds.ks_alpha):
        return Status.WARN
    return Status.OK


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    kind: str
    psi: float
    ks_statistic: float | None
    ks_pvalue: float | None
    status: Status


@dataclass(frozen=True)
class DriftReport:
    n_reference: int
    n_current: int
    status: Status
    features: list[FeatureDrift]
    prediction: FeatureDrift | None = None
    label: FeatureDrift | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _numeric_drift(name: str, ref: Any, cur: Any, thresholds: DriftThresholds) -> FeatureDrift:
    ref_values = np.asarray(ref, dtype=float)
    cur_values = np.asarray(cur, dtype=float)
    psi = psi_numeric(ref_values, cur_values, thresholds.bins)
    statistic, pvalue = ks_test(
        ref_values[~np.isnan(ref_values)], cur_values[~np.isnan(cur_values)]
    )
    return FeatureDrift(name, "numeric", psi, statistic, pvalue, classify(psi, pvalue, thresholds))


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    schema: DataSchema,
    thresholds: DriftThresholds,
    reference_scores: np.ndarray | None = None,
    current_scores: np.ndarray | None = None,
) -> DriftReport:
    """Compare ``current`` with ``reference`` feature by feature.

    Fewer than ``min_samples`` current rows gives ``insufficient_data`` with no verdicts,
    because PSI on a handful of rows mostly measures sampling noise.
    """
    if len(current) < thresholds.min_samples:
        return DriftReport(len(reference), len(current), Status.INSUFFICIENT_DATA, [])

    features: list[FeatureDrift] = []
    for name in schema.numeric_features:
        features.append(_numeric_drift(name, reference[name], current[name], thresholds))
    for name in schema.categorical_features:
        psi = psi_categorical(reference[name], current[name])
        features.append(FeatureDrift(name, "categorical", psi, None, None,
                                     classify(psi, None, thresholds)))

    prediction = None
    if reference_scores is not None and current_scores is not None:
        prediction = _numeric_drift("prediction", reference_scores, current_scores, thresholds)

    label = None
    target = schema.target
    if target in reference.columns and target in current.columns:
        psi = psi_categorical(reference[target], current[target])
        label = FeatureDrift(target, "label", psi, None, None, classify(psi, None, thresholds))

    verdicts = [f.status for f in features] + [x.status for x in (prediction, label) if x]
    worst = max(verdicts, key=lambda s: _ORDER[s], default=Status.OK)
    return DriftReport(len(reference), len(current), worst, features, prediction, label)
