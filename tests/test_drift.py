import numpy as np
import pytest

from driftwatch.config import DriftThresholds
from driftwatch.drift import (
    Status,
    classify,
    detect_drift,
    ks_test,
    psi_categorical,
    psi_numeric,
)
from driftwatch.synthetic import make_dataset

THRESHOLDS = DriftThresholds()


def test_psi_is_zero_for_identical_samples() -> None:
    sample = np.linspace(0.0, 1.0, 1000)
    assert psi_numeric(sample, sample) < 1e-9


def test_psi_is_large_when_the_distribution_shifts() -> None:
    sample = np.linspace(0.0, 1.0, 1000)
    assert psi_numeric(sample, sample + 0.5) > 0.25


def test_psi_counts_values_outside_the_reference_range() -> None:
    reference = np.linspace(0.0, 1.0, 500)
    assert psi_numeric(reference, np.full(500, 100.0)) > 1.0


def test_constant_reference_feature_still_detects_change() -> None:
    assert psi_numeric(np.full(100, 5.0), np.full(100, 5.0)) == 0.0
    assert psi_numeric(np.full(100, 5.0), np.full(100, 3.0)) > 1.0


def test_psi_ignores_nan_and_rejects_empty_input() -> None:
    reference = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0])
    assert psi_numeric(reference, reference) < 1e-9
    with pytest.raises(ValueError):
        psi_numeric(np.array([np.nan]), np.array([1.0]))


def test_categorical_psi_detects_mix_changes_and_new_categories() -> None:
    reference = ["a"] * 60 + ["b"] * 40
    assert psi_categorical(reference, reference) < 1e-9
    assert psi_categorical(reference, ["a"] * 20 + ["b"] * 80) > 0.25
    assert psi_categorical(reference, ["a"] * 50 + ["b"] * 40 + ["z"] * 10) > 0.1


def test_ks_statistic_and_p_value() -> None:
    sample = np.linspace(0.0, 1.0, 400)
    statistic, p_value = ks_test(sample, sample)
    assert statistic == 0.0 and p_value == 1.0
    statistic, p_value = ks_test(sample, sample + 0.5)
    assert statistic > 0.4 and p_value < 0.01


@pytest.mark.parametrize(
    ("psi", "p_value", "expected"),
    [
        (0.30, 0.9, Status.ALERT),
        (0.15, 0.9, Status.WARN),
        (0.01, 0.01, Status.WARN),  # KS alone never raises an alert
        (0.01, 0.90, Status.OK),
        (0.01, None, Status.OK),
    ],
)
def test_classification_rules(psi: float, p_value: float | None, expected: Status) -> None:
    assert classify(psi, p_value, THRESHOLDS) is expected


def test_too_little_data_gives_no_verdict(settings) -> None:
    reference = make_dataset(500, seed=1)
    report = detect_drift(reference, make_dataset(10, seed=2), settings.data, settings.drift)
    assert report.status is Status.INSUFFICIENT_DATA and report.features == []


def test_same_process_does_not_alert(settings) -> None:
    reference, current = make_dataset(3000, seed=1), make_dataset(3000, seed=2)
    report = detect_drift(reference, current, settings.data, settings.drift)
    assert report.status is not Status.ALERT
    assert report.label is not None


def test_strong_drift_is_flagged_on_the_shifted_features(settings) -> None:
    reference = make_dataset(3000, seed=1)
    current = make_dataset(3000, seed=3, drift=1.0)
    report = detect_drift(reference, current, settings.data, settings.drift)
    by_name = {f.feature: f for f in report.features}
    assert report.status is Status.ALERT
    assert by_name["tenure_months"].status is Status.ALERT
    assert by_name["monthly_spend"].status is Status.ALERT


def test_prediction_drift_is_reported_when_scores_are_supplied(settings) -> None:
    rng = np.random.default_rng(0)
    reference, current = make_dataset(1000, seed=1), make_dataset(1000, seed=2)
    report = detect_drift(
        reference, current, settings.data, settings.drift,
        reference_scores=rng.beta(2, 5, 1000), current_scores=rng.beta(5, 2, 1000),
    )
    assert report.prediction is not None and report.prediction.status is Status.ALERT
    assert report.to_dict()["prediction"]["feature"] == "prediction"
