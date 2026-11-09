import json
from datetime import UTC, datetime, timedelta

import pytest

from driftwatch.config import PromotionConfig, TriggerConfig
from driftwatch.drift import DriftReport, FeatureDrift, Status
from driftwatch.policy import DecisionLog, decide_retraining
from driftwatch.registry import ModelRegistry, RegistryError, decide_promotion

POLICY = PromotionConfig(metric="f1", margin=0.01, min_absolute=0.5)


def test_first_model_only_needs_the_minimum() -> None:
    assert decide_promotion(POLICY, {"f1": 0.6}, None).allowed
    assert not decide_promotion(POLICY, {"f1": 0.4}, None).allowed


def test_candidate_must_beat_baseline_by_the_margin() -> None:
    assert decide_promotion(POLICY, {"f1": 0.72}, {"f1": 0.70}).allowed
    assert not decide_promotion(POLICY, {"f1": 0.705}, {"f1": 0.70}).allowed
    assert not decide_promotion(POLICY, {"f1": 0.70}, {"f1": 0.70}).allowed


def test_missing_metric_is_denied_with_a_reason() -> None:
    decision = decide_promotion(POLICY, {"accuracy": 0.9}, None)
    assert not decision.allowed and "f1" in decision.reason


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path / "registry")


def make_artifact(tmp_path, name: str, f1: float):
    path = tmp_path / name
    path.mkdir()
    (path / "metadata.json").write_text(json.dumps({"metrics": {"f1": f1}}), encoding="utf-8")
    return path


def test_versions_increase_and_invalid_input_is_rejected(registry, tmp_path) -> None:
    first = make_artifact(tmp_path, "a", 0.6)
    assert registry.register("demo", first) == 1
    assert registry.register("demo", first) == 2
    assert registry.versions("demo") == [1, 2]
    with pytest.raises(RegistryError):
        registry.register("../escape", first)
    with pytest.raises(RegistryError):
        registry.register("demo", tmp_path)  # no metadata.json
    with pytest.raises(RegistryError):
        registry.version_dir("demo", 9)


def test_promotion_flow_is_gated_and_logged(registry, tmp_path) -> None:
    for name, f1 in [("a", 0.70), ("b", 0.705), ("c", 0.75)]:
        registry.register("demo", make_artifact(tmp_path, name, f1))

    assert registry.promote("demo", 1, POLICY).allowed
    assert registry.production_version("demo") == 1

    denied = registry.promote("demo", 2, POLICY)  # +0.005 is below the 0.01 margin
    assert not denied.allowed and registry.production_version("demo") == 1

    assert registry.promote("demo", 3, POLICY).allowed
    assert registry.production_version("demo") == 3
    assert not registry.promote("demo", 3, POLICY).allowed  # already in production

    log = (tmp_path / "registry" / "demo" / "decisions.jsonl").read_text("utf-8").splitlines()
    assert [json.loads(line)["allowed"] for line in log] == [True, False, True, False]
    assert registry.production_dir("demo") == registry.version_dir("demo", 3)


# ---- retraining policy

NOW = datetime(2026, 6, 1, tzinfo=UTC)
TRIGGER = TriggerConfig(alert_features=1, warn_features=2, schedule_days=30)


def drift_report(*statuses: Status, prediction: Status | None = None, n: int = 500) -> DriftReport:
    features = [FeatureDrift(f"f{i}", "numeric", 0.0, 0.0, 1.0, s) for i, s in enumerate(statuses)]
    pred = FeatureDrift("prediction", "numeric", 0.0, 0.0, 1.0, prediction) if prediction else None
    overall = Status.INSUFFICIENT_DATA if n < 200 else Status.OK
    return DriftReport(1000, n, overall, features, pred)


def test_alerts_trigger_retraining() -> None:
    decision = decide_retraining(drift_report(Status.ALERT, Status.OK), TRIGGER, NOW)
    assert decision.retrain and "alert" in decision.reasons[0]


def test_single_warning_is_not_enough() -> None:
    decision = decide_retraining(drift_report(Status.WARN, Status.OK), TRIGGER, NOW)
    assert not decision.retrain and decision.reasons == []


def test_enough_warnings_trigger_retraining() -> None:
    assert decide_retraining(drift_report(Status.WARN, Status.WARN), TRIGGER, NOW).retrain


def test_prediction_alert_triggers_retraining() -> None:
    report = drift_report(Status.OK, prediction=Status.ALERT)
    assert decide_retraining(report, TRIGGER, NOW).retrain
    quiet = TriggerConfig(prediction_drift_alert=False)
    assert not decide_retraining(report, quiet, NOW).retrain


def test_schedule_applies_even_without_enough_data() -> None:
    thin = drift_report(n=10)
    fresh = decide_retraining(thin, TRIGGER, NOW - timedelta(days=5), NOW)
    assert not fresh.retrain and "not evaluated" in fresh.reasons[0]
    stale = decide_retraining(thin, TRIGGER, NOW - timedelta(days=45), NOW)
    assert stale.retrain and any("45 days old" in r for r in stale.reasons)


def test_naive_datetimes_are_rejected() -> None:
    with pytest.raises(ValueError):
        decide_retraining(drift_report(Status.OK), TRIGGER, datetime(2026, 1, 1))


def test_decision_log_appends_and_reads_back(tmp_path) -> None:
    log = DecisionLog(tmp_path / "logs" / "decisions.jsonl")
    assert log.read_all() == []
    decision = decide_retraining(drift_report(Status.ALERT), TRIGGER, NOW)
    log.append(decision, Status.ALERT)
    log.append(decision, Status.ALERT)
    entries = log.read_all()
    assert len(entries) == 2 and entries[0]["drift_status"] == "alert"
    assert entries[0]["retrain"] is True
