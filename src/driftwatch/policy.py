"""Retraining trigger policy with an append-only decision log."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from driftwatch.config import TriggerConfig
from driftwatch.drift import DriftReport, Status


@dataclass(frozen=True)
class RetrainDecision:
    retrain: bool
    reasons: list[str]
    evaluated_at: str


def decide_retraining(
    report: DriftReport,
    config: TriggerConfig,
    last_trained_at: datetime,
    now: datetime | None = None,
) -> RetrainDecision:
    """Retrain when drift is strong enough, when prediction drift alerts, or on schedule.

    Drift signals are ignored when the report has too little data; the schedule still applies.
    All datetimes must be timezone-aware.
    """
    now = now or datetime.now(UTC)
    if last_trained_at.tzinfo is None or now.tzinfo is None:
        raise ValueError("datetimes must be timezone-aware")

    reasons: list[str] = []
    drift_reasons: list[str] = []
    if report.status is Status.INSUFFICIENT_DATA:
        reasons.append(f"drift not evaluated: only {report.n_current} current rows")
    else:
        alerts = [f.feature for f in report.features if f.status is Status.ALERT]
        warns = [f.feature for f in report.features if f.status is Status.WARN]
        if len(alerts) >= config.alert_features:
            drift_reasons.append(f"{len(alerts)} feature(s) in alert: {', '.join(alerts)}")
        if len(warns) >= config.warn_features:
            drift_reasons.append(f"{len(warns)} feature(s) in warn: {', '.join(warns)}")
        if (
            config.prediction_drift_alert
            and report.prediction is not None
            and report.prediction.status is Status.ALERT
        ):
            drift_reasons.append("prediction distribution is in alert")

    age_days = (now - last_trained_at).days
    if age_days >= config.schedule_days:
        drift_reasons.append(f"model is {age_days} days old (schedule: {config.schedule_days})")
    stamp = now.isoformat(timespec="seconds")
    return RetrainDecision(bool(drift_reasons), reasons + drift_reasons, stamp)


class DecisionLog:
    """JSON-lines log so every retraining decision can be audited later."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def append(self, decision: RetrainDecision, drift_status: Status) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {**asdict(decision), "drift_status": str(drift_status)}
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
