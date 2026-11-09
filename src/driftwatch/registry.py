"""A small file-based model registry with a gated promotion rule.

Layout::

    <root>/<name>/versions/<n>/{model.joblib, reference.csv, metadata.json}
    <root>/<name>/stages.json         {"production": <n>}
    <root>/<name>/decisions.jsonl     every promotion or rollback attempt, allowed or not
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from driftwatch.config import PromotionConfig
from driftwatch.train import METADATA_FILE

# Must start with a letter or digit so a name can never be "." or ".." and leave the registry root.
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class RegistryError(ValueError):
    """Invalid name, unknown version or malformed registry state."""


@dataclass(frozen=True)
class PromotionDecision:
    allowed: bool
    reason: str
    candidate: float | None
    baseline: float | None


def decide_promotion(
    policy: PromotionConfig,
    candidate_metrics: dict[str, float],
    baseline_metrics: dict[str, float] | None,
) -> PromotionDecision:
    """Promote only if ``candidate >= baseline + margin`` on the configured metric.

    With no production model the candidate only has to reach ``min_absolute``.
    """
    if policy.metric not in candidate_metrics:
        return PromotionDecision(False, f"candidate has no {policy.metric!r} metric", None, None)
    candidate = candidate_metrics[policy.metric]
    if baseline_metrics is None or policy.metric not in baseline_metrics:
        ok = candidate >= policy.min_absolute
        reason = (
            f"no production baseline; {policy.metric} {candidate:.4f} "
            f"{'meets' if ok else 'is below'} the minimum {policy.min_absolute:.4f}"
        )
        return PromotionDecision(ok, reason, candidate, None)
    baseline = baseline_metrics[policy.metric]
    required = baseline + policy.margin
    ok = candidate >= required
    reason = (
        f"{policy.metric} {candidate:.4f} {'meets' if ok else 'is below'} "
        f"baseline {baseline:.4f} + margin {policy.margin:.4f} = {required:.4f}"
    )
    return PromotionDecision(ok, reason, candidate, baseline)


class ModelRegistry:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _model_dir(self, name: str) -> Path:
        if not _NAME.match(name):
            raise RegistryError(f"invalid model name {name!r}")
        return self.root / name

    def versions(self, name: str) -> list[int]:
        base = self._model_dir(name) / "versions"
        if not base.exists():
            return []
        return sorted(int(p.name) for p in base.iterdir() if p.name.isdigit())

    def register(self, name: str, artifact_dir: Path | str) -> int:
        source = Path(artifact_dir)
        if not (source / METADATA_FILE).is_file():
            raise RegistryError(f"{source} has no {METADATA_FILE}")
        version = max(self.versions(name), default=0) + 1
        shutil.copytree(source, self._model_dir(name) / "versions" / str(version))
        return version

    def version_dir(self, name: str, version: int) -> Path:
        path = self._model_dir(name) / "versions" / str(version)
        if not path.is_dir():
            raise RegistryError(f"{name} has no version {version}")
        return path

    def metrics(self, name: str, version: int) -> dict[str, float]:
        raw = json.loads((self.version_dir(name, version) / METADATA_FILE).read_text("utf-8"))
        return {k: float(v) for k, v in raw.get("metrics", {}).items()}

    def stages(self, name: str) -> dict[str, int]:
        path = self._model_dir(name) / "stages.json"
        if not path.exists():
            return {}
        return {k: int(v) for k, v in json.loads(path.read_text("utf-8")).items()}

    def production_version(self, name: str) -> int | None:
        return self.stages(name).get("production")

    def production_dir(self, name: str) -> Path | None:
        version = self.production_version(name)
        return self.version_dir(name, version) if version is not None else None

    def promote(self, name: str, version: int, policy: PromotionConfig) -> PromotionDecision:
        """Apply the policy; write the stage file only when allowed. Always log the attempt."""
        current = self.production_version(name)
        if current == version:
            decision = PromotionDecision(False, "version is already in production", None, None)
        else:
            baseline = self.metrics(name, current) if current is not None else None
            decision = decide_promotion(policy, self.metrics(name, version), baseline)
        if decision.allowed:
            self._set_production(name, version)
        self._log(name, version, decision, action="promote")
        return decision

    def rollback(self, name: str, version: int, reason: str) -> PromotionDecision:
        """Point production at an existing version without applying the promotion gate.

        A rollback usually goes to an older, lower-scoring version, which the gate would refuse.
        A reason is required and the change is logged like any other decision.
        """
        if not reason.strip():
            raise RegistryError("a rollback needs a reason")
        self.version_dir(name, version)  # raises RegistryError for an unknown version
        if self.production_version(name) == version:
            decision = PromotionDecision(False, "version is already in production", None, None)
        else:
            decision = PromotionDecision(True, f"rollback: {reason.strip()}", None, None)
            self._set_production(name, version)
        self._log(name, version, decision, action="rollback")
        return decision

    def _set_production(self, name: str, version: int) -> None:
        (self._model_dir(name) / "stages.json").write_text(
            json.dumps({"production": version}), encoding="utf-8"
        )

    def _log(self, name: str, version: int, decision: PromotionDecision, action: str) -> None:
        entry: dict[str, Any] = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "action": action,
            "version": version,
            **asdict(decision),
        }
        with open(self._model_dir(name) / "decisions.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
