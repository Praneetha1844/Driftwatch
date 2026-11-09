"""Command line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from driftwatch.config import Settings, load_settings
from driftwatch.data import DataValidationError, load_csv, require_valid
from driftwatch.drift import DriftReport, detect_drift
from driftwatch.policy import DecisionLog, decide_retraining
from driftwatch.registry import ModelRegistry, RegistryError
from driftwatch.synthetic import make_dataset
from driftwatch.tracking import get_tracker
from driftwatch.train import METADATA_FILE, LoadedArtifact, load_artifact, save_artifact, train

DEFAULT_CONFIG = "config/default.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="driftwatch", description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="settings YAML")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="write a seeded synthetic dataset")
    sim.add_argument("--output", type=Path, required=True)
    sim.add_argument("--n", type=int, default=5000)
    sim.add_argument("--seed", type=int, default=0)
    sim.add_argument("--drift", type=float, default=0.0)
    sim.add_argument("--label-shift", type=float, default=0.0)

    tr = sub.add_parser("train", help="train, save an artifact and log the run")
    tr.add_argument("--data", type=Path, required=True)
    tr.add_argument("--artifact-dir", type=Path, required=True)
    tr.add_argument("--runs-dir", type=Path, default=Path("runs"))
    tr.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))

    reg = sub.add_parser("register", help="copy an artifact into the registry as a new version")
    reg.add_argument("--name", required=True)
    reg.add_argument("--artifact-dir", type=Path, required=True)
    reg.add_argument("--registry", type=Path, default=Path("registry"))

    pro = sub.add_parser("promote", help="promote a version to production if the policy allows")
    pro.add_argument("--name", required=True)
    pro.add_argument("--version", type=int, required=True)
    pro.add_argument("--registry", type=Path, default=Path("registry"))

    for name, help_text in (("drift", "report drift"), ("decide", "report drift and decide")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--artifact-dir", type=Path, required=True)
        p.add_argument("--current", type=Path, required=True)
        p.add_argument("--output", type=Path)
        if name == "decide":
            p.add_argument("--decision-log", type=Path, default=Path("decisions.jsonl"))
            p.add_argument("--fail-on-retrain", action="store_true")

    serve = sub.add_parser("serve", help="run the API for the production model")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def _drift(
    settings: Settings, artifact_dir: Path, current_csv: Path
) -> tuple[DriftReport, LoadedArtifact]:
    artifact = load_artifact(artifact_dir)
    current = load_csv(current_csv)
    features = settings.data.features
    missing = [c for c in features if c not in current.columns]
    if missing:
        raise DataValidationError(f"current data is missing columns: {missing}")
    reference_scores = artifact.pipeline.predict_proba(artifact.reference[features])[:, 1]
    current_scores = artifact.pipeline.predict_proba(current[features])[:, 1]
    report = detect_drift(
        artifact.reference,
        current,
        settings.data,
        settings.drift,
        reference_scores=np.asarray(reference_scores),
        current_scores=np.asarray(current_scores),
    )
    return report, artifact


def _emit(payload: object, output: Path | None) -> None:
    text = json.dumps(payload, indent=2)
    if output:
        output.write_text(text, encoding="utf-8")
        print(f"wrote {output}")
    else:
        print(text)


def run(args: argparse.Namespace) -> int:
    if args.command == "simulate":
        make_dataset(args.n, args.seed, args.drift, args.label_shift).to_csv(
            args.output, index=False
        )
        print(f"wrote {args.n} rows to {args.output}")
        return 0

    settings = load_settings(args.config)

    if args.command == "train":
        frame = load_csv(args.data)
        require_valid(frame, settings.data)
        result = train(frame, settings)
        out = save_artifact(result, args.artifact_dir, settings)
        run_id = get_tracker(args.tracking_uri, args.runs_dir).log_run(
            "train",
            settings.model_dump(by_alias=True),
            result.metrics,
            [out / METADATA_FILE],
        )
        print(json.dumps({"run_id": run_id, "artifact_dir": str(out), "metrics": result.metrics}))
        return 0

    if args.command == "register":
        version = ModelRegistry(args.registry).register(args.name, args.artifact_dir)
        print(f"registered {args.name} version {version}")
        return 0

    if args.command == "promote":
        decision = ModelRegistry(args.registry).promote(args.name, args.version, settings.promotion)
        print(("promoted: " if decision.allowed else "not promoted: ") + decision.reason)
        return 0 if decision.allowed else 2

    if args.command == "drift":
        report, _ = _drift(settings, args.artifact_dir, args.current)
        _emit(report.to_dict(), args.output)
        return 0

    if args.command == "decide":
        report, artifact = _drift(settings, args.artifact_dir, args.current)
        trained_at = datetime.fromisoformat(str(artifact.metadata["created_at"]))
        decision = decide_retraining(report, settings.trigger, trained_at)
        DecisionLog(args.decision_log).append(decision, report.status)
        _emit({"decision": asdict(decision), "drift": report.to_dict()}, args.output)
        return 3 if (args.fail_on_retrain and decision.retrain) else 0

    import uvicorn

    from driftwatch.serving import app_from_env

    uvicorn.run(app_from_env(), host=args.host, port=args.port)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(_parser().parse_args(argv))
    except (DataValidationError, RegistryError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

