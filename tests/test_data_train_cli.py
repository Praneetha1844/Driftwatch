import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from driftwatch.cli import main
from driftwatch.config import DriftThresholds, load_settings
from driftwatch.data import DataValidationError, require_valid, validate_frame
from driftwatch.synthetic import make_dataset
from driftwatch.tracking import FileTracker, flatten, get_tracker
from driftwatch.train import load_artifact, save_artifact, train

from conftest import CONFIG


def test_valid_frame_passes(settings) -> None:
    assert validate_frame(make_dataset(50), settings.data).ok


def test_every_problem_is_reported(settings) -> None:
    frame = make_dataset(20).drop(columns="plan")
    frame.loc[0, "tenure_months"] = -1.0
    frame.loc[1, "monthly_spend"] = np.nan
    frame["support_calls"] = frame["support_calls"] + 0.5
    frame.loc[2, "churned"] = 3
    errors = validate_frame(frame, settings.data).errors
    joined = " | ".join(errors)
    assert "missing column 'plan'" in joined
    assert "below 0" in joined and "null" in joined
    assert "whole numbers" in joined and "above 1" in joined
    with pytest.raises(DataValidationError):
        require_valid(frame, settings.data)


def test_synthetic_data_is_deterministic_per_seed() -> None:
    pd.testing.assert_frame_equal(make_dataset(200, seed=3), make_dataset(200, seed=3))
    assert not make_dataset(200, seed=3).equals(make_dataset(200, seed=4))


def test_drift_parameter_moves_the_distributions() -> None:
    base, drifted = make_dataset(5000, seed=1), make_dataset(5000, seed=1, drift=1.0)
    assert drifted["tenure_months"].mean() < base["tenure_months"].mean()
    assert drifted["monthly_spend"].mean() > base["monthly_spend"].mean()
    assert drifted["support_calls"].mean() > base["support_calls"].mean()


def test_label_shift_changes_the_positive_rate_only() -> None:
    base, shifted = make_dataset(5000, seed=1), make_dataset(5000, seed=1, label_shift=1.5)
    assert shifted["churned"].mean() > base["churned"].mean()
    pd.testing.assert_series_equal(base["tenure_months"], shifted["tenure_months"])


def test_synthetic_generator_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        make_dataset(0)
    with pytest.raises(ValueError):
        make_dataset(10, drift=-0.1)


def test_settings_validate_their_own_consistency() -> None:
    assert load_settings(CONFIG).drift.psi_alert == 0.25
    with pytest.raises(ValidationError):
        DriftThresholds(psi_warn=0.3, psi_alert=0.2)


def test_training_returns_metrics_and_a_reference_sample(settings) -> None:
    result = train(make_dataset(1500, seed=0), settings)
    assert set(result.metrics) == {"accuracy", "f1", "roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in result.metrics.values())
    assert settings.data.target in result.reference.columns


def test_training_rejects_a_single_class_target(settings) -> None:
    frame = make_dataset(200, seed=0)
    frame["churned"] = 0
    with pytest.raises(ValueError):
        train(frame, settings)


def test_artifact_round_trip_preserves_predictions(settings, tmp_path) -> None:
    frame = make_dataset(1000, seed=0)
    result = train(frame, settings)
    save_artifact(result, tmp_path / "artifact", settings)
    loaded = load_artifact(tmp_path / "artifact")
    features = frame[settings.data.features]
    np.testing.assert_allclose(
        result.pipeline.predict_proba(features), loaded.pipeline.predict_proba(features)
    )
    assert loaded.metadata["features"] == settings.data.features
    with pytest.raises(ValueError):
        load_artifact(tmp_path / "missing")


def test_file_tracker_writes_params_metrics_and_artifacts(tmp_path) -> None:
    note = tmp_path / "note.txt"
    note.write_text("x", encoding="utf-8")
    run_id = FileTracker(tmp_path / "runs").log_run(
        "my run!", {"train": {"seed": 1}}, {"f1": 0.5}, [note]
    )
    run_dir = tmp_path / "runs" / run_id
    assert run_id.endswith("my-run-")
    assert json.loads((run_dir / "params.json").read_text("utf-8")) == {"train.seed": 1}
    assert json.loads((run_dir / "metrics.json").read_text("utf-8")) == {"f1": 0.5}
    assert (run_dir / "artifacts" / "note.txt").exists()


def test_flatten_and_tracker_selection(tmp_path) -> None:
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}
    assert isinstance(get_tracker(None, tmp_path), FileTracker)


def test_cli_end_to_end_flow(tmp_path, capsys) -> None:
    cfg = ["--config", str(CONFIG)]
    reference, current = tmp_path / "ref.csv", tmp_path / "cur.csv"
    artifact, registry = tmp_path / "artifact", tmp_path / "registry"

    assert main(["simulate", "--output", str(reference), "--n", "1500", "--seed", "1"]) == 0
    drifted = ["simulate", "--output", str(current), "--n", "600", "--seed", "2", "--drift", "1"]
    assert main(drifted) == 0
    assert main([*cfg, "train", "--data", str(reference), "--artifact-dir", str(artifact),
                 "--runs-dir", str(tmp_path / "runs"), "--tracking-uri", ""]) == 0
    assert main([*cfg, "register", "--name", "demo", "--artifact-dir", str(artifact),
                 "--registry", str(registry)]) == 0
    assert main([*cfg, "promote", "--name", "demo", "--version", "1",
                 "--registry", str(registry)]) == 0
    # the same version again is refused with exit code 2
    assert main([*cfg, "promote", "--name", "demo", "--version", "1",
                 "--registry", str(registry)]) == 2

    report_file = tmp_path / "drift.json"
    assert main([*cfg, "drift", "--artifact-dir", str(artifact), "--current", str(current),
                 "--output", str(report_file)]) == 0
    assert json.loads(report_file.read_text("utf-8"))["status"] == "alert"

    log = tmp_path / "decisions.jsonl"
    code = main([*cfg, "decide", "--artifact-dir", str(artifact), "--current", str(current),
                 "--decision-log", str(log), "--output", str(tmp_path / "d.json"),
                 "--fail-on-retrain"])
    assert code == 3 and len(log.read_text("utf-8").splitlines()) == 1


def test_cli_reports_errors_with_exit_code_one(tmp_path, capsys) -> None:
    code = main(["--config", str(CONFIG), "train", "--data", str(tmp_path / "nope.csv"),
                 "--artifact-dir", str(tmp_path / "a")])
    assert code == 1 and "error:" in capsys.readouterr().err


def test_simulate_creates_a_missing_output_directory(tmp_path) -> None:
    target = tmp_path / "data" / "nested" / "reference.csv"
    assert main(["--config", str(CONFIG), "simulate", "--output", str(target), "--n", "50"]) == 0
    assert target.exists()


def test_cli_rollback_moves_production_and_refuses_a_noop(tmp_path, capsys) -> None:
    from driftwatch.registry import ModelRegistry

    registry = tmp_path / "registry"
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "metadata.json").write_text(json.dumps({"metrics": {"f1": 0.6}}), "utf-8")
    store = ModelRegistry(registry)
    store.register("demo", artifact)
    store.register("demo", artifact)
    args = ["--config", str(CONFIG), "rollback", "--name", "demo", "--version", "1",
            "--reason", "version 2 misbehaves", "--registry", str(registry)]

    assert main(args) == 0
    assert store.production_version("demo") == 1
    assert "rolled back" in capsys.readouterr().out
    assert main(args) == 2  # already in production


def test_drift_command_rejects_current_data_with_nulls(settings, tmp_path, capsys) -> None:
    artifact = save_artifact(train(make_dataset(600, seed=0), settings), tmp_path / "a", settings)
    current = make_dataset(300, seed=1)
    current.loc[0, "monthly_spend"] = np.nan
    current.to_csv(tmp_path / "cur.csv", index=False)

    code = main(["--config", str(CONFIG), "drift", "--artifact-dir", str(artifact),
                 "--current", str(tmp_path / "cur.csv")])

    assert code == 1 and "null values in: ['monthly_spend']" in capsys.readouterr().err
