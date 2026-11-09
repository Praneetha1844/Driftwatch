# Development guide

## Setup

Python 3.12 or newer.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"          # add ,mlflow to try the MLflow tracker
pre-commit install
```

## Make targets

| Target | What it runs |
|---|---|
| `make lint` | `ruff check .` |
| `make format` | `ruff format .` |
| `make typecheck` | `mypy` (strict) over `src` |
| `make test` | `pytest` with a coverage report |
| `make check` | lint, typecheck and test |

CI runs `ruff check`, `mypy` and `pytest`, and builds the Docker image. It does not enforce
`ruff format`, so run `make format` before committing. The pre-commit hook does it for you.

## Project layout

```
src/driftwatch/
  config.py      typed settings
  data.py        schema validation
  synthetic.py   seeded synthetic data with drift
  train.py       pipeline, metrics, artifact format
  tracking.py    MLflow and file trackers
  registry.py    versions, stages, promotion gate, rollback
  drift.py       PSI, KS, drift report
  policy.py      retraining triggers and decision log
  serving.py     FastAPI app and Prometheus metrics
  cli.py         command line
config/          default settings
deploy/          Prometheus and Grafana provisioning
tests/           pytest suite
```

## Tests

| Module | Covers |
|---|---|
| `test_drift.py` | PSI, KS, classification and `detect_drift` |
| `test_registry_policy.py` | promotion gate, registry layout, rollback, retraining policy, decision log |
| `test_data_train_cli.py` | validation, synthetic data, training, artifacts, trackers, the CLI flow |
| `test_api.py` | the serving API with a stand-in model |

- The serving tests use a fake model, so they need no trained artifact.
- Training tests fit small scikit-learn models on seeded synthetic data.
- There are no tests against a running container, a real MLflow server, Prometheus or Grafana.

## Extending

**A new model type:** add a value to `ModelConfig.type` in `config.py` and a branch in
`_estimator` in `train.py`. The rest of the pipeline is model-agnostic.

**A new drift statistic:** add a pure function in `drift.py`, use it in `detect_drift`, and
decide how it feeds `classify`. Test it on constructed arrays with a known answer.

**A new trigger rule:** add a field to `TriggerConfig` and a branch in `decide_retraining`.
Keep that function pure, with `now` passed in so it stays testable.

**A new tracker:** implement the `Tracker` protocol in `tracking.py` and select it in
`get_tracker`.

## Conventions

- Line length 100, `ruff` for lint and format, `mypy --strict` for the package.
- Conventional commit messages (`feat`, `fix`, `docs`, `test`, `build`, `ci`, `chore`).
- Thresholds belong in the settings file, not in the code.
- Decisions that change what runs in production (promotion, rollback, retraining) must be
  logged.

## Glossary

| Term | Meaning |
|---|---|
| Reference sample | The training rows kept with a model as the drift baseline |
| PSI | Population Stability Index, a measure of how far a distribution has moved |
| KS test | Two-sample Kolmogorov-Smirnov test of whether two samples differ |
| Artifact | A saved model, its reference sample and metadata |
| Stage | A named pointer to a version. Only `production` exists. |
| Gate | The promotion rule: beat production by a margin on a chosen metric |
| Rollback | Moving production to an earlier version without the gate, with a reason |
