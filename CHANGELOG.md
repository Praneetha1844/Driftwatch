# Changelog

All notable changes are recorded here. The format follows Keep a Changelog and the project uses
semantic versioning.

## [Unreleased]

### Added
- `rollback` command and `ModelRegistry.rollback`: move production to an existing version
  without the gate, with a required reason and a logged `action`.
- `serve` accepts `--name` and `--registry`, and honours `--config` and `DRIFTWATCH_CONFIG`.
- Command-line, configuration and serving API references, a development guide, a security
  policy and a documentation index.
- `.dockerignore`, project URLs, classifiers and the `py.typed` marker.

### Fixed
- Model names beginning with a dot, such as `..`, are rejected instead of resolving outside the
  registry root.
- `simulate` and `--output` create missing directories, so the README quick start works.
- `drift` and `decide` report null feature values with a clear error instead of failing inside
  scikit-learn.
- `serve` no longer ignores the model name, registry and config given to the other commands.
- `train` no longer validates the data twice.
- The environment example lists only variables that are read.

### Changed
- Docker image uses the `slim-bookworm` base and carries image labels.
- CI caches pip by `pyproject.toml`, cancels superseded runs, can be started manually and builds
  the Docker image.

## [0.1.0]

### Added

- Typed YAML settings and schema validation for training data.
- Config-driven scikit-learn training with a portable artifact format.
- File-based and MLflow trackers behind one interface.
- File registry with a gated promotion rule and a decision log.
- PSI and KS drift statistics, prediction and label drift, and a combined report.
- Retraining trigger policy with an append-only decision log.
- FastAPI service with Prometheus metrics and an on-demand drift endpoint.
- Seeded synthetic data generator with controllable drift.
- Command line interface, Dockerfile and a Compose stack (not yet run end to end).
- Model card template.
