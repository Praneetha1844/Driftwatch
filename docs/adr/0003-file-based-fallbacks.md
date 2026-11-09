# ADR 0003: File-based tracker and registry as the default

Status: accepted

## Context

MLflow is a good tracker and registry, but it is a service to run. A learner, a CI job or a quick
experiment should not need one.

## Decision

Define a small `Tracker` protocol with two implementations. `FileTracker` writes
`params.json`, `metrics.json` and artifacts per run and has no dependencies. `MlflowTracker` is
selected only when a tracking URI is configured, and `mlflow` is an optional extra imported
lazily. The registry is file-based for the same reason and uses a plain directory layout that is
easy to inspect and back up.

## Consequences

- The whole loop runs with no external services, which also keeps tests hermetic.
- The file registry has no locking, so it supports one writer. Multi-user or multi-host use should
  move to a registry service behind the same method names.
- Run metadata is less searchable than in MLflow. Docker Compose includes an MLflow server for
  teams that want the UI.
