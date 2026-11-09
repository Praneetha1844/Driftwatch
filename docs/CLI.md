# Command-line reference

```
driftwatch [--config FILE] <command> [options]
```

`--config` is the settings YAML. If it is not given, `DRIFTWATCH_CONFIG` is used, then
`config/default.yaml` relative to the working directory. It goes **before** the command.
See [CONFIGURATION.md](CONFIGURATION.md) for what the file contains.

## Commands

### `simulate`

Write a seeded synthetic dataset. The data is invented, for demos and tests only.

| Option | Default | Meaning |
|---|---|---|
| `--output FILE` | required | CSV to write. Missing directories are created. |
| `--n N` | `5000` | Number of rows. |
| `--seed N` | `0` | Random seed. |
| `--drift D` | `0.0` | 0 means none, 1 is strong. Shrinks tenure, raises spend and support calls, shifts the plan mix. |
| `--label-shift S` | `0.0` | Constant added to the label logit, changing the positive rate only. |

### `train`

Validate the data against the schema, train, write an artifact and log the run.

| Option | Default | Meaning |
|---|---|---|
| `--data FILE` | required | Training CSV. |
| `--artifact-dir DIR` | required | Where to write `model.joblib`, `reference.csv` and `metadata.json`. |
| `--runs-dir DIR` | `runs` | File tracker output, used when no tracking URI is set. |
| `--tracking-uri URI` | `MLFLOW_TRACKING_URI` | Log to MLflow instead of files. Needs the `mlflow` extra. |

Prints the run id, the artifact directory and the held-out metrics as JSON.

### `register`

Copy an artifact into the registry as the next version of a model.

| Option | Default | Meaning |
|---|---|---|
| `--name NAME` | required | Model name. Letters, digits, `_`, `.` and `-`, starting with a letter or digit. |
| `--artifact-dir DIR` | required | Artifact to copy. |
| `--registry DIR` | `registry` | Registry root. |

### `promote`

Promote a version to production if the gate allows it.

| Option | Default | Meaning |
|---|---|---|
| `--name NAME`, `--version N` | required | The version to promote. |
| `--registry DIR` | `registry` | Registry root. |

The candidate must beat the production version by `promotion.margin` on `promotion.metric`.
With no production version it only has to reach `promotion.min_absolute`. Every attempt is
appended to `<registry>/<name>/decisions.jsonl`.

### `rollback`

Point production at an existing version **without** the gate, for example to undo a bad release.

| Option | Default | Meaning |
|---|---|---|
| `--name NAME`, `--version N` | required | The version to restore. |
| `--reason TEXT` | required | Why. Must not be blank. Recorded in the decision log. |
| `--registry DIR` | `registry` | Registry root. |

### `drift`

Compare current data with the artifact's reference sample and print a drift report.

| Option | Default | Meaning |
|---|---|---|
| `--artifact-dir DIR` | required | The model artifact whose reference sample is the baseline. |
| `--current FILE` | required | CSV of recent data. It needs every feature column, with no nulls. |
| `--output FILE` | stdout | Write the JSON report to a file. |

### `decide`

Run `drift`, then apply the retraining policy and append the decision to a log.

| Option | Default | Meaning |
|---|---|---|
| `--artifact-dir`, `--current`, `--output` | as for `drift` | |
| `--decision-log FILE` | `decisions.jsonl` | Append-only JSON-lines audit log. |
| `--fail-on-retrain` | off | Exit with code 3 when retraining is recommended. |

### `serve`

Run the API for the production version of a model.

| Option | Default | Meaning |
|---|---|---|
| `--host`, `--port` | `127.0.0.1`, `8000` | Bind address. |
| `--name NAME` | `DRIFTWATCH_MODEL_NAME`, then `churn-demo` | Model to serve. |
| `--registry DIR` | `DRIFTWATCH_REGISTRY_DIR`, then `registry` | Registry root. |

Refuses to start if the model has no production version. See [API.md](API.md).

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | An error: invalid data, unknown version or name, unreadable file, bad settings |
| `2` | `promote` or `rollback` was refused |
| `3` | `decide --fail-on-retrain` and retraining is recommended |

Errors are printed to stderr as `error: <message>`.

## Example session

```bash
driftwatch simulate --output data/reference.csv --n 5000 --seed 1
driftwatch simulate --output data/current.csv --n 1000 --seed 2 --drift 1.0
driftwatch train --data data/reference.csv --artifact-dir artifacts/v1
driftwatch register --name churn-demo --artifact-dir artifacts/v1
driftwatch promote --name churn-demo --version 1
driftwatch decide --artifact-dir artifacts/v1 --current data/current.csv --fail-on-retrain
driftwatch rollback --name churn-demo --version 1 --reason "version 2 regressed"
driftwatch serve --name churn-demo
```
