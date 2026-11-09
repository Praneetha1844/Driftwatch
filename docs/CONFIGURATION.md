# Configuration reference

Behaviour is configured in one YAML file, [`config/default.yaml`](../config/default.yaml),
validated by typed models in `src/driftwatch/config.py`. Invalid values stop the command with
`error: ...` and exit code 1. Every threshold lives here, not in the code.

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `DRIFTWATCH_CONFIG` | all commands, `serve`, the container | Path of the settings YAML. Default `config/default.yaml`. |
| `DRIFTWATCH_MODEL_NAME` | `serve`, the container | Model to serve. Default `churn-demo`. |
| `DRIFTWATCH_REGISTRY_DIR` | `serve`, the container | Registry root. Default `registry` (`/data/registry` in the image). |
| `MLFLOW_TRACKING_URI` | `train` | Log runs to MLflow instead of files. |
| `GRAFANA_ADMIN_PASSWORD` | `docker-compose.yml` | Grafana admin password. Required by the Compose file. |

Command-line options override the matching variables.

## `schema`

Describes the training data.

| Key | Meaning |
|---|---|
| `target` | Name of the label column. It must be declared in `columns`. |
| `columns.<name>.dtype` | `float`, `int` or `category`. |
| `columns.<name>.nullable` | Allow missing values. Default `false`. |
| `columns.<name>.min`, `max` | Inclusive bounds for numeric columns. |

Numeric columns other than the target become scaled numeric features. `category` columns are
one-hot encoded, and categories not seen in training are ignored at serving time.

## `train`

| Key | Default | Meaning |
|---|---|---|
| `test_size` | `0.2` | Held-out fraction, strictly between 0 and 1. The split is stratified. |
| `seed` | `42` | Seed for the split, the model and the reference sample. |
| `reference_rows` | `5000` | Maximum rows kept as the drift baseline. At least 100. |
| `model.type` | `logistic_regression` | `logistic_regression` or `random_forest`. |
| `model.params` | `{}` | Extra scikit-learn parameters passed to the estimator. |

## `drift`

| Key | Default | Meaning |
|---|---|---|
| `bins` | `10` | PSI bins for numeric features, taken from reference quantiles. At least 2. |
| `psi_warn` | `0.1` | PSI at or above this is a warning. Must be below `psi_alert`. |
| `psi_alert` | `0.25` | PSI at or above this is an alert. |
| `ks_alpha` | `0.05` | A KS p-value below this raises a warning, never an alert. |
| `min_samples` | `200` | Fewer current rows than this gives `insufficient_data`. |

The PSI cut-offs are conventional rules of thumb. Tune them per feature and sample size.

## `promotion`

| Key | Default | Meaning |
|---|---|---|
| `metric` | `f1` | `accuracy`, `f1` or `roc_auc`, measured on the held-out split. |
| `margin` | `0.01` | The candidate must beat production by at least this much. |
| `min_absolute` | `0.0` | Minimum metric when nothing is in production yet. |

## `trigger`

| Key | Default | Meaning |
|---|---|---|
| `alert_features` | `1` | Features in alert that recommend retraining. |
| `warn_features` | `3` | Features in warning that recommend retraining. |
| `schedule_days` | `30` | Recommend retraining when the model is this old. |
| `prediction_drift_alert` | `true` | An alert on the prediction distribution recommends retraining. |

## Using a different dataset

Copy `config/default.yaml`, edit `schema` to match your columns, and pass it with `--config`.
The demo generator only produces the demo columns, so supply your own CSVs.
