# Serving API reference

`driftwatch serve` runs a FastAPI application for the production version of one model.
Interactive OpenAPI documentation is served at `/docs` while it is running.
There is no authentication. Keep it on an internal network.

## `POST /predict`

Score up to 1000 records.

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "records": [{"tenure_months": 6, "monthly_spend": 80, "support_calls": 4, "plan": "basic"}]
}'
```

Response (illustrative values):

```json
{
  "model_version": "1",
  "predictions": [{"probability": 0.73, "label": 1}]
}
```

- `label` is `1` when the probability is at least `0.5`. The threshold is fixed.
- Every feature the model was trained on must be present and non-null. Extra keys are ignored.
- The scored rows and probabilities are kept in a bounded in-memory buffer (default 5000) for
  `/drift`.

| Status | Meaning |
|---|---|
| `200` | Scored |
| `422` | Empty or oversized `records`, missing or null features (the names are listed), or values the model cannot score |

## `GET /drift`

Compare the buffered requests with the model's reference sample.

- Fewer than `drift.min_samples` buffered rows returns
  `{"status": "insufficient_data", "n_current": N, "min_samples": M}`.
- Otherwise the response is a drift report: overall `status` (`ok`, `warn` or `alert`), a
  verdict per feature with PSI and the KS statistic and p-value, and the prediction drift.
- Calling it updates the `driftwatch_feature_psi` gauge.
- `503` if drift monitoring is not configured for the app.

Label drift is not reported here, because requests carry no outcomes.

## `GET /health`

```json
{"status": "ok", "model_version": "1"}
```

## `GET /metrics`

Prometheus text format.

| Metric | Type | Meaning |
|---|---|---|
| `driftwatch_predictions_total{model_version}` | counter | Rows scored |
| `driftwatch_rejected_requests_total` | counter | Requests refused with `422` for features or scoring |
| `driftwatch_predict_seconds` | histogram | `/predict` latency |
| `driftwatch_model_info{model_version}` | gauge | Which version is serving |
| `driftwatch_feature_psi{feature}` | gauge | PSI from the most recent `/drift` call |

## Behaviour to know about

- The service loads the production version once at start-up. After a promotion or rollback,
  restart it to serve the new version.
- The buffer and metrics belong to one process. They reset on restart and are not shared
  across replicas.
- Artifacts are loaded with `joblib`, which uses pickle. Only serve artifacts you produced.
