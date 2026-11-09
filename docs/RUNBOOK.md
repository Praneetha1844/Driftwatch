# Operations runbook

A short guide to acting on Driftwatch output. Thresholds are starting points; adjust them in
`config/default.yaml` and record why in the model card.

## A drift report says `insufficient_data`

Fewer than `min_samples` rows were available. Wait for more traffic or lower `min_samples`
deliberately. Do not act on a verdict computed from a handful of rows.

## One feature is in `alert`

1. Confirm the data, not the model, changed: check upstream pipelines, units, encodings and
   new categories first. A schema or unit change is a bug to fix, not drift to retrain on.
2. Compare the feature's recent distribution with the reference sample.
3. If the change is real, retrain on recent data, evaluate, register, and let the promotion
   gate decide.

## Several features are in `warn`

Warnings are common on large samples (see ADR 0001). Treat a single warning as a note. The
policy recommends retraining only when `warn_features` features warn at once.

## Prediction drift without feature drift

The model's output moved while its inputs did not. Check for a changed upstream score, a
different mix of scoring requests, or a bug in preprocessing at serving time.

## `decide` exits with code 3

Retraining is recommended. The reasons are in the JSON output and in `decisions.jsonl`. Exit
code 3 is a signal for a scheduler, not an instruction to retrain blindly.

## `drift` or `decide` says the current data has null values

The command refuses to score rows with missing features. Fix or impute them upstream. Do not
drop the affected rows silently, because that changes the distribution being measured.

## `promote` exits with code 2

The gate refused. Read the reason in the output or `decisions.jsonl`: either the candidate does
not beat production by the margin, or it is below `min_absolute`. Do not lower the margin to
force a promotion without recording the justification.

## Rolling back

The promotion gate refuses to move production to a lower-scoring version, so use the rollback
command, which bypasses the gate and requires a reason:

```bash
driftwatch rollback --name churn-demo --version 1 --reason "version 2 regressed on live data"
```

Then restart the service, because it loads the production version once at start-up. The change
is appended to `<registry>/<name>/decisions.jsonl` with `"action": "rollback"`, next to every
promotion attempt. Rolling back to the version already in production is refused (exit code 2).
