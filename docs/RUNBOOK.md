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

## `promote` exits with code 2

The gate refused. Read the reason in the output or `decisions.jsonl`: either the candidate does
not beat production by the margin, or it is below `min_absolute`. Do not lower the margin to
force a promotion without recording the justification.

## Rolling back

Point `stages.json` at an earlier version, or promote the earlier version again after
removing the current stage file, and restart the service. The decision log keeps the history.
