# Model card (template)

This is a **template**. Driftwatch does not ship a trained model and makes no claim about the
performance of any model trained with it. Copy this file next to a model you actually train and
fill in every section from your own evaluation.

| Field | Value |
| --- | --- |
| Model name and version | _from the registry_ |
| Trained on (date, commit) | _from `metadata.json` and git_ |
| Model type and parameters | _from the settings file_ |
| Owner | _name and contact_ |

## Intended use

- What decision does the model support, and who acts on it?
- What is explicitly out of scope?

## Training data

- Source, collection period and licence.
- Row count, label definition and class balance.
- Known gaps: groups, regions or time spans that are missing or thin.
- Whether any of the data is synthetic. (The demo dataset produced by `driftwatch simulate` is
  synthetic and invented; do not describe a model trained on it as validated.)

## Evaluation

| Metric | Value | Data it was measured on |
| --- | --- | --- |
| accuracy | _fill in_ | _held-out split, size, date_ |
| f1 | _fill in_ | |
| roc_auc | _fill in_ | |

- How the split was made and whether it respects time.
- Performance per relevant subgroup, if the model affects people.
- Calibration of the predicted probabilities.

## Monitoring

- Thresholds in force (`psi_warn`, `psi_alert`, `ks_alpha`, `min_samples`) and why.
- Who reviews drift reports and how often.
- Retraining policy (`alert_features`, `warn_features`, `schedule_days`) and the decision log
  location.

## Risks and limitations

- Failure modes you have observed or expect.
- Features that could act as proxies for sensitive attributes.
- Conditions under which the model should be switched off.

## Change log

| Version | Date | Change | Promotion decision |
| --- | --- | --- | --- |
| | | | _link to the `decisions.jsonl` entry_ |
