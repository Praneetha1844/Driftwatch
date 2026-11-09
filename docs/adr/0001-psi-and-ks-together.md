# ADR 0001: Use PSI for verdicts and KS as a secondary signal

Status: accepted

## Context

Two common drift statistics behave differently:

- **PSI** compares binned proportions and gives an effect-size-like number with conventional cut-offs
  (about 0.1 and 0.25). It works for numeric and categorical features.
- **Two-sample KS** is a significance test for numeric features. With large samples it rejects for
  differences that are statistically real but practically irrelevant.

## Decision

`alert` requires PSI >= `psi_alert`. A KS p-value below `ks_alpha` can raise a feature to `warn`
but never to `alert`. Categorical features use PSI only.

## Alternatives considered

- **KS only.** Simple, but constant false alarms on big samples and no categorical support.
- **PSI only.** Misses nothing PSI can see, but gives no significance signal on small samples.
- **Wasserstein or Jensen-Shannon distance.** Reasonable, but lack widely used thresholds, so the
  numbers would be harder for a reader to interpret.

## Consequences

- Alerts track effect size, so they stay meaningful as traffic grows.
- Warnings can fire often on large samples; the trigger policy therefore requires several warnings
  (`warn_features`) before acting on them.
- PSI depends on binning. Edges come from reference quantiles with open-ended outer bins, and
  proportions are floored at `1e-4` so empty bins stay finite.
