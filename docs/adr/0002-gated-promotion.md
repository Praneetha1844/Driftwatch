# ADR 0002: Gate promotion on a margin over the current production model

Status: accepted

## Context

A newly trained model is not automatically better. Noise in a single held-out split can make a
worse model look slightly better, and every promotion carries operational risk.

## Decision

Promotion to `production` requires `candidate >= baseline + margin` on one configured metric
(`f1` by default, margin 0.01). With no production model, the candidate only needs
`min_absolute`. Every attempt, allowed or refused, is appended to `decisions.jsonl` with the
numbers and the reason.

## Alternatives considered

- **Always promote the newest.** Simple, but drift-triggered retrains can silently regress.
- **Statistical test on paired predictions.** More rigorous, but needs the same evaluation rows
  for both models, which the registry does not hold.
- **Multiple metrics with guardrails.** More realistic, deferred until there is a use case that
  needs it.

## Consequences

- A tie or a tiny gain is refused, which protects against noise but can delay a useful model if the
  margin is set too high.
- The comparison uses each artifact's own recorded metrics, which come from different splits if the
  data changed. The decision log keeps the evidence so a reviewer can see that.
- The CLI returns exit code 2 when refused, so automation can stop.
