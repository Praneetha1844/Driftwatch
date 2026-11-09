# ADR 0004: Seeded synthetic data with explicit drift controls

Status: accepted

## Context

Drift detection cannot be demonstrated or tested without data whose drift is known. Real datasets
come with licences, size and privacy concerns, and they do not offer a "no drift" and a "strong
drift" version of themselves.

## Decision

Provide `make_dataset(n, seed, drift, label_shift)`. `drift` shifts feature distributions
(tenure, spend, support calls and plan mix) and `label_shift` changes only the label logit, so
feature drift and label drift can be tested independently. Everything is derived from a seeded
`numpy` generator.

## Consequences

- Tests can assert that identical seeds reproduce exactly and that drift moves the right means.
- The data is invented. It must never be presented as a real population, and no model trained on
  it should be described as validated. The README and model card template say so.
- The generator is deliberately simple, so results on it say little about performance on messy real
  data.
