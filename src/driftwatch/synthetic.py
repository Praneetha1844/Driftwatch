"""Seeded synthetic data with controllable drift, for demos and tests.

The data is invented. It exists to exercise the pipeline, not to model any real population.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PLANS = np.array(["basic", "plus", "pro"])
BASE_PLAN_PROBS = np.array([0.6, 0.3, 0.1])
DRIFT_PLAN_SHIFT = np.array([-0.3, 0.1, 0.2])


def make_dataset(
    n: int, seed: int = 0, drift: float = 0.0, label_shift: float = 0.0
) -> pd.DataFrame:
    """Generate ``n`` rows.

    ``drift`` (0 means none, 1 is strong) moves the feature distributions: tenure shrinks,
    spend rises and widens, support calls rise and the plan mix moves towards higher tiers.
    ``label_shift`` adds a constant to the label logit, changing the positive rate without
    touching the features.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if drift < 0:
        raise ValueError("drift must not be negative")
    rng = np.random.default_rng(seed)
    tenure = np.clip(rng.normal(24 - 8 * drift, 10, n), 0, None)
    spend = np.clip(rng.normal(60 + 25 * drift, 15 * (1 + 0.5 * drift), n), 0, None)
    calls = rng.poisson(2 + 2 * drift, n)
    probs = np.clip(BASE_PLAN_PROBS + min(drift, 1.0) * DRIFT_PLAN_SHIFT, 0.01, None)
    plan = rng.choice(PLANS, size=n, p=probs / probs.sum())

    logit = (
        -1.2
        - 0.05 * tenure
        + 0.02 * spend
        + 0.3 * calls
        + 0.4 * (plan == "basic")
        + label_shift
    )
    churned = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_spend": spend,
            "support_calls": calls,
            "plan": plan,
            "churned": churned,
        }
    )
