"""Serves the frozen Balanced Cumulative multi-checkpoint Random Forest
(`effective_models/pheme_multicheckpoint_rf`) as a callable scoring tool.

No model is retrained here. This loads the "cumulative_all_events.joblib"
bundle from the audited reference bundle -- the RF fitted on all PHEME
events for later inference/demo use, distinct from the per-fold estimators
used to produce the bundle's held-out (LOEO) evaluation numbers. See that
bundle's README for the full evaluation: mean horizon reduction 0.326085
under the balanced cumulative sequential policy.

The RF only knows the 57 `cumulative__` propagation-structure features
computed by effective_models/pheme_multicheckpoint_rf/features/
cumulative_features.py (reply timing, depth, fan-out, text shape --
never content truth or source credibility). Callers combine this score
with other signals (e.g. the verification agent's fact-check) rather than
treating it as a final decision.

This is not the 62-feature Hawkes-inspired model. Deploying that model would
require a separate live builder for its five decay features and a matching
62-column deployment bundle; it must not be substituted behind this interface.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BUNDLE_PATH = (
    ROOT / "effective_models" / "pheme_multicheckpoint_rf" / "reference_result"
    / "20260903_160025_balanced_cumulative_sequential_policy" / "models"
    / "cumulative_all_events.joblib"
)

# The reference bundle's balanced cumulative sequential policy: fixed budget
# of 50 threads per eligible PHEME event, split across checkpoints as
# [9, 9, 8, 8, 8, 8]. Each RadarThread cluster is this project's equivalent
# of one PHEME event (a real-world story), and each of its member posts is
# one "thread" -- these quotas/budget are scoped PER EVENT, matching the
# original policy exactly (see radar.py's _remaining_budget): an event's
# own threads compete for that event's own budget, never against an
# unrelated event's threads. Adapted to a rolling 24h window per event,
# instead of PHEME's single full-replay window -- an exploratory
# reinterpretation of the *time* boundary, not the per-event scoping,
# which is unchanged from the original.
CHECKPOINT_QUOTAS = {10: 9, 20: 9, 30: 8, 40: 8, 50: 8, 60: 8}
TOTAL_BUDGET = 50

# A post with zero observed replies gives the RF's cumulative__ features no
# real propagation structure to compute from (reply depth, gaps, latency,
# parent concentration are all trivially zero) -- confirmed directly:
# comparing two live zero-reply posts, both at the same checkpoint with
# identical timing features, still produced wildly different predicted
# impact (11.738 vs 0.364) driven almost entirely by superficial source-text
# formatting (has a URL, word-repetition ratio) rather than any real
# propagation signal, because there wasn't any yet. Requiring at least one
# real reply before a post is even considered an automatic-intervention
# candidate ensures the RF has actual structure to reason from -- it does
# not apply to on-demand manual checks (a human can still ask about any
# post they're curious about).
MIN_REPLIES_FOR_AUTO_INTERVENTION = 1


@lru_cache(maxsize=1)
def _load_bundle() -> dict:
    return joblib.load(BUNDLE_PATH)


def feature_columns() -> list[str]:
    """The exact 57 `cumulative__...` column names the RF expects, in order."""
    return list(_load_bundle()["feature_columns"])


def predict_impact(features: dict[str, float]) -> dict:
    """Score one thread at one checkpoint.

    `features` must already contain every name in feature_columns(),
    computed the same leakage-safe way as training (only using observations
    up to the checkpoint). Missing columns raise rather than silently
    defaulting -- the RF will happily produce a confident-looking number
    from a wrong-shaped or wrong-domain input without complaint.
    """
    bundle = _load_bundle()
    model = bundle["model"]
    columns = bundle["feature_columns"]

    missing = [c for c in columns if c not in features]
    if missing:
        raise ValueError(f"missing {len(missing)} required features, e.g. {missing[:5]}")

    row = pd.DataFrame([[features[c] for c in columns]], columns=columns)
    log1p_impact = float(model.predict(row)[0])

    importances = model.feature_importances_
    top_features = sorted(
        (
            {"feature": name, "importance": float(importance), "value": features[name]}
            for name, importance in zip(columns, importances)
        ),
        key=lambda item: -item["importance"],
    )[:8]

    return {
        "predicted_log1p_impact": log1p_impact,
        "predicted_impact": math.expm1(log1p_impact),
        "top_features": top_features,
    }
