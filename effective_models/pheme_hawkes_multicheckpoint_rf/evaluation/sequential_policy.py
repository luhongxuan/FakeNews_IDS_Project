from __future__ import annotations

import pandas as pd
from effective_models.pheme_hawkes_multicheckpoint_rf import config


def select(scored: pd.DataFrame) -> pd.DataFrame:
    selected_ids: set[str] = set()
    parts = []
    for checkpoint, quota in zip(config.CHECKPOINTS_SECONDS, config.BALANCED_QUOTAS):
        candidates = scored.loc[
            scored.checkpoint_sec.eq(checkpoint) & ~scored.thread_id.isin(selected_ids)
        ].sort_values(["prediction", "thread_id"], ascending=[False, True], kind="stable")
        chosen = candidates.head(quota).copy()
        if len(chosen) != quota:
            raise ValueError(f"Insufficient candidates at checkpoint {checkpoint}")
        chosen["action_checkpoint_sec"] = checkpoint
        parts.append(chosen)
        selected_ids.update(chosen.thread_id.astype(str))
    result = pd.concat(parts, ignore_index=True)
    if len(result) != 50 or result.thread_id.nunique() != 50:
        raise ValueError("Balanced policy must select 50 unique threads")
    return result


def metrics(scored: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    chosen = select(scored)
    horizon = float(scored.loc[scored.checkpoint_sec.eq(600), "future_growth"].sum())
    blocked = float(chosen.dynamic_preventable_impact.sum())
    return {
        "interventions": 50,
        "blocked_impact": blocked,
        "horizon_future_at_10m": horizon,
        "horizon_reduction": blocked / horizon if horizon else 0.0,
        "mean_action_minute": float(chosen.action_checkpoint_sec.mean() / 60.0),
    }, chosen

