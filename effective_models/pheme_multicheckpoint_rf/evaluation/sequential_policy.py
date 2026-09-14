"""Balanced Budget-50 cumulative sequential intervention replay."""
from __future__ import annotations

import pandas as pd

from effective_models.pheme_multicheckpoint_rf import config


def sequential_select(event: pd.DataFrame) -> pd.DataFrame:
    selected_ids: set[str] = set()
    chunks = []
    for checkpoint, quota in zip(config.CHECKPOINTS, config.BALANCED_QUOTAS, strict=True):
        available = event.loc[
            event["checkpoint_sec"].eq(checkpoint)
            & ~event["thread_id"].isin(selected_ids)
        ].copy()
        chosen = available.sort_values(
            ["prediction", "thread_id"], ascending=[False, True], kind="stable"
        ).head(quota)
        if len(chosen) != quota:
            raise ValueError(f"Insufficient candidates at checkpoint {checkpoint}")
        selected_ids.update(chosen["thread_id"])
        chunks.append(chosen.assign(action_checkpoint_sec=checkpoint, selection_channel="cumulative"))
    result = pd.concat(chunks, ignore_index=True)
    if result["thread_id"].duplicated().any() or len(result) != config.TOTAL_BUDGET:
        raise ValueError("Duplicate or incorrect sequential intervention count")
    return result


def event_metrics(event: pd.DataFrame) -> tuple[dict[str, float | int], pd.DataFrame]:
    selected = sequential_select(event)
    blocked = float(selected["dynamic_preventable_impact"].sum())
    horizon = float(
        event.loc[event["checkpoint_sec"].eq(config.CHECKPOINTS[0]), "future_growth"].sum()
    )
    return (
        {
            "event_id": str(event["event_id"].iloc[0]),
            "interventions": len(selected),
            "positive_interventions": int(selected["dynamic_preventable_impact"].gt(0).sum()),
            "blocked_impact": blocked,
            "horizon_future_at_10m": horizon,
            "horizon_reduction": blocked / horizon if horizon else 0.0,
            "mean_action_minute": float(selected["action_checkpoint_sec"].mean() / 60.0),
        },
        selected,
    )
