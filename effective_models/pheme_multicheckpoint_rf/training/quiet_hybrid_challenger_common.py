"""Fixed-budget replay for early quiet challengers and a frozen 30m hybrid ranker."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


BUDGET = 50
CHECKPOINTS = (1200, 1800)
RESERVE_PROFILES = ((0, 0), (2, 3), (5, 5), (10, 10))


@dataclass(frozen=True)
class ChallengerConfig:
    gate_policy: str
    acceleration_quantile: float
    min_predicted_impact: float
    acceleration_threshold_20m: float
    acceleration_threshold_30m: float
    reserve_20m: int
    reserve_30m: int

    @property
    def policy_name(self) -> str:
        if self.reserve_20m == self.reserve_30m == 0:
            return "frozen_quiet_tier_hybrid_top50"
        return (
            f"{self.gate_policy}_reserve"
            f"{self.reserve_20m}_{self.reserve_30m}"
        )


def replay(
    dual_event: pd.DataFrame,
    hybrid_event: pd.DataFrame,
    config: ChallengerConfig,
) -> pd.DataFrame:
    if dual_event.event_id.nunique() != 1 or hybrid_event.event_id.nunique() != 1:
        raise ValueError("challenger replay requires one event")
    if set(dual_event.event_id) != set(hybrid_event.event_id):
        raise ValueError("dual and hybrid event mismatch")
    used: set[str] = set()
    chunks = []
    thresholds = {
        1200: config.acceleration_threshold_20m,
        1800: config.acceleration_threshold_30m,
    }
    reserves = {1200: config.reserve_20m, 1800: config.reserve_30m}
    for checkpoint in CHECKPOINTS:
        available = dual_event.loc[
            dual_event.checkpoint_sec.eq(checkpoint)
            & ~dual_event.thread_id.isin(used)
            & dual_event.acceleration_prediction.ge(thresholds[checkpoint])
            & dual_event.predicted_impact.ge(config.min_predicted_impact)
        ].sort_values(
            ["predicted_impact", "acceleration_prediction", "thread_id"],
            ascending=[False, False, True], kind="stable",
        )
        chosen = available.head(reserves[checkpoint]).copy()
        if len(chosen):
            chosen["action_checkpoint_sec"] = checkpoint
            chosen["selection_source"] = "quiet_challenger"
            chosen["hybrid_calibrated_score"] = np.nan
            chunks.append(chosen[[
                "thread_id", "event_id", "action_checkpoint_sec", "selection_source",
                "dynamic_preventable_impact", "future_growth", "predicted_impact",
                "acceleration_prediction", "hybrid_calibrated_score",
            ]])
            used.update(chosen.thread_id)
    remaining = BUDGET - len(used)
    fill = hybrid_event.loc[~hybrid_event.thread_id.isin(used)].sort_values(
        ["hybrid_calibrated_score", "thread_id"],
        ascending=[False, True], kind="stable",
    ).head(remaining).copy()
    if len(fill) != remaining:
        raise ValueError("insufficient hybrid candidates to fill fixed budget")
    fill["action_checkpoint_sec"] = 1800
    fill["selection_source"] = "hybrid_fill"
    fill["predicted_impact"] = np.nan
    fill["acceleration_prediction"] = np.nan
    chunks.append(fill[[
        "thread_id", "event_id", "action_checkpoint_sec", "selection_source",
        "dynamic_preventable_impact", "future_growth", "predicted_impact",
        "acceleration_prediction", "hybrid_calibrated_score",
    ]])
    result = pd.concat(chunks, ignore_index=True)
    result["policy_name"] = config.policy_name
    if len(result) != BUDGET or result.thread_id.duplicated().any():
        raise ValueError("fixed-budget challenger replay violated budget or identity")
    return result
