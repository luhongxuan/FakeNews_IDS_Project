"""Leakage-safe timing-only replay with identities committed at minute 20."""
from __future__ import annotations

import numpy as np
import pandas as pd


BUDGET = 50


def commit_at_20(utility: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    if utility.event_id.nunique() != 1 or outcomes.event_id.nunique() != 1:
        raise ValueError("commitment requires one event")
    if set(utility.event_id) != set(outcomes.event_id):
        raise ValueError("utility/outcome event mismatch")
    ranked = utility.sort_values(
        ["utility_prediction", "thread_id"],
        ascending=[False, True], kind="stable",
    ).head(BUDGET).copy()
    wide = outcomes.pivot(
        index=["thread_id", "event_id"], columns="checkpoint_sec",
        values="dynamic_preventable_impact",
    ).reset_index().rename(columns={1200: "impact_20m", 1800: "impact_30m"})
    result = ranked.merge(wide, on=["thread_id", "event_id"], how="left", validate="one_to_one")
    if result[["impact_20m", "impact_30m"]].isna().any().any():
        raise ValueError("missing committed outcome")
    result["wait_loss_20_to_30"] = result.impact_20m - result.impact_30m
    if (result.wait_loss_20_to_30 < 0).any() or len(result) != BUDGET:
        raise ValueError("invalid commitment or wait loss")
    return result


def timing_replay(
    committed: pd.DataFrame,
    quiet_scores: pd.DataFrame,
    acceleration_threshold: float,
    min_predicted_impact: float,
    policy_name: str,
) -> pd.DataFrame:
    score_columns = [
        "thread_id", "event_id", "acceleration_prediction",
        "predicted_positive_acceleration", "next10_positive_acceleration",
    ]
    scored = committed.merge(
        quiet_scores[score_columns], on=["thread_id", "event_id"],
        how="left", validate="one_to_one",
    )
    qualified = (
        scored.acceleration_prediction.ge(acceleration_threshold).fillna(False)
        & scored.predicted_impact.ge(min_predicted_impact)
    )
    scored["early_intervention"] = qualified
    scored["action_checkpoint_sec"] = np.where(qualified, 1200, 1800)
    scored["realized_blocked_impact"] = np.where(
        qualified, scored.impact_20m, scored.impact_30m
    )
    scored["policy_name"] = policy_name
    if len(scored) != BUDGET or scored.thread_id.duplicated().any():
        raise ValueError("timing replay changed committed identities")
    return scored


def timing_metrics(selected: pd.DataFrame) -> dict[str, float | int]:
    early = selected.early_intervention.astype(bool)
    early_count = int(early.sum())
    baseline = float(selected.impact_30m.sum())
    gain = float(selected.loc[early, "wait_loss_20_to_30"].sum())
    total_wait_loss = float(selected.wait_loss_20_to_30.sum())
    oracle_gain = float(
        selected.nlargest(early_count, "wait_loss_20_to_30").wait_loss_20_to_30.sum()
    ) if early_count else 0.0
    random_expected = early_count / len(selected) * total_wait_loss
    actual_acceleration_positive = selected.next10_positive_acceleration.gt(0).fillna(False)
    return {
        "interventions": len(selected),
        "early_interventions": early_count,
        "baseline_blocked_impact_at_30m": baseline,
        "timing_blocked_impact": float(selected.realized_blocked_impact.sum()),
        "timing_gain": gain,
        "total_available_wait_loss": total_wait_loss,
        "wait_loss_capture": gain / total_wait_loss if total_wait_loss else 0.0,
        "same_count_oracle_timing_gain": oracle_gain,
        "oracle_timing_efficiency": gain / oracle_gain if oracle_gain else 0.0,
        "same_count_random_expected_gain": random_expected,
        "gain_minus_random_expected": gain - random_expected,
        "early_positive_acceleration_hits": int((early & actual_acceleration_positive).sum()),
        "early_positive_acceleration_precision": (
            float((early & actual_acceleration_positive).sum() / early_count)
            if early_count else 0.0
        ),
    }
