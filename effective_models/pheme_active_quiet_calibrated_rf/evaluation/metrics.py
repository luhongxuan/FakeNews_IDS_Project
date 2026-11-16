"""Fixed-budget ranking metrics for calibrated Active/Quiet scores."""
from __future__ import annotations

import pandas as pd

from effective_models.pheme_active_quiet_calibrated_rf import config


def curve_rows(pool: pd.DataFrame, score_column: str, model: str) -> list[dict[str, object]]:
    ranked = pool.sort_values([score_column, "thread_id"], ascending=[False, True], kind="stable")
    oracle = pool.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable")
    total = float(pool["preventable_impact"].sum())
    rows = []
    for budget in config.BUDGETS:
        actual = min(budget, len(pool))
        blocked = float(ranked.head(actual)["preventable_impact"].sum())
        optimal = float(oracle.head(actual)["preventable_impact"].sum())
        rows.append(
            {
                "model": model,
                "event_id": str(pool["event_id"].iloc[0]),
                "budget": budget,
                "actual_budget": actual,
                "model_blocked_impact": blocked,
                "oracle_blocked_impact": optimal,
                "model_reduction": blocked / total if total else 0.0,
                "oracle_efficiency": blocked / optimal if optimal else 0.0,
            }
        )
    return rows
