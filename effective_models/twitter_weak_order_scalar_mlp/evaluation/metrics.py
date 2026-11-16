from __future__ import annotations
import numpy as np
import pandas as pd
from effective_models.twitter_weak_order_scalar_mlp import config


def budget_curve(frame: pd.DataFrame) -> list[dict]:
    ranked = frame.sort_values(["priority_score", "thread_id"], ascending=[False, True], kind="stable")
    oracle = frame.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable")
    total = float(frame.preventable_impact.sum()); rows = []
    for budget in config.BUDGETS:
        count = min(budget, len(frame)); blocked = float(ranked.preventable_impact.iloc[:count].sum()); best = float(oracle.preventable_impact.iloc[:count].sum())
        rows.append({"budget": budget, "actual_budget": count, "blocked_future_nodes": blocked,
                     "oracle_blocked_future_nodes": best, "reduction": blocked / total if total else 0.0,
                     "oracle_efficiency": blocked / best if best else 0.0})
    return rows

