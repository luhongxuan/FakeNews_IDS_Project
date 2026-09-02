"""Serves the already-validated v5 text RF nested-LOEO out-of-fold predictions.

No model is retrained here. This reads the existing, already-audited OOF CSV
(effective_models/pheme_v5_text_rf/reference_result/.../oof_thread_scores.csv)
so the dashboard shows the project's real, leakage-safe research result
instead of placeholder or random values.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
V5_OOF = (
    ROOT / "effective_models" / "pheme_v5_text_rf" / "reference_result"
    / "20260831_223827_909201_v5_best_text_oof_min_interventions" / "oof_thread_scores.csv"
)
CUTOFF_SECONDS = 1800


@lru_cache(maxsize=1)
def _load_scores() -> pd.DataFrame:
    frame = pd.read_csv(V5_OOF, dtype={"thread_id": str, "event_id": str})
    frame["percentile_score"] = 0.0
    for _, group in frame.groupby("event_id"):
        percentile = group["prediction"].rank(method="min", pct=True) * 100.0
        frame.loc[group.index, "percentile_score"] = percentile.round(1)
    return frame


def ranked_threads(event_id: str, limit: int = 20) -> list[dict]:
    """Threads for one event, ranked by the real OOF predicted score (highest first)."""
    frame = _load_scores()
    subset = frame.loc[frame.event_id == event_id].copy()
    if subset.empty:
        return []
    subset = subset.sort_values(["prediction", "thread_id"], ascending=[False, True]).reset_index(drop=True)
    subset["rank"] = subset.index + 1
    subset = subset.head(limit)
    return subset[["thread_id", "rank", "percentile_score", "prediction", "preventable_impact"]].to_dict(orient="records")


def event_id_for_thread(thread_id: str) -> str | None:
    frame = _load_scores()
    row = frame.loc[frame.thread_id == thread_id]
    return None if row.empty else str(row.iloc[0].event_id)


@lru_cache(maxsize=1)
def _event_severity_lookup() -> dict[str, str]:
    """Cross-event tertiles of each event's top-decile mean predicted score.

    percentile_score is relative to its own event by construction, so
    averaging an event's own top decile of percentile_score is circular and
    always lands near 100 -- it can't distinguish events from each other.
    This instead compares the raw `prediction` (log1p predicted impact, on
    the same scale across all events) across events, so severity actually
    reflects how one event's predicted risk compares to the others.
    """
    frame = _load_scores()
    top_decile_means = {}
    for event, group in frame.groupby("event_id"):
        top_k = max(1, int(len(group) * 0.1))
        top_decile_means[event] = float(group.sort_values("prediction", ascending=False).head(top_k).prediction.mean())
    ordered = sorted(top_decile_means.values())
    n = len(ordered)
    low_cut = ordered[n // 3]
    high_cut = ordered[(2 * n) // 3]
    return {
        event: ("high" if value >= high_cut else "medium" if value >= low_cut else "low")
        for event, value in top_decile_means.items()
    }


def event_severity(event_id: str) -> str:
    """Deterministic severity from the real model's cross-event risk tertile."""
    return _event_severity_lookup().get(event_id, "low")
