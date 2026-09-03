"""Inner-only thresholding and replay for quiet acceleration qualification."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CHECKPOINTS = (1200, 1800)
RECALL_TARGETS = (0.25, 0.50, 0.75)
WORST_EVENT_FRACTION = 0.50
TOP_K = 50


@dataclass(frozen=True)
class ThresholdChoice:
    checkpoint_sec: int
    target_recall: float
    threshold: float


def validate_prediction_frame(frame: pd.DataFrame, inner: bool) -> None:
    required = {
        "thread_id", "event_id", "checkpoint_sec", "prediction",
        "predicted_positive_acceleration", "next10_positive_acceleration",
    }
    if inner:
        required.add("outer_event")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    keys = ["thread_id", "event_id", "checkpoint_sec"]
    if inner:
        keys.insert(0, "outer_event")
    if frame.duplicated(keys).any():
        raise ValueError("duplicate quiet acceleration prediction")
    numeric = frame[[
        "prediction", "predicted_positive_acceleration",
        "next10_positive_acceleration",
    ]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric[:, 1:] < 0).any():
        raise ValueError("invalid quiet acceleration score or outcome")
    if inner and (frame.outer_event == frame.event_id).any():
        raise ValueError("outer event leaked into inner OOF")


def threshold_frontier(frame: pd.DataFrame, checkpoint: int) -> pd.DataFrame:
    """Enumerate score thresholds using only supplied inner OOF outcomes."""
    work = frame.loc[frame.checkpoint_sec.eq(checkpoint)].copy()
    if work.empty:
        raise ValueError(f"no inner rows at checkpoint {checkpoint}")
    work["positive"] = work.next10_positive_acceleration.gt(0)
    positive_events = [
        event for event, group in work.groupby("event_id", sort=True)
        if group.positive.any()
    ]
    if not positive_events:
        raise ValueError(f"no positive acceleration at checkpoint {checkpoint}")
    rows = []
    for threshold in np.sort(work.prediction.unique())[::-1]:
        selected = work.prediction.ge(float(threshold))
        recalls = []
        for event in positive_events:
            event_rows = work.event_id.eq(event)
            positives = event_rows & work.positive
            recalls.append(float((selected & positives).sum() / positives.sum()))
        selected_count = int(selected.sum())
        positive_hits = int((selected & work.positive).sum())
        rows.append({
            "checkpoint_sec": checkpoint,
            "threshold": float(threshold),
            "selected_rows": selected_count,
            "selection_rate": float(selected.mean()),
            "positive_hits": positive_hits,
            "positive_precision": positive_hits / selected_count if selected_count else 0.0,
            "macro_positive_recall": float(np.mean(recalls)),
            "worst_event_positive_recall": float(np.min(recalls)),
            "positive_events": len(positive_events),
        })
    return pd.DataFrame(rows)


def choose_threshold(
    frontier: pd.DataFrame,
    target_recall: float,
    worst_event_fraction: float | None = WORST_EVENT_FRACTION,
) -> pd.Series:
    feasible = frontier.loc[
        frontier.macro_positive_recall.ge(target_recall)
    ].copy()
    if worst_event_fraction is not None:
        required_worst = target_recall * worst_event_fraction
        feasible = feasible.loc[
            feasible.worst_event_positive_recall.ge(required_worst)
        ]
    if feasible.empty:
        raise ValueError(f"no feasible threshold for recall target {target_recall}")
    return feasible.sort_values(
        ["selected_rows", "macro_positive_recall", "positive_precision", "threshold"],
        ascending=[True, False, False, False], kind="stable",
    ).iloc[0]


def attach_outcomes(predictions: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "thread_id", "event_id", "checkpoint_sec", "dynamic_preventable_impact",
        "future_growth",
    ]
    if outcomes.duplicated(columns[:3]).any():
        raise ValueError("duplicate materialized outcome key")
    result = predictions.merge(
        outcomes[columns], on=columns[:3], how="left", validate="many_to_one",
    )
    if result[columns[3:]].isna().any().any():
        raise ValueError("prediction could not be traced to materialized outcomes")
    if (result.dynamic_preventable_impact < 0).any():
        raise ValueError("negative preventable impact")
    return result


def apply_thresholds(
    event: pd.DataFrame, choices: dict[int, float], policy_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply score thresholds at 20 then 30 minutes without ranking or repeats."""
    if event.event_id.nunique() != 1:
        raise ValueError("qualification replay requires one event")
    selected_ids: set[str] = set()
    selected_chunks = []
    checkpoint_rows = []
    for checkpoint in CHECKPOINTS:
        rows = event.loc[event.checkpoint_sec.eq(checkpoint)].copy()
        threshold = float(choices[checkpoint])
        raw_qualified = rows.loc[rows.prediction.ge(threshold)].copy()
        available = rows.loc[~rows.thread_id.isin(selected_ids)].copy()
        chosen = available.loc[available.prediction.ge(threshold)].copy()
        positive = rows.next10_positive_acceleration.gt(0)
        positive_total = int(positive.sum())
        raw_hits = int(raw_qualified.next10_positive_acceleration.gt(0).sum())
        acceleration_total = float(rows.next10_positive_acceleration.sum())
        checkpoint_rows.append({
            "policy_name": policy_name,
            "checkpoint_sec": checkpoint,
            "threshold": threshold,
            "quiet_rows": len(rows),
            "raw_qualified": len(raw_qualified),
            "new_interventions": len(chosen),
            "positive_rows": positive_total,
            "raw_positive_hits": raw_hits,
            "raw_positive_recall": raw_hits / positive_total if positive_total else 0.0,
            "raw_positive_precision": raw_hits / len(raw_qualified) if len(raw_qualified) else 0.0,
            "raw_acceleration_mass_capture": (
                float(raw_qualified.next10_positive_acceleration.sum()) / acceleration_total
                if acceleration_total else 0.0
            ),
        })
        if len(chosen):
            chosen["action_checkpoint_sec"] = checkpoint
            chosen["qualification_threshold"] = threshold
            chosen["policy_name"] = policy_name
            selected_chunks.append(chosen)
            selected_ids.update(chosen.thread_id)
    selected = (
        pd.concat(selected_chunks, ignore_index=True)
        if selected_chunks else event.head(0).assign(
            action_checkpoint_sec=pd.Series(dtype=int),
            qualification_threshold=pd.Series(dtype=float),
            policy_name=pd.Series(dtype=str),
        )
    )
    if selected.thread_id.duplicated().any():
        raise ValueError("thread selected more than once")
    return selected, pd.DataFrame(checkpoint_rows)


def oracle_top50_union(all_event: pd.DataFrame) -> set[str]:
    result: set[str] = set()
    for checkpoint in CHECKPOINTS:
        rows = all_event.loc[all_event.checkpoint_sec.eq(checkpoint)].sort_values(
            ["dynamic_preventable_impact", "thread_id"],
            ascending=[False, True], kind="stable",
        )
        result.update(rows.head(min(TOP_K, len(rows))).thread_id.astype(str))
    return result


def same_count_quiet_oracle(event: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """Outcome-only ceiling with the same per-checkpoint intervention counts."""
    used: set[str] = set()
    chunks = []
    for checkpoint in CHECKPOINTS:
        count = int(selected.action_checkpoint_sec.eq(checkpoint).sum())
        available = event.loc[
            event.checkpoint_sec.eq(checkpoint) & ~event.thread_id.isin(used)
        ].sort_values(
            ["dynamic_preventable_impact", "thread_id"],
            ascending=[False, True], kind="stable",
        )
        chosen = available.head(count).copy()
        chunks.append(chosen)
        used.update(chosen.thread_id)
    return pd.concat(chunks, ignore_index=True) if chunks else event.head(0)
