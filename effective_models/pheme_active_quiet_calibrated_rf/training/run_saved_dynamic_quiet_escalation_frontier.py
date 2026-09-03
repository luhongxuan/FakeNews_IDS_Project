"""Evaluate a frozen-quiet 30--60 minute escalation-trigger policy.

This is a read-only saved-score diagnostic.  It does not fit a model or use
outcomes to choose threads.  A thread's intervention time is the first
observable reply-to-reply edge, or the second new reply within ten minutes,
after the strict-30 cutoff and no later than 60 minutes.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE_RUN = (
    HERE.parent
    / "experiments"
    / "20260902_132700_413954_quiet_tier_hybrid_policy_full"
)
SCORES = SOURCE_RUN / "outer_scores.csv"
REPLIES = (
    ROOT
    / "data"
    / "protected_research_assets"
    / "pheme_v5_strict30"
    / "pheme_reply_level_v4.csv"
)
OUT_ROOT = HERE.parent / "experiments"
SCORE_COLUMN = "hybrid_calibrated_score"
TOTAL_BUDGET = 50
RESERVED_SLOTS = (0, 5, 10, 15, 20)
START_SEC = 1800.0
END_SEC = 3600.0
BURST_WINDOW_SEC = 600.0


def write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


def first_trigger(group: pd.DataFrame, source_id: str) -> tuple[float | None, str | None]:
    candidates = group.loc[
        group.is_source.ne(1)
        & group.offset_sec.gt(START_SEC)
        & group.offset_sec.le(END_SEC)
    ].sort_values(["offset_sec", "tweet_id"], kind="stable")
    recent: deque[float] = deque()
    for row in candidates.itertuples(index=False):
        timestamp = float(row.offset_sec)
        while recent and recent[0] < timestamp - BURST_WINDOW_SEC:
            recent.popleft()
        recent.append(timestamp)
        reply_to_reply = str(row.parent_id) != source_id
        burst = len(recent) >= 2
        if reply_to_reply or burst:
            if reply_to_reply and burst:
                reason = "reply_to_reply_and_two_in_10m"
            elif reply_to_reply:
                reason = "reply_to_reply"
            else:
                reason = "two_in_10m"
            return timestamp, reason
    return None, None


def preventable_after(
    parents: dict[str, str | None], offsets: dict[str, float], cutoff: float
) -> int:
    """Count post-cutoff descendants of nodes observable by this cutoff."""
    children: dict[str, list[str]] = defaultdict(list)
    roots = []
    for node_id, parent_id in parents.items():
        if parent_id is None or parent_id not in parents:
            roots.append(node_id)
        else:
            children[parent_id].append(node_id)
    if not roots:
        raise ValueError("Thread has no traversable root")
    observed = {node_id for node_id, value in offsets.items() if value <= cutoff}
    blocked = 0

    def visit(node_id: str, inherited_block: bool) -> None:
        nonlocal blocked
        if offsets[node_id] <= cutoff:
            next_block = node_id in observed
        else:
            if inherited_block:
                blocked += 1
            next_block = inherited_block
        for child_id in children.get(node_id, []):
            visit(child_id, next_block)

    for root in roots:
        visit(root, False)
    return blocked


def build_trigger_table(raw: pd.DataFrame, eligible_ids: set[str]) -> pd.DataFrame:
    rows = []
    filtered = raw.loc[raw.thread_id.isin(eligible_ids)].copy()
    total = filtered.thread_id.nunique()
    for index, (thread_id, group) in enumerate(
        filtered.groupby("thread_id", sort=True), 1
    ):
        source_rows = group.loc[group.is_source.eq(1)]
        if len(source_rows) != 1:
            raise ValueError(f"{thread_id}: expected one source row")
        source_id = str(source_rows.iloc[0].tweet_id)
        trigger_time, trigger_reason = first_trigger(group, source_id)
        if trigger_time is not None:
            parents = {
                str(row.tweet_id): (
                    None if pd.isna(row.parent_id) else str(row.parent_id)
                )
                for row in group.itertuples(index=False)
            }
            offsets = {
                str(row.tweet_id): float(row.offset_sec)
                for row in group.itertuples(index=False)
            }
            impact = preventable_after(parents, offsets, trigger_time)
            rows.append(
                {
                    "thread_id": str(thread_id),
                    "trigger_time_sec": trigger_time,
                    "trigger_reason": trigger_reason,
                    "dynamic_preventable_impact": impact,
                }
            )
        if index % 400 == 0 or index == total:
            print(f"  trigger/impact audit {index}/{total}", flush=True)
    return pd.DataFrame(rows)


def evaluate_event(
    event: pd.DataFrame, triggers: pd.DataFrame, reserved: int
) -> tuple[dict, pd.DataFrame]:
    ranked = event.sort_values(
        [SCORE_COLUMN, "thread_id"], ascending=[False, True], kind="stable"
    )
    early_slots = TOTAL_BUDGET - reserved
    early = ranked.head(early_slots).copy()
    early["stage"] = "strict30"
    early["intervention_time_sec"] = START_SEC
    early["blocked_impact"] = early.preventable_impact.astype(float)
    early["trigger_reason"] = "strict30_score"

    early_ids = set(early.thread_id)
    dynamic_pool = (
        event.loc[event.quiet & ~event.thread_id.isin(early_ids)]
        .merge(triggers, on="thread_id", how="inner", validate="one_to_one")
        .sort_values(
            ["trigger_time_sec", SCORE_COLUMN, "thread_id"],
            ascending=[True, False, True],
            kind="stable",
        )
    )
    dynamic = dynamic_pool.head(reserved).copy()
    dynamic["stage"] = "dynamic_escalation"
    dynamic["intervention_time_sec"] = dynamic.trigger_time_sec
    dynamic["blocked_impact"] = dynamic.dynamic_preventable_impact.astype(float)
    selected = pd.concat([early, dynamic], ignore_index=True, sort=False)
    if selected.thread_id.duplicated().any() or len(selected) > TOTAL_BUDGET:
        raise AssertionError("Invalid two-stage selection")

    oracle = event.sort_values(
        ["preventable_impact", "thread_id"],
        ascending=[False, True],
        kind="stable",
    ).head(TOTAL_BUDGET)
    oracle_ids = set(oracle.thread_id)
    selected_ids = set(selected.thread_id)
    blocked = float(selected.blocked_impact.sum())
    total = float(event.preventable_impact.sum())
    oracle_blocked = float(oracle.preventable_impact.sum())
    metrics = {
        "event_id": str(event.event_id.iloc[0]),
        "reserved_slots": reserved,
        "early_interventions": len(early),
        "dynamic_interventions": len(dynamic),
        "unused_slots": TOTAL_BUDGET - len(selected),
        "mean_dynamic_trigger_sec": (
            float(dynamic.trigger_time_sec.mean()) if len(dynamic) else np.nan
        ),
        "blocked_impact": blocked,
        "model_reduction": blocked / total if total else 0.0,
        "oracle_efficiency": blocked / oracle_blocked if oracle_blocked else 0.0,
        "oracle_top50_hits": len(selected_ids & oracle_ids),
        "dynamic_oracle_top50_hits": len(set(dynamic.thread_id) & oracle_ids),
        "dynamic_blocked_impact": float(dynamic.blocked_impact.sum()),
    }
    selected["event_id"] = str(event.event_id.iloc[0])
    selected["reserved_slots"] = reserved
    selected["oracle_top50"] = selected.thread_id.isin(oracle_ids)
    return metrics, selected


def main() -> None:
    output = OUT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + "_saved_dynamic_quiet_escalation_frontier"
    )
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "saved-score dynamic strict30-to-60 quiet escalation frontier",
        "source_scores": str(SCORES.resolve()),
        "reply_source": str(REPLIES.resolve()),
        "score_column": SCORE_COLUMN,
        "total_budget_per_event": TOTAL_BUDGET,
        "reserved_slots": list(RESERVED_SLOTS),
        "trigger": (
            "first observable reply-to-reply OR second new reply within 600 seconds, "
            "strictly after 1800 and at/before 3600 seconds"
        ),
        "research_safety": (
            "The frozen quiet cohort and early score are saved held-out strict-30 "
            "outputs. Trigger selection uses only replies observable at trigger time. "
            "Post-trigger nodes are outcomes only. No test outcome selects a policy."
        ),
    }
    write_record(output, record)
    try:
        print("Starting saved dynamic quiet-escalation frontier.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/5] Loading and validating saved strict-30 held-out scores...", flush=True)
        scores = pd.read_csv(SCORES, dtype={"thread_id": str, "event_id": str})
        required = {
            "thread_id",
            "event_id",
            "preventable_impact",
            SCORE_COLUMN,
            "quiet",
        }
        if required - set(scores) or scores.thread_id.duplicated().any():
            raise ValueError("Invalid saved score artifact")
        if scores.event_id.nunique() != 7:
            raise ValueError("Expected seven eligible held-out events")

        print("[2/5] Loading protected reply trees...", flush=True)
        raw = pd.read_csv(
            REPLIES,
            dtype={"thread_id": str, "tweet_id": str, "parent_id": str},
            usecols=[
                "thread_id",
                "tweet_id",
                "parent_id",
                "is_source",
                "offset_sec",
            ],
            low_memory=False,
        )
        print("[3/5] Materializing observable triggers and trigger-time impacts...", flush=True)
        triggers = build_trigger_table(raw, set(scores.thread_id))
        if triggers.thread_id.duplicated().any():
            raise ValueError("Duplicate trigger identity")

        print("[4/5] Evaluating fixed Top-50 reservation frontier...", flush=True)
        metrics_rows = []
        selection_rows = []
        for reserved in RESERVED_SLOTS:
            print(f"  reserved dynamic slots {reserved}/{TOTAL_BUDGET}", flush=True)
            for _, event in scores.groupby("event_id", sort=True):
                metrics, selected = evaluate_event(event, triggers, reserved)
                metrics_rows.append(metrics)
                selection_rows.append(selected)
        metrics_frame = pd.DataFrame(metrics_rows)
        selections = pd.concat(selection_rows, ignore_index=True)
        summary = metrics_frame.groupby("reserved_slots", as_index=False).agg(
            eligible_events=("event_id", "nunique"),
            mean_model_reduction=("model_reduction", "mean"),
            mean_oracle_efficiency=("oracle_efficiency", "mean"),
            total_oracle_top50_hits=("oracle_top50_hits", "sum"),
            total_dynamic_interventions=("dynamic_interventions", "sum"),
            total_unused_slots=("unused_slots", "sum"),
            total_dynamic_oracle_hits=("dynamic_oracle_top50_hits", "sum"),
            total_dynamic_blocked_impact=("dynamic_blocked_impact", "sum"),
            mean_dynamic_trigger_sec=("mean_dynamic_trigger_sec", "mean"),
        )

        print("[5/5] Writing diagnostic evidence...", flush=True)
        triggers.to_csv(output / "observable_dynamic_triggers.csv", index=False)
        metrics_frame.to_csv(output / "per_event_dynamic_frontier.csv", index=False)
        summary.to_csv(output / "dynamic_frontier_summary.csv", index=False)
        selections.to_csv(output / "selected_threads_by_reservation.csv", index=False)
        record.update(
            {
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "eligible_threads": len(scores),
                "triggered_threads": len(triggers),
                "summary": summary.to_dict(orient="records"),
                "output_files": [
                    "observable_dynamic_triggers.csv",
                    "per_event_dynamic_frontier.csv",
                    "dynamic_frontier_summary.csv",
                    "selected_threads_by_reservation.csv",
                ],
            }
        )
        write_record(output, record)
        print(
            f"SUCCESS: saved dynamic quiet-escalation frontier saved to: {output.resolve()}",
            flush=True,
        )
    except BaseException as error:
        record.update(
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        write_record(output, record)
        print(
            f"FAILURE: dynamic quiet-escalation frontier preserved at: {output.resolve()}",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
