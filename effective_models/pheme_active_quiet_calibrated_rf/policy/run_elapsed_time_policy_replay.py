"""Replay the selected intervention policy at a user-supplied elapsed time.

This is a historical OOF replay, not a deployable production predictor.  The
base ranking and quiet cohort are frozen strict-30 held-out outputs.  After 30
minutes, the policy state changes using only replies observable by the supplied
elapsed time.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL_ROOT = HERE.parent
EXPERIMENTS = MODEL_ROOT / "experiments"
SCORE_RUN = EXPERIMENTS / "20260902_132700_413954_quiet_tier_hybrid_policy_full"
SCORES = SCORE_RUN / "outer_scores.csv"
REPLIES = (
    ROOT
    / "data"
    / "protected_research_assets"
    / "pheme_v5_strict30"
    / "pheme_reply_level_v4.csv"
)

BASE_CUTOFF_SEC = 1800.0
MONITOR_START_SEC = 2100.0
MONITOR_END_SEC = 2400.0
CHECKPOINT_STEP_SEC = 300.0
BUDGET = 50
HELD_SLOTS = 1
MIN_LAST5_COUNT = 3
MIN_ACCELERATION = 1.25
MIN_REPLY_TO_REPLY = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the strict-30 adaptive intervention policy at the current elapsed time."
    )
    parser.add_argument("--event", required=True, help="PHEME event ID, for example sydneysiege")
    parser.add_argument(
        "--elapsed-minutes",
        required=True,
        type=float,
        help="Minutes elapsed since the source post was created.",
    )
    return parser.parse_args()


def save_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def observable_trigger_rows(
    raw: pd.DataFrame,
    candidate_ids: set[str],
    elapsed_sec: float,
) -> pd.DataFrame:
    checkpoints = np.arange(
        MONITOR_START_SEC,
        min(elapsed_sec, MONITOR_END_SEC) + 1e-9,
        CHECKPOINT_STEP_SEC,
    )
    rows: list[dict] = []
    for thread_id, group in raw.loc[raw.thread_id.isin(candidate_ids)].groupby(
        "thread_id", sort=True
    ):
        source = group.loc[group.is_source.eq(1)]
        if len(source) != 1:
            raise ValueError(f"{thread_id}: expected exactly one source row")
        source_id = str(source.iloc[0].tweet_id)
        replies = group.loc[group.is_source.ne(1)].copy()
        for checkpoint in checkpoints:
            last5 = replies.loc[
                replies.offset_sec.gt(checkpoint - CHECKPOINT_STEP_SEC)
                & replies.offset_sec.le(checkpoint)
            ]
            prev5 = replies.loc[
                replies.offset_sec.gt(checkpoint - 2 * CHECKPOINT_STEP_SEC)
                & replies.offset_sec.le(checkpoint - CHECKPOINT_STEP_SEC)
            ]
            last_count = len(last5)
            prev_count = len(prev5)
            acceleration = (last_count + 1.0) / (prev_count + 1.0)
            recent_parents = last5.parent_id.fillna(source_id).astype(str)
            reply_to_reply = int(recent_parents.ne(source_id).sum())
            rows.append(
                {
                    "thread_id": str(thread_id),
                    "checkpoint_sec": float(checkpoint),
                    "last5_count": last_count,
                    "prev5_count": prev_count,
                    "acceleration": acceleration,
                    "reply_to_reply_last5": reply_to_reply,
                    "qualified": bool(
                        last_count >= MIN_LAST5_COUNT
                        and acceleration >= MIN_ACCELERATION
                        and reply_to_reply >= MIN_REPLY_TO_REPLY
                    ),
                }
            )
    return pd.DataFrame(rows)


def policy_state(
    event_scores: pd.DataFrame,
    raw: pd.DataFrame,
    elapsed_sec: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    ranked = event_scores.sort_values(
        ["hybrid_calibrated_score", "thread_id"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    if len(ranked) < BUDGET:
        raise ValueError("event has fewer than 50 candidate threads")
    baseline = ranked.head(BUDGET).copy()
    immediate = baseline.head(BUDGET - HELD_SLOTS).copy()
    deferred = baseline.iloc[BUDGET - 1]

    if elapsed_sec < BASE_CUTOFF_SEC:
        state = "waiting_for_strict30_base_ranking"
        selected = baseline.head(0).copy()
        audit = pd.DataFrame()
        challenger_id = None
    else:
        candidate_ids = set(
            ranked.loc[ranked.quiet & ~ranked.thread_id.isin(set(baseline.thread_id)), "thread_id"]
        )
        audit = observable_trigger_rows(raw, candidate_ids, elapsed_sec)
        qualified = (
            audit.loc[audit.qualified]
            .sort_values(["checkpoint_sec", "thread_id"], kind="stable")
            .drop_duplicates("thread_id", keep="first")
            if not audit.empty
            else audit
        )
        if not qualified.empty:
            challenger_id = str(qualified.iloc[0].thread_id)
            challenger = ranked.loc[ranked.thread_id.eq(challenger_id)]
            selected = pd.concat([immediate, challenger], ignore_index=True)
            state = "dynamic_quiet_challenger_selected"
        elif elapsed_sec >= MONITOR_END_SEC:
            challenger_id = None
            selected = baseline.copy()
            state = "monitor_closed_deferred_incumbent_selected"
        else:
            challenger_id = None
            selected = immediate.copy()
            state = "monitoring_one_slot_held"

    selected = selected.copy()
    selected["intervention_role"] = "strict30_immediate"
    if len(selected) == BUDGET:
        final_id = str(selected.iloc[-1].thread_id)
        selected.loc[selected.thread_id.eq(final_id), "intervention_role"] = (
            "dynamic_quiet_challenger"
            if challenger_id is not None
            else "deferred_incumbent"
        )
    info = {
        "state": state,
        "elapsed_minutes": elapsed_sec / 60.0,
        "base_cutoff_minutes": BASE_CUTOFF_SEC / 60.0,
        "monitor_start_minutes": MONITOR_START_SEC / 60.0,
        "monitor_end_minutes": MONITOR_END_SEC / 60.0,
        "selected_interventions": len(selected),
        "held_slots": BUDGET - len(selected),
        "deferred_thread_id": str(deferred.thread_id),
        "challenger_thread_id": challenger_id,
        "trigger": {
            "min_last5_count": MIN_LAST5_COUNT,
            "min_acceleration": MIN_ACCELERATION,
            "min_reply_to_reply_last5": MIN_REPLY_TO_REPLY,
        },
    }
    return info, selected, audit


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.elapsed_minutes) or args.elapsed_minutes < 0:
        raise ValueError("--elapsed-minutes must be a finite non-negative number")
    output = EXPERIMENTS / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_elapsed_time_policy_replay"
    )
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "started_at": utc_now(),
        "mode": "historical_oof_replay_only",
        "event": args.event,
        "elapsed_minutes": args.elapsed_minutes,
        "inputs": {"scores": str(SCORES.resolve()), "replies": str(REPLIES.resolve())},
        "policy": "strict30 quiet-tier Top-50 with one slot monitored until minute 40",
        "research_safety": (
            "The base ranking is a saved held-out strict-30 artifact. Dynamic trigger features "
            "read only reply rows observable by the requested elapsed time and never read outcome columns."
        ),
    }
    save_record(output, record)
    try:
        print("Starting elapsed-time intervention policy replay.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/4] Loading frozen strict-30 held-out ranking...", flush=True)
        scores = pd.read_csv(SCORES, dtype={"thread_id": str, "event_id": str})
        available = sorted(scores.event_id.unique())
        if args.event not in available:
            raise ValueError(f"unknown event {args.event!r}; choose one of {available}")
        event_scores = scores.loc[scores.event_id.eq(args.event)].copy()

        print("[2/4] Loading only observable reply fields for the requested event...", flush=True)
        raw = pd.read_csv(
            REPLIES,
            usecols=["thread_id", "tweet_id", "parent_id", "is_source", "offset_sec", "event_id"],
            dtype={"thread_id": str, "tweet_id": str, "parent_id": str, "event_id": str},
            low_memory=False,
        )
        raw = raw.loc[raw.event_id.eq(args.event) & raw.offset_sec.le(args.elapsed_minutes * 60.0)].copy()
        if raw.offset_sec.isna().any() or (raw.offset_sec < 0).any():
            raise ValueError("observable reply rows contain invalid offsets")

        print("[3/4] Computing the policy state at the supplied elapsed time...", flush=True)
        state, selected, audit = policy_state(
            event_scores, raw, args.elapsed_minutes * 60.0
        )
        print(
            f"  state={state['state']}; selected={state['selected_interventions']}; "
            f"held={state['held_slots']}",
            flush=True,
        )

        print("[4/4] Writing replay decision and audit evidence...", flush=True)
        selected.to_csv(output / "selected_interventions.csv", index=False)
        audit.to_csv(output / "observable_trigger_audit.csv", index=False)
        (output / "policy_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        record.update(
            {
                "status": "complete",
                "completed_at": utc_now(),
                "state": state,
                "output_files": [
                    "selected_interventions.csv",
                    "observable_trigger_audit.csv",
                    "policy_state.json",
                    "run_record.json",
                ],
            }
        )
        save_record(output, record)
        print(f"SUCCESS: elapsed-time policy replay saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update(
            {
                "status": "failed",
                "failed_at": utc_now(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "output_files": sorted(path.name for path in output.iterdir()),
            }
        )
        save_record(output, record)
        print(f"FAILURE: elapsed-time policy replay preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
