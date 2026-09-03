"""Nested-LOEO, causally valid one-slot defer-and-monitor diagnostic.

At 30 minutes the policy intervenes on 49 of the baseline Top-50 threads and
defers one tail position.  The final slot is used by the first qualifying quiet
change point observed from 35--60 minutes, or by the deferred incumbent at 60
minutes when no challenger qualifies.  Outcomes after an action time are used
only for inner-event policy selection and held-out outer-event evaluation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

import run_strict30_quiet_change_point_qualification as change
from pheme_account_age_v5_common import load_graphs


HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parent / "experiments"
SOURCE_RUN = OUT_ROOT / "20260903_033243_667703_strict30_quiet_change_point_qualification"
CHECKPOINTS = SOURCE_RUN / "observable_checkpoint_features.csv"
SCORES = change.SCORES
TAIL_RANKS = (46, 47, 48, 49, 50)
BASELINE_BUDGET = 50
FALLBACK_SEC = 3600.0
MIN_INNER_NONWORSE_FRACTION = 2.0 / 3.0
MIN_RELATIVE_GAIN = 0.01
MIN_OUTER_NONWORSE_EVENTS = 5
SEED = 42


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def first_challenger(
    checkpoints: pd.DataFrame,
    quiet_ids: set[str],
    baseline_ids: set[str],
    rule: dict,
) -> pd.Series | None:
    triggers = change.first_triggers(checkpoints, quiet_ids - baseline_ids, rule)
    if triggers.empty:
        return None
    # Time is the policy priority.  thread_id is only a deterministic tie-break
    # when multiple threads first qualify at the same five-minute checkpoint.
    return triggers.sort_values(["checkpoint_sec", "thread_id"], kind="stable").iloc[0]


def evaluate_policy(
    event: pd.DataFrame,
    checkpoints: pd.DataFrame,
    quiet_ids: set[str],
    rule: dict,
    deferred_rank: int,
) -> dict:
    ranked = event.sort_values(
        ["hybrid_calibrated_score", "thread_id"],
        ascending=[False, True],
        kind="stable",
    )
    baseline = ranked.head(BASELINE_BUDGET).copy().reset_index(drop=True)
    if len(baseline) != BASELINE_BUDGET:
        raise ValueError(f"{event.event_id.iloc[0]}: fewer than 50 baseline candidates")
    deferred = baseline.iloc[deferred_rank - 1]
    baseline_ids = set(baseline.thread_id)
    challenger = first_challenger(checkpoints, quiet_ids, baseline_ids, rule)

    fallback_rows = checkpoints.loc[
        checkpoints.thread_id.eq(str(deferred.thread_id))
        & checkpoints.checkpoint_sec.eq(FALLBACK_SEC)
    ]
    if len(fallback_rows) != 1:
        raise ValueError(f"missing 60-minute fallback outcome for {deferred.thread_id}")
    fallback_impact = float(fallback_rows.iloc[0].dynamic_preventable_impact)

    if challenger is None:
        action = "deferred_incumbent_at_60"
        later_thread_id = str(deferred.thread_id)
        action_sec = FALLBACK_SEC
        later_impact = fallback_impact
    else:
        action = "quiet_challenger"
        later_thread_id = str(challenger.thread_id)
        action_sec = float(challenger.checkpoint_sec)
        later_impact = float(challenger.dynamic_preventable_impact)

    baseline_impact = float(baseline.preventable_impact.sum())
    policy_impact = baseline_impact - float(deferred.preventable_impact) + later_impact
    total_impact = float(event.preventable_impact.sum())
    return {
        "event_id": str(event.event_id.iloc[0]),
        "rule_id": rule["rule_id"],
        "deferred_rank": deferred_rank,
        "deferred_thread_id": str(deferred.thread_id),
        "deferred_full_impact": float(deferred.preventable_impact),
        "fallback_impact_at_60": fallback_impact,
        "action": action,
        "later_thread_id": later_thread_id,
        "action_sec": action_sec,
        "later_impact": later_impact,
        "baseline_blocked_impact": baseline_impact,
        "policy_blocked_impact": policy_impact,
        "delta": policy_impact - baseline_impact,
        "baseline_reduction": baseline_impact / total_impact if total_impact else 0.0,
        "policy_reduction": policy_impact / total_impact if total_impact else 0.0,
        "challenger_used": int(challenger is not None),
        "challenger_positive": int(challenger is not None and later_impact > 0),
    }


def choose_config(
    outer_event: str,
    eligible: list[str],
    all_events: set[str],
    scores: pd.DataFrame,
    checkpoints: pd.DataFrame,
    graph_frame: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    inner_events = [event for event in eligible if event != outer_event]
    rows: list[dict] = []
    for rule in change.RULES:
        for deferred_rank in TAIL_RANKS:
            folds = []
            for inner_event in inner_events:
                quiet_ids, _ = change.quiet_ids_for_fold(
                    graph_frame, all_events - {outer_event, inner_event}, inner_event
                )
                folds.append(
                    evaluate_policy(
                        scores.loc[scores.event_id.eq(inner_event)],
                        checkpoints.loc[checkpoints.event_id.eq(inner_event)],
                        quiet_ids,
                        rule,
                        deferred_rank,
                    )
                )
            rows.append(
                {
                    "outer_event": outer_event,
                    "rule_id": rule["rule_id"],
                    "deferred_rank": deferred_rank,
                    "inner_events": len(folds),
                    "mean_delta": float(np.mean([fold["delta"] for fold in folds])),
                    "total_delta": float(np.sum([fold["delta"] for fold in folds])),
                    "nonworse_events": int(np.sum([fold["delta"] >= 0 for fold in folds])),
                    "challenger_events": int(np.sum([fold["challenger_used"] for fold in folds])),
                    "positive_challenger_events": int(
                        np.sum([fold["challenger_positive"] for fold in folds])
                    ),
                }
            )
    selection = pd.DataFrame(rows).sort_values(
        ["mean_delta", "nonworse_events", "positive_challenger_events", "rule_id", "deferred_rank"],
        ascending=[False, False, False, True, True],
        kind="stable",
    )
    best = selection.iloc[0].to_dict()
    required_nonworse = int(np.ceil(len(inner_events) * MIN_INNER_NONWORSE_FRACTION))
    best["safe_to_deploy"] = bool(
        best["mean_delta"] > 0 and best["nonworse_events"] >= required_nonworse
    )
    selection["selected_for_outer"] = False
    selection.loc[selection.index[0], "selected_for_outer"] = True
    return best, selection


def main() -> None:
    output = OUT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_strict30_one_slot_defer_policy"
    )
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "started_at": utc_now(),
        "purpose": "causally valid nested-LOEO one-slot defer-and-monitor policy",
        "inputs": {"scores": str(SCORES.resolve()), "checkpoints": str(CHECKPOINTS.resolve())},
        "split": "seven outer LOEO events; six inner events jointly select rule and deferred tail rank",
        "cutoff_seconds": 1800,
        "monitor_checkpoints_seconds": list(change.CHECKPOINTS),
        "fallback_seconds": FALLBACK_SEC,
        "baseline_budget": BASELINE_BUDGET,
        "tail_ranks": list(TAIL_RANKS),
        "rule_count": len(change.RULES),
        "seed": SEED,
        "success_gate": {
            "minimum_relative_gain": MIN_RELATIVE_GAIN,
            "minimum_outer_nonworse_events": MIN_OUTER_NONWORSE_EVENTS,
        },
        "research_safety": (
            "At 30 minutes only 49 interventions occur. The held slot uses the first observable "
            "35--60 minute trigger, otherwise the deferred incumbent at 60. Benefits are counted "
            "only after each actual action time. Outer outcomes never select a configuration."
        ),
    }
    save_record(output, record)
    try:
        print("Starting strict-30 one-slot defer-and-monitor diagnostic.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/5] Loading held-out scores and audited checkpoint artifact...", flush=True)
        source_record = json.loads((SOURCE_RUN / "run_record.json").read_text(encoding="utf-8"))
        if source_record.get("status") != "complete":
            raise ValueError("source checkpoint run is not complete")
        scores = pd.read_csv(SCORES, dtype={"thread_id": str, "event_id": str})
        checkpoints = pd.read_csv(CHECKPOINTS, dtype={"thread_id": str, "event_id": str})
        if scores.thread_id.duplicated().any() or checkpoints.duplicated(["thread_id", "checkpoint_sec"]).any():
            raise ValueError("duplicate identities in source artifacts")

        print("[2/5] Reconstructing fold-safe quiet gates...", flush=True)
        graphs, all_event_list, eligible = load_graphs()
        graph_frame = pd.DataFrame(
            {
                "thread_id": [str(graph.thread_id) for graph in graphs],
                "event_id": [str(graph.event_id) for graph in graphs],
                "recency": change.recency_values(graphs),
            }
        )
        all_events = set(all_event_list)

        print("[3/5] Jointly selecting trigger rule and deferred tail position...", flush=True)
        selection_frames = []
        outer_rows = []
        chosen_rows = []
        for number, outer_event in enumerate(sorted(eligible), 1):
            print(f"  outer fold {number}/{len(eligible)}: {outer_event}", flush=True)
            chosen, selection = choose_config(
                outer_event, eligible, all_events, scores, checkpoints, graph_frame
            )
            selection_frames.append(selection)
            rule = next(rule for rule in change.RULES if rule["rule_id"] == chosen["rule_id"])
            quiet_ids, threshold = change.quiet_ids_for_fold(
                graph_frame, all_events - {outer_event}, outer_event
            )
            saved_quiet = set(scores.loc[scores.event_id.eq(outer_event) & scores.quiet, "thread_id"])
            if quiet_ids != saved_quiet:
                raise ValueError(f"{outer_event}: quiet gate differs from official hybrid")
            forced = evaluate_policy(
                scores.loc[scores.event_id.eq(outer_event)],
                checkpoints.loc[checkpoints.event_id.eq(outer_event)],
                quiet_ids,
                rule,
                int(chosen["deferred_rank"]),
            )
            forced["policy_mode"] = "forced_inner_best"
            outer_rows.append(forced)
            safe = forced.copy()
            safe["policy_mode"] = "safety_gated"
            if not chosen["safe_to_deploy"]:
                safe.update(
                    {
                        "action": "abstain_keep_baseline",
                        "later_thread_id": "",
                        "action_sec": np.nan,
                        "later_impact": 0.0,
                        "policy_blocked_impact": safe["baseline_blocked_impact"],
                        "delta": 0.0,
                        "policy_reduction": safe["baseline_reduction"],
                        "challenger_used": 0,
                        "challenger_positive": 0,
                    }
                )
            outer_rows.append(safe)
            chosen_rows.append(
                {
                    "outer_event": outer_event,
                    "quiet_threshold": threshold,
                    **chosen,
                }
            )

        print("[4/5] Aggregating honest sequential-policy outcomes...", flush=True)
        outer = pd.DataFrame(outer_rows)
        summary = outer.groupby("policy_mode", as_index=False).agg(
            eligible_events=("event_id", "nunique"),
            deployed_events=("action", lambda values: int(np.sum(values != "abstain_keep_baseline"))),
            challenger_events=("challenger_used", "sum"),
            positive_challenger_events=("challenger_positive", "sum"),
            total_delta=("delta", "sum"),
            mean_baseline_reduction=("baseline_reduction", "mean"),
            mean_policy_reduction=("policy_reduction", "mean"),
            nonworse_events=("delta", lambda values: int(np.sum(values >= 0))),
        )
        summary["relative_gain"] = summary.mean_policy_reduction / summary.mean_baseline_reduction - 1.0
        passing = summary.loc[
            summary.policy_mode.eq("safety_gated")
            & summary.relative_gain.ge(MIN_RELATIVE_GAIN)
            & summary.nonworse_events.ge(MIN_OUTER_NONWORSE_EVENTS)
        ]
        decision = {
            "passed": not passing.empty,
            "interpretation": (
                "Proceed to a separately validated one-slot sequential policy."
                if not passing.empty
                else "Do not deploy one-slot deferral; retain the immediate strict-30 Top-50 baseline."
            ),
        }

        print("[5/5] Writing complete run evidence...", flush=True)
        outer.to_csv(output / "outer_policy_outcomes.csv", index=False)
        summary.to_csv(output / "policy_summary.csv", index=False)
        pd.DataFrame(chosen_rows).to_csv(output / "outer_selected_configs.csv", index=False)
        pd.concat(selection_frames, ignore_index=True).to_csv(output / "inner_config_selection.csv", index=False)
        (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        record.update(
            {
                "status": "complete",
                "completed_at": utc_now(),
                "eligible_events": sorted(eligible),
                "selected_configs": chosen_rows,
                "summary": summary.to_dict(orient="records"),
                "decision": decision,
                "output_files": [
                    "outer_policy_outcomes.csv",
                    "policy_summary.csv",
                    "outer_selected_configs.csv",
                    "inner_config_selection.csv",
                    "decision.json",
                    "run_record.json",
                ],
            }
        )
        save_record(output, record)
        print(
            f"  decision: {'PASS' if decision['passed'] else 'FAIL'}; "
            f"safety-gated total delta={float(summary.loc[summary.policy_mode.eq('safety_gated'), 'total_delta'].iloc[0]):.1f}",
            flush=True,
        )
        print(f"SUCCESS: strict-30 one-slot defer policy saved to: {output.resolve()}", flush=True)
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
        print(f"FAILURE: one-slot defer policy preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
