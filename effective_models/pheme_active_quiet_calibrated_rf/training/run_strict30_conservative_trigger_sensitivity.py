"""Post-hoc exploratory sensitivity for conservative quiet triggers.

This script is hypothesis generation, not confirmatory evaluation: the trigger
family was motivated by already-inspected outer-event outcomes.  It preserves
the causally valid one-slot defer policy and reports every tested combination.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import pandas as pd

import run_strict30_one_slot_defer_policy as defer
import run_strict30_quiet_change_point_qualification as change
from pheme_account_age_v5_common import load_graphs


HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parent / "experiments"
SCORES = change.SCORES
CHECKPOINTS = defer.CHECKPOINTS
TAIL_RANKS = (48, 49, 50)
SEED = 42

RULES = tuple(
    {
        "rule_id": f"conservative_c{count}_r{r2r}_d{depth}",
        "min_last5_count": count,
        "min_acceleration": 1.25,
        "min_reply_to_reply": r2r,
        "min_depth_growth": depth,
        "min_parent_concentration": 0,
        "min_gap_compression": 0.0,
    }
    for count in (3, 4, 5)
    for r2r in (1, 2)
    for depth in (0, 1)
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def main() -> None:
    output = OUT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + "_strict30_conservative_trigger_sensitivity"
    )
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "started_at": utc_now(),
        "purpose": "post-hoc exploratory conservative-trigger sensitivity",
        "confirmatory": False,
        "posthoc_reason": (
            "The count>=3 hypothesis was formed after inspecting held-out outer outcomes; "
            "results cannot be used as an unbiased generalization estimate."
        ),
        "inputs": {"scores": str(SCORES.resolve()), "checkpoints": str(CHECKPOINTS.resolve())},
        "split": "descriptive seven-event sensitivity; no configuration is selected for deployment",
        "cutoff_seconds": 1800,
        "monitor_checkpoints_seconds": list(change.CHECKPOINTS),
        "tail_ranks": list(TAIL_RANKS),
        "rules": list(RULES),
        "seed": SEED,
        "research_safety": (
            "Trigger features use only information observable at each checkpoint. Benefits are "
            "counted only after the actual intervention time. Future outcomes are used only for "
            "this explicitly post-hoc diagnostic, never as model inputs."
        ),
    }
    save_record(output, record)
    try:
        print("Starting strict-30 conservative-trigger sensitivity diagnostic.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/4] Loading audited held-out score and checkpoint artifacts...", flush=True)
        scores = pd.read_csv(SCORES, dtype={"thread_id": str, "event_id": str})
        checkpoints = pd.read_csv(CHECKPOINTS, dtype={"thread_id": str, "event_id": str})
        if scores.thread_id.duplicated().any():
            raise ValueError("duplicate score thread IDs")
        if checkpoints.duplicated(["thread_id", "checkpoint_sec"]).any():
            raise ValueError("duplicate checkpoint identities")

        print("[2/4] Reconstructing official fold-safe quiet cohorts...", flush=True)
        graphs, all_event_list, eligible = load_graphs()
        graph_frame = pd.DataFrame(
            {
                "thread_id": [str(graph.thread_id) for graph in graphs],
                "event_id": [str(graph.event_id) for graph in graphs],
                "recency": change.recency_values(graphs),
            }
        )
        all_events = set(all_event_list)
        quiet_by_event: dict[str, set[str]] = {}
        for event in sorted(eligible):
            quiet_ids, _ = change.quiet_ids_for_fold(graph_frame, all_events - {event}, event)
            saved = set(scores.loc[scores.event_id.eq(event) & scores.quiet, "thread_id"])
            if quiet_ids != saved:
                raise ValueError(f"{event}: quiet cohort differs from official hybrid")
            quiet_by_event[event] = quiet_ids

        print("[3/4] Evaluating all conservative rules without selecting a winner...", flush=True)
        rows = []
        total = len(RULES) * len(TAIL_RANKS)
        completed = 0
        for rule in RULES:
            for rank in TAIL_RANKS:
                completed += 1
                print(f"  configuration {completed}/{total}: {rule['rule_id']}, rank={rank}", flush=True)
                for event in sorted(eligible):
                    result = defer.evaluate_policy(
                        scores.loc[scores.event_id.eq(event)],
                        checkpoints.loc[checkpoints.event_id.eq(event)],
                        quiet_by_event[event],
                        rule,
                        rank,
                    )
                    result.update(
                        {
                            "min_last5_count": rule["min_last5_count"],
                            "min_reply_to_reply": rule["min_reply_to_reply"],
                            "min_depth_growth": rule["min_depth_growth"],
                        }
                    )
                    rows.append(result)

        print("[4/4] Aggregating robustness and writing complete evidence...", flush=True)
        detail = pd.DataFrame(rows)
        summary = detail.groupby(
            [
                "rule_id",
                "min_last5_count",
                "min_reply_to_reply",
                "min_depth_growth",
                "deferred_rank",
            ],
            as_index=False,
        ).agg(
            eligible_events=("event_id", "nunique"),
            challenger_events=("challenger_used", "sum"),
            positive_challenger_events=("challenger_positive", "sum"),
            total_delta=("delta", "sum"),
            mean_baseline_reduction=("baseline_reduction", "mean"),
            mean_policy_reduction=("policy_reduction", "mean"),
            nonworse_events=("delta", lambda values: int((values >= 0).sum())),
            worst_event_delta=("delta", "min"),
            best_event_delta=("delta", "max"),
        )
        summary["relative_gain"] = (
            summary.mean_policy_reduction / summary.mean_baseline_reduction - 1.0
        )
        summary.sort_values(
            ["total_delta", "nonworse_events", "rule_id", "deferred_rank"],
            ascending=[False, False, True, True],
            kind="stable",
            inplace=True,
        )
        robust = (
            summary.groupby("rule_id", as_index=False)
            .agg(
                tested_tail_ranks=("deferred_rank", "nunique"),
                minimum_total_delta_across_ranks=("total_delta", "min"),
                maximum_total_delta_across_ranks=("total_delta", "max"),
                minimum_nonworse_events=("nonworse_events", "min"),
                maximum_worst_event_delta=("worst_event_delta", "max"),
            )
            .sort_values(
                ["minimum_total_delta_across_ranks", "minimum_nonworse_events", "rule_id"],
                ascending=[False, False, True],
                kind="stable",
            )
        )
        decision = {
            "confirmatory": False,
            "deployment_decision": "not_allowed_from_posthoc_sensitivity",
            "interpretation": (
                "Use the sensitivity only to pre-register a hypothesis for genuinely new data; "
                "do not replace the strict-30 baseline from these results."
            ),
        }
        detail.to_csv(output / "per_event_sensitivity.csv", index=False)
        summary.to_csv(output / "configuration_summary.csv", index=False)
        robust.to_csv(output / "rule_rank_robustness.csv", index=False)
        (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        record.update(
            {
                "status": "complete",
                "completed_at": utc_now(),
                "eligible_events": sorted(eligible),
                "best_descriptive_configuration": summary.iloc[0].to_dict(),
                "most_rank_robust_descriptive_rule": robust.iloc[0].to_dict(),
                "decision": decision,
                "output_files": [
                    "per_event_sensitivity.csv",
                    "configuration_summary.csv",
                    "rule_rank_robustness.csv",
                    "decision.json",
                    "run_record.json",
                ],
            }
        )
        save_record(output, record)
        print(
            "  descriptive best: "
            f"{summary.iloc[0].rule_id}, rank={int(summary.iloc[0].deferred_rank)}, "
            f"total delta={float(summary.iloc[0].total_delta):.1f}",
            flush=True,
        )
        print(f"SUCCESS: conservative-trigger sensitivity saved to: {output.resolve()}", flush=True)
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
        print(f"FAILURE: conservative-trigger sensitivity preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
