"""Post-hoc sensitivity of the one-slot monitoring horizon.

The analysis is explicitly exploratory because horizon hypotheses were formed
after inspecting prior held-out outcomes.  Every tested horizon is reported.
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
HORIZONS = (2400.0, 2700.0, 3000.0, 3300.0, 3600.0)
TAIL_RANKS = (48, 49, 50)
RULES = (
    {
        "rule_id": "conservative_c3_r1",
        "min_last5_count": 3,
        "min_acceleration": 1.25,
        "min_reply_to_reply": 1,
        "min_depth_growth": 0,
        "min_parent_concentration": 0,
        "min_gap_compression": 0.0,
    },
    {
        "rule_id": "conservative_c3_r2",
        "min_last5_count": 3,
        "min_acceleration": 1.25,
        "min_reply_to_reply": 2,
        "min_depth_growth": 0,
        "min_parent_concentration": 0,
        "min_gap_compression": 0.0,
    },
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def evaluate(
    event: pd.DataFrame,
    checkpoints: pd.DataFrame,
    quiet_ids: set[str],
    rule: dict,
    rank: int,
    horizon: float,
) -> dict:
    ranked = event.sort_values(
        ["hybrid_calibrated_score", "thread_id"],
        ascending=[False, True], kind="stable",
    )
    baseline = ranked.head(50).reset_index(drop=True)
    deferred = baseline.iloc[rank - 1]
    baseline_ids = set(baseline.thread_id)
    observable = checkpoints.loc[checkpoints.checkpoint_sec.le(horizon)]
    challenger = defer.first_challenger(observable, quiet_ids, baseline_ids, rule)
    fallback = checkpoints.loc[
        checkpoints.thread_id.eq(str(deferred.thread_id))
        & checkpoints.checkpoint_sec.eq(horizon)
    ]
    if len(fallback) != 1:
        raise ValueError(f"missing fallback outcome for {deferred.thread_id} at {horizon}")
    if challenger is None:
        action = "deferred_incumbent"
        action_sec = horizon
        later_thread = str(deferred.thread_id)
        later_impact = float(fallback.iloc[0].dynamic_preventable_impact)
    else:
        action = "quiet_challenger"
        action_sec = float(challenger.checkpoint_sec)
        later_thread = str(challenger.thread_id)
        later_impact = float(challenger.dynamic_preventable_impact)
    baseline_impact = float(baseline.preventable_impact.sum())
    policy_impact = baseline_impact - float(deferred.preventable_impact) + later_impact
    total = float(event.preventable_impact.sum())
    return {
        "event_id": str(event.event_id.iloc[0]),
        "rule_id": rule["rule_id"],
        "deferred_rank": rank,
        "monitor_horizon_sec": horizon,
        "action": action,
        "action_sec": action_sec,
        "deferred_thread_id": str(deferred.thread_id),
        "later_thread_id": later_thread,
        "deferred_full_impact": float(deferred.preventable_impact),
        "later_impact": later_impact,
        "delta": policy_impact - baseline_impact,
        "baseline_reduction": baseline_impact / total if total else 0.0,
        "policy_reduction": policy_impact / total if total else 0.0,
        "challenger_used": int(challenger is not None),
        "challenger_positive": int(challenger is not None and later_impact > 0),
    }


def main() -> None:
    output = OUT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + "_strict30_monitor_horizon_sensitivity"
    )
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "started_at": now(),
        "purpose": "post-hoc one-slot monitor-horizon sensitivity",
        "confirmatory": False,
        "inputs": {"scores": str(SCORES.resolve()), "checkpoints": str(CHECKPOINTS.resolve())},
        "cutoff_seconds": 1800,
        "horizons_seconds": list(HORIZONS),
        "tail_ranks": list(TAIL_RANKS),
        "rules": list(RULES),
        "research_safety": (
            "All triggers are observable by their checkpoint and fallback benefit starts at the "
            "tested horizon. Outcomes are used only for explicitly post-hoc sensitivity."
        ),
    }
    save(output, record)
    try:
        print("Starting strict-30 monitor-horizon sensitivity diagnostic.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/4] Loading audited artifacts...", flush=True)
        scores = pd.read_csv(SCORES, dtype={"thread_id": str, "event_id": str})
        checkpoints = pd.read_csv(CHECKPOINTS, dtype={"thread_id": str, "event_id": str})
        print("[2/4] Reconstructing official quiet cohorts...", flush=True)
        graphs, all_event_list, eligible = load_graphs()
        graph_frame = pd.DataFrame({
            "thread_id": [str(g.thread_id) for g in graphs],
            "event_id": [str(g.event_id) for g in graphs],
            "recency": change.recency_values(graphs),
        })
        all_events = set(all_event_list)
        quiet = {}
        for event in sorted(eligible):
            ids, _ = change.quiet_ids_for_fold(graph_frame, all_events - {event}, event)
            if ids != set(scores.loc[scores.event_id.eq(event) & scores.quiet, "thread_id"]):
                raise ValueError(f"{event}: quiet cohort mismatch")
            quiet[event] = ids
        print("[3/4] Evaluating 30 horizon/rule/rank configurations...", flush=True)
        rows = []
        completed = 0
        for rule in RULES:
            for horizon in HORIZONS:
                for rank in TAIL_RANKS:
                    completed += 1
                    print(f"  configuration {completed}/30: {rule['rule_id']}, horizon={int(horizon)}, rank={rank}", flush=True)
                    for event in sorted(eligible):
                        rows.append(evaluate(
                            scores.loc[scores.event_id.eq(event)],
                            checkpoints.loc[checkpoints.event_id.eq(event)],
                            quiet[event], rule, rank, horizon,
                        ))
        print("[4/4] Aggregating full horizon curves and writing evidence...", flush=True)
        detail = pd.DataFrame(rows)
        summary = detail.groupby(
            ["rule_id", "monitor_horizon_sec", "deferred_rank"], as_index=False
        ).agg(
            eligible_events=("event_id", "nunique"),
            challenger_events=("challenger_used", "sum"),
            positive_challenger_events=("challenger_positive", "sum"),
            total_delta=("delta", "sum"),
            mean_baseline_reduction=("baseline_reduction", "mean"),
            mean_policy_reduction=("policy_reduction", "mean"),
            nonworse_events=("delta", lambda x: int((x >= 0).sum())),
            worst_event_delta=("delta", "min"),
        )
        summary["relative_gain"] = summary.mean_policy_reduction / summary.mean_baseline_reduction - 1.0
        summary.sort_values(
            ["total_delta", "nonworse_events", "rule_id", "monitor_horizon_sec", "deferred_rank"],
            ascending=[False, False, True, True, True], kind="stable", inplace=True,
        )
        horizon_curve = summary.groupby(
            ["rule_id", "monitor_horizon_sec"], as_index=False
        ).agg(
            minimum_delta_across_tail_ranks=("total_delta", "min"),
            maximum_delta_across_tail_ranks=("total_delta", "max"),
            minimum_nonworse_events=("nonworse_events", "min"),
            best_worst_event_delta=("worst_event_delta", "max"),
        )
        decision = {
            "confirmatory": False,
            "deployment_decision": "not_allowed_from_posthoc_sensitivity",
            "interpretation": "Pre-register a horizon before evaluation on genuinely new data.",
        }
        detail.to_csv(output / "per_event_horizon_sensitivity.csv", index=False)
        summary.to_csv(output / "configuration_summary.csv", index=False)
        horizon_curve.to_csv(output / "horizon_rank_robustness.csv", index=False)
        (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        record.update({
            "status": "complete",
            "completed_at": now(),
            "eligible_events": sorted(eligible),
            "best_descriptive_configuration": summary.iloc[0].to_dict(),
            "decision": decision,
            "output_files": ["per_event_horizon_sensitivity.csv", "configuration_summary.csv", "horizon_rank_robustness.csv", "decision.json", "run_record.json"],
        })
        save(output, record)
        best = summary.iloc[0]
        print(f"  descriptive best: {best.rule_id}, horizon={int(best.monitor_horizon_sec)}, rank={int(best.deferred_rank)}, total delta={float(best.total_delta):.1f}", flush=True)
        print(f"SUCCESS: monitor-horizon sensitivity saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": now(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(), "output_files": sorted(p.name for p in output.iterdir())})
        save(output, record)
        print(f"FAILURE: monitor-horizon sensitivity preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
