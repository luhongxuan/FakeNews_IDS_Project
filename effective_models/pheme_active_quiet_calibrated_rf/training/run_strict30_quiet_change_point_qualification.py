"""Nested-LOEO 30--60 minute binary change-point qualification policy.

Quiet threads are checked every five minutes.  A train-selected binary rule
admits every qualifying thread; quiet threads are never mutually ranked.
Post-trigger descendants are used only to evaluate blocked impact.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from pheme_account_age_v5_common import load_graphs
from pheme_quiet_expert_common import QUIET_QUANTILE, recency_values
from run_saved_dynamic_quiet_escalation_frontier import preventable_after


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT_ROOT = HERE.parent / "experiments"
SCORE_RUN = OUT_ROOT / "20260902_132700_413954_quiet_tier_hybrid_policy_full"
SCORES = SCORE_RUN / "outer_scores.csv"
REPLIES = (
    ROOT
    / "data"
    / "protected_research_assets"
    / "pheme_v5_strict30"
    / "pheme_reply_level_v4.csv"
)

START_SEC = 1800.0
CHECKPOINTS = tuple(float(value) for value in range(2100, 3601, 300))
TARGET_MEAN_ADMISSIONS = (1, 2, 3, 5, 10)
BASELINE_BUDGET = 50
SEED = 42
MIN_RELATIVE_EQUAL_BUDGET_GAIN = 0.10
MIN_NONWORSE_EVENTS = 5

# Each rule is fixed before outer evaluation.  The inner folds choose one rule
# per target mean admission count; they never choose individual threads.
RULES = tuple(
    {
        "rule_id": f"c{count}_a{accel}_r{r2r}_d{depth}_p{parent}_g{gap}",
        "min_last5_count": count,
        "min_acceleration": accel,
        "min_reply_to_reply": r2r,
        "min_depth_growth": depth,
        "min_parent_concentration": parent,
        "min_gap_compression": gap,
    }
    for count in (1, 2, 3)
    for accel in (1.25, 1.75, 2.5)
    for r2r in (0, 1)
    for depth in (0, 1)
    for parent in (0, 2)
    for gap in (0.0, 1.5)
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def checkpoint_features(group: pd.DataFrame) -> list[dict]:
    source = group.loc[group.is_source.eq(1)]
    if len(source) != 1:
        raise ValueError(f"{group.thread_id.iloc[0]}: expected one source row")
    source_id = str(source.iloc[0].tweet_id)
    replies = group.loc[group.is_source.ne(1)].sort_values(
        ["offset_sec", "tweet_id"], kind="stable"
    )
    parents = {
        str(row.tweet_id): None if pd.isna(row.parent_id) else str(row.parent_id)
        for row in group.itertuples(index=False)
    }
    offsets = {str(row.tweet_id): float(row.offset_sec) for row in group.itertuples(index=False)}
    rows = []
    for checkpoint in CHECKPOINTS:
        last5 = replies.loc[
            replies.offset_sec.gt(checkpoint - 300) & replies.offset_sec.le(checkpoint)
        ]
        prev5 = replies.loc[
            replies.offset_sec.gt(checkpoint - 600)
            & replies.offset_sec.le(checkpoint - 300)
        ]
        observed_before = replies.loc[replies.offset_sec.le(checkpoint - 300)]
        last_count = len(last5)
        previous_count = len(prev5)
        acceleration = (last_count + 1.0) / (previous_count + 1.0)
        recent_parents = last5.parent_id.fillna(source_id).astype(str)
        reply_to_reply = int(recent_parents.ne(source_id).sum())
        previous_depth = float(observed_before.depth.max()) if len(observed_before) else 0.0
        recent_depth = float(last5.depth.max()) if len(last5) else previous_depth
        depth_growth = max(0.0, recent_depth - previous_depth)
        parent_concentration = int(recent_parents.value_counts().max()) if len(last5) else 0

        recent10 = replies.loc[
            replies.offset_sec.gt(checkpoint - 600) & replies.offset_sec.le(checkpoint)
        ].offset_sec.to_numpy(float)
        earlier = replies.loc[
            replies.offset_sec.le(checkpoint - 600)
        ].offset_sec.to_numpy(float)
        recent_gap = float(np.median(np.diff(recent10))) if len(recent10) >= 2 else np.nan
        earlier_gap = float(np.median(np.diff(earlier[-6:]))) if len(earlier) >= 2 else np.nan
        gap_compression = (
            (earlier_gap + 1.0) / (recent_gap + 1.0)
            if np.isfinite(recent_gap) and np.isfinite(earlier_gap)
            else 0.0
        )
        rows.append(
            {
                "thread_id": str(group.thread_id.iloc[0]),
                "event_id": str(group.event_id.iloc[0]),
                "checkpoint_sec": checkpoint,
                "last5_count": last_count,
                "prev5_count": previous_count,
                "acceleration": acceleration,
                "reply_to_reply_last5": reply_to_reply,
                "depth_growth_last5": depth_growth,
                "parent_concentration_last5": parent_concentration,
                "gap_compression": gap_compression,
                "dynamic_preventable_impact": preventable_after(parents, offsets, checkpoint),
            }
        )
    return rows


def build_checkpoint_table(raw: pd.DataFrame, eligible_ids: set[str]) -> pd.DataFrame:
    rows = []
    filtered = raw.loc[raw.thread_id.isin(eligible_ids)].copy()
    groups = list(filtered.groupby("thread_id", sort=True))
    for index, (_, group) in enumerate(groups, 1):
        rows.extend(checkpoint_features(group))
        if index % 300 == 0 or index == len(groups):
            print(f"  checkpoint feature/impact audit {index}/{len(groups)}", flush=True)
    result = pd.DataFrame(rows)
    if result.duplicated(["thread_id", "checkpoint_sec"]).any():
        raise ValueError("duplicate checkpoint identity")
    numeric = result.drop(columns=["thread_id", "event_id"]).to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("non-finite checkpoint feature")
    return result


def first_triggers(checkpoints: pd.DataFrame, quiet_ids: set[str], rule: dict) -> pd.DataFrame:
    candidates = checkpoints.loc[checkpoints.thread_id.isin(quiet_ids)].copy()
    qualified = candidates.loc[
        candidates.last5_count.ge(rule["min_last5_count"])
        & candidates.acceleration.ge(rule["min_acceleration"])
        & candidates.reply_to_reply_last5.ge(rule["min_reply_to_reply"])
        & candidates.depth_growth_last5.ge(rule["min_depth_growth"])
        & candidates.parent_concentration_last5.ge(rule["min_parent_concentration"])
        & candidates.gap_compression.ge(rule["min_gap_compression"])
    ]
    return (
        qualified.sort_values(["checkpoint_sec", "thread_id"], kind="stable")
        .drop_duplicates("thread_id", keep="first")
        .copy()
    )


def quiet_ids_for_fold(
    graph_frame: pd.DataFrame, reference_events: set[str], query_event: str
) -> tuple[set[str], float]:
    reference = graph_frame.loc[graph_frame.event_id.isin(reference_events)]
    threshold = float(np.quantile(reference.recency, QUIET_QUANTILE))
    query = graph_frame.loc[graph_frame.event_id.eq(query_event)]
    return set(query.loc[query.recency >= threshold, "thread_id"]), threshold


def rule_stats(triggers: pd.DataFrame) -> dict:
    return {
        "admissions": len(triggers),
        "blocked_impact": float(triggers.dynamic_preventable_impact.sum()),
        "positive_admissions": int((triggers.dynamic_preventable_impact > 0).sum()),
        "mean_trigger_sec": float(triggers.checkpoint_sec.mean()) if len(triggers) else np.nan,
    }


def choose_rules(
    outer_event: str,
    eligible: list[str],
    all_events: set[str],
    graph_frame: pd.DataFrame,
    checkpoints: pd.DataFrame,
) -> tuple[dict[int, dict | None], pd.DataFrame]:
    rows = []
    inner_events = [event for event in eligible if event != outer_event]
    for rule in RULES:
        folds = []
        for inner_event in inner_events:
            references = all_events - {outer_event, inner_event}
            quiet_ids, _ = quiet_ids_for_fold(graph_frame, references, inner_event)
            triggers = first_triggers(
                checkpoints.loc[checkpoints.event_id.eq(inner_event)], quiet_ids, rule
            )
            folds.append(rule_stats(triggers))
        finite_trigger_times = [
            fold["mean_trigger_sec"]
            for fold in folds
            if np.isfinite(fold["mean_trigger_sec"])
        ]
        rows.append(
            {
                "outer_event": outer_event,
                **rule,
                "inner_events": len(folds),
                "mean_admissions": float(np.mean([fold["admissions"] for fold in folds])),
                "mean_blocked_impact": float(np.mean([fold["blocked_impact"] for fold in folds])),
                "mean_positive_admissions": float(np.mean([fold["positive_admissions"] for fold in folds])),
                "mean_trigger_sec": (
                    float(np.mean(finite_trigger_times)) if finite_trigger_times else np.nan
                ),
            }
        )
    selection = pd.DataFrame(rows)
    chosen = {}
    selection["selected_for_caps"] = ""
    for cap in TARGET_MEAN_ADMISSIONS:
        feasible = selection.loc[
            selection.mean_admissions.gt(0) & selection.mean_admissions.le(cap)
        ].copy()
        if feasible.empty:
            chosen[cap] = None
            continue
        feasible.sort_values(
            ["mean_blocked_impact", "mean_positive_admissions", "mean_admissions", "rule_id"],
            ascending=[False, False, True, True],
            kind="stable",
            inplace=True,
        )
        rule_id = str(feasible.iloc[0].rule_id)
        chosen[cap] = next(rule for rule in RULES if rule["rule_id"] == rule_id)
        index = selection.index[selection.rule_id.eq(rule_id)][0]
        previous = selection.at[index, "selected_for_caps"]
        selection.at[index, "selected_for_caps"] = f"{previous},{cap}".strip(",")
    return chosen, selection


def evaluate_outer(
    event: pd.DataFrame,
    checkpoints: pd.DataFrame,
    quiet_ids: set[str],
    rule: dict | None,
    target_cap: int,
) -> tuple[dict, pd.DataFrame]:
    ranked = event.sort_values(
        ["hybrid_calibrated_score", "thread_id"],
        ascending=[False, True], kind="stable",
    )
    baseline = ranked.head(BASELINE_BUDGET).copy()
    baseline_ids = set(baseline.thread_id)
    available_quiet = quiet_ids - baseline_ids
    triggers = (
        first_triggers(checkpoints, available_quiet, rule)
        if rule is not None
        else checkpoints.head(0).copy()
    )
    trigger_ids = set(triggers.thread_id)
    if trigger_ids & baseline_ids:
        raise AssertionError("dynamic trigger overlaps strict-30 baseline")

    dynamic_count = len(triggers)
    baseline_blocked = float(baseline.preventable_impact.sum())
    dynamic_blocked = float(triggers.dynamic_preventable_impact.sum())
    total_impact = float(event.preventable_impact.sum())
    removed = baseline.tail(min(dynamic_count, BASELINE_BUDGET))
    equal_budget_feasible = dynamic_count <= BASELINE_BUDGET
    equal_budget_blocked = (
        baseline_blocked - float(removed.preventable_impact.sum()) + dynamic_blocked
        if equal_budget_feasible
        else np.nan
    )
    oracle_ids = set(
        event.sort_values(
            ["preventable_impact", "thread_id"],
            ascending=[False, True], kind="stable",
        ).head(BASELINE_BUDGET).thread_id
    )
    metrics = {
        "event_id": str(event.event_id.iloc[0]),
        "target_mean_admissions": target_cap,
        "rule_id": rule["rule_id"] if rule else "none",
        "baseline_interventions": len(baseline),
        "dynamic_interventions": dynamic_count,
        "dynamic_positive_interventions": int((triggers.dynamic_preventable_impact > 0).sum()),
        "baseline_blocked_impact": baseline_blocked,
        "dynamic_blocked_impact": dynamic_blocked,
        "addon_blocked_impact": baseline_blocked + dynamic_blocked,
        "equal_budget_blocked_impact": equal_budget_blocked,
        "baseline_reduction": baseline_blocked / total_impact if total_impact else 0.0,
        "addon_reduction": (baseline_blocked + dynamic_blocked) / total_impact if total_impact else 0.0,
        "equal_budget_reduction": equal_budget_blocked / total_impact if total_impact and equal_budget_feasible else np.nan,
        "equal_budget_delta": equal_budget_blocked - baseline_blocked if equal_budget_feasible else np.nan,
        "dynamic_impact_per_intervention": dynamic_blocked / dynamic_count if dynamic_count else 0.0,
        "dynamic_oracle_top50_hits": len(trigger_ids & oracle_ids),
        "mean_trigger_sec": float(triggers.checkpoint_sec.mean()) if dynamic_count else np.nan,
    }
    selected = triggers.copy()
    selected["target_mean_admissions"] = target_cap
    selected["rule_id"] = metrics["rule_id"]
    selected["oracle_top50"] = selected.thread_id.isin(oracle_ids)
    return metrics, selected


def main() -> None:
    output = OUT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + "_strict30_quiet_change_point_qualification"
    )
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "started_at": utc_now(),
        "purpose": "nested-LOEO binary 35--60 minute quiet change-point qualification",
        "scores": str(SCORES.resolve()),
        "reply_timeline": str(REPLIES.resolve()),
        "split": "seven eligible outer LOEO; inner events choose one binary rule per target mean admission count",
        "cutoff_seconds": 1800,
        "checkpoints_seconds": list(CHECKPOINTS),
        "target_mean_admissions": list(TARGET_MEAN_ADMISSIONS),
        "baseline_budget": BASELINE_BUDGET,
        "rule_count": len(RULES),
        "seed": SEED,
        "success_gate": {
            "minimum_relative_equal_budget_gain": MIN_RELATIVE_EQUAL_BUDGET_GAIN,
            "minimum_nonworse_events": MIN_NONWORSE_EVENTS,
        },
        "research_safety": (
            "At checkpoint t, trigger features use only replies with offset<=t. Outer labels never choose a rule. "
            "Post-checkpoint descendants and preventable impact are outcomes only. Quiet gates use reference-event recency only."
        ),
    }
    write_record(output, record)
    try:
        print("Starting strict-30 quiet change-point qualification diagnostic.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/6] Loading saved held-out scores and protected graph recency...", flush=True)
        scores = pd.read_csv(SCORES, dtype={"thread_id": str, "event_id": str})
        required = {"thread_id", "event_id", "preventable_impact", "hybrid_calibrated_score", "quiet"}
        if not required.issubset(scores) or scores.thread_id.duplicated().any():
            raise ValueError("invalid saved score artifact")
        graphs, all_event_list, eligible = load_graphs()
        graph_frame = pd.DataFrame(
            {
                "thread_id": [str(graph.thread_id) for graph in graphs],
                "event_id": [str(graph.event_id) for graph in graphs],
                "recency": recency_values(graphs),
            }
        )
        if sorted(scores.event_id.unique()) != sorted(eligible):
            raise ValueError("saved-score eligible events differ from graph artifact")

        print("[2/6] Loading full reply timelines with a feature/outcome firewall...", flush=True)
        raw = pd.read_csv(
            REPLIES,
            dtype={"thread_id": str, "tweet_id": str, "parent_id": str, "event_id": str},
            usecols=["thread_id", "tweet_id", "parent_id", "is_source", "depth", "offset_sec", "event_id"],
            low_memory=False,
        )
        if raw.offset_sec.isna().any() or (raw.offset_sec < 0).any():
            raise ValueError("reply timeline has invalid offsets")

        print("[3/6] Materializing five-minute observable change points and trigger-time outcomes...", flush=True)
        checkpoints = build_checkpoint_table(raw, set(scores.thread_id))

        print("[4/6] Selecting binary rules inside each outer fold...", flush=True)
        all_events = set(all_event_list)
        selection_frames = []
        metric_rows = []
        trigger_rows = []
        fold_details = []
        for outer_number, outer_event in enumerate(sorted(eligible), 1):
            print(f"  outer fold {outer_number}/{len(eligible)}: {outer_event}", flush=True)
            chosen, selection = choose_rules(
                outer_event, eligible, all_events, graph_frame, checkpoints
            )
            selection_frames.append(selection)
            quiet_ids, threshold = quiet_ids_for_fold(
                graph_frame, all_events - {outer_event}, outer_event
            )
            saved_quiet = set(scores.loc[scores.event_id.eq(outer_event) & scores.quiet, "thread_id"])
            if quiet_ids != saved_quiet:
                raise ValueError(f"{outer_event}: recomputed quiet gate differs from official hybrid")
            event = scores.loc[scores.event_id.eq(outer_event)].copy()
            event_checkpoints = checkpoints.loc[checkpoints.event_id.eq(outer_event)]
            fold_details.append(
                {"outer_event": outer_event, "quiet_threshold": threshold, "quiet_threads": len(quiet_ids)}
            )
            for cap in TARGET_MEAN_ADMISSIONS:
                metrics, triggers = evaluate_outer(
                    event, event_checkpoints, quiet_ids, chosen[cap], cap
                )
                metric_rows.append(metrics)
                trigger_rows.append(triggers)

        print("[5/6] Aggregating intervention-count and blocked-impact frontiers...", flush=True)
        metrics = pd.DataFrame(metric_rows)
        triggers = pd.concat(trigger_rows, ignore_index=True) if trigger_rows else pd.DataFrame()
        summary = metrics.groupby("target_mean_admissions", as_index=False).agg(
            eligible_events=("event_id", "nunique"),
            total_dynamic_interventions=("dynamic_interventions", "sum"),
            total_dynamic_positive_interventions=("dynamic_positive_interventions", "sum"),
            total_dynamic_blocked_impact=("dynamic_blocked_impact", "sum"),
            mean_dynamic_impact_per_intervention=("dynamic_impact_per_intervention", "mean"),
            mean_baseline_reduction=("baseline_reduction", "mean"),
            mean_addon_reduction=("addon_reduction", "mean"),
            mean_equal_budget_reduction=("equal_budget_reduction", "mean"),
            total_equal_budget_delta=("equal_budget_delta", "sum"),
            total_dynamic_oracle_top50_hits=("dynamic_oracle_top50_hits", "sum"),
            mean_trigger_sec=("mean_trigger_sec", "mean"),
        )
        summary["relative_equal_budget_gain"] = (
            summary.mean_equal_budget_reduction / summary.mean_baseline_reduction - 1.0
        )
        nonworse = (
            metrics.assign(nonworse=metrics.equal_budget_delta.ge(0))
            .groupby("target_mean_admissions", as_index=False)
            .nonworse.sum()
            .rename(columns={"nonworse": "nonworse_events"})
        )
        summary = summary.merge(nonworse, on="target_mean_admissions", validate="one_to_one")
        passing = summary.loc[
            summary.relative_equal_budget_gain.ge(MIN_RELATIVE_EQUAL_BUDGET_GAIN)
            & summary.nonworse_events.ge(MIN_NONWORSE_EVENTS)
        ]
        decision = {
            "passed": not passing.empty,
            "passing_target_mean_admissions": passing.target_mean_admissions.astype(int).tolist(),
            "best_relative_equal_budget_gain": float(summary.relative_equal_budget_gain.max()),
            "best_total_equal_budget_delta": float(summary.total_equal_budget_delta.max()),
            "interpretation": (
                "Proceed to a separately validated sequential policy experiment."
                if not passing.empty
                else "Do not replace the strict-30 baseline with this change-point qualification policy."
            ),
        }

        print("[6/6] Writing complete diagnostic evidence...", flush=True)
        checkpoints.to_csv(output / "observable_checkpoint_features.csv", index=False)
        pd.concat(selection_frames, ignore_index=True).to_csv(output / "inner_rule_selection.csv", index=False)
        metrics.to_csv(output / "per_event_policy_metrics.csv", index=False)
        summary.to_csv(output / "policy_frontier_summary.csv", index=False)
        triggers.to_csv(output / "outer_dynamic_triggers.csv", index=False)
        (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        record.update(
            {
                "status": "complete",
                "completed_at": utc_now(),
                "eligible_events": sorted(eligible),
                "fold_details": fold_details,
                "summary": summary.to_dict(orient="records"),
                "decision": decision,
                "output_files": [
                    "observable_checkpoint_features.csv",
                    "inner_rule_selection.csv",
                    "per_event_policy_metrics.csv",
                    "policy_frontier_summary.csv",
                    "outer_dynamic_triggers.csv",
                    "decision.json",
                    "run_record.json",
                ],
            }
        )
        write_record(output, record)
        print(
            f"  decision: {'PASS' if decision['passed'] else 'FAIL'}; "
            f"best equal-budget relative gain={decision['best_relative_equal_budget_gain']:.3%}",
            flush=True,
        )
        print(f"SUCCESS: strict-30 quiet change-point qualification saved to: {output.resolve()}", flush=True)
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
        write_record(output, record)
        print(f"FAILURE: quiet change-point qualification preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
