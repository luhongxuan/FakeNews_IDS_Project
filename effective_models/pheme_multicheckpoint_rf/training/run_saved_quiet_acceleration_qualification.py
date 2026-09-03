"""Evaluate inner-selected 20/30-minute quiet acceleration score thresholds."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

import quiet_acceleration_qualification_common as common


HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parent / "experiments"
SCORES = OUT_ROOT / "20260903_194947_531733_quiet_acceleration_rf_full"
FEATURES = OUT_ROOT / "20260903_183904_151619_quiet_acceleration_materialization"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    output = OUT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + "_saved_quiet_acceleration_qualification"
    )
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "started_at": now(),
        "purpose": "20/30-minute quiet acceleration threshold qualification without ranking",
        "model_run": str(SCORES.resolve()), "outcome_source": str(FEATURES.resolve()),
        "split": "thresholds selected separately from nested inner OOF for each outer event",
        "checkpoints_seconds": list(common.CHECKPOINTS),
        "target_positive_recall": list(common.RECALL_TARGETS),
        "threshold_strategies": {
            "macro": "minimum selections meeting inner macro event recall",
            "robust": "macro target plus worst inner event >= half of target",
        },
        "selection_rule": "prediction >= inner-selected threshold; no Top-K; no repeated thread",
        "research_safety": (
            "No model fitting. Outer labels, Oracle Top-50, and future impact are evaluation-only. "
            "Each threshold is selected using only the matching outer fold's nested inner OOF."
        ),
    }
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        print("Starting saved quiet acceleration qualification evaluation.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/6] Loading completed inner/outer OOF and traceable outcomes...", flush=True)
        inner = pd.read_csv(
            SCORES / "inner_quiet_acceleration_oof.csv",
            dtype={"thread_id": str, "event_id": str, "outer_event": str},
        )
        outer = pd.read_csv(
            SCORES / "outer_quiet_acceleration_oof.csv",
            dtype={"thread_id": str, "event_id": str},
        )
        outcomes = pd.read_csv(
            FEATURES / "pheme_quiet_acceleration_features.csv",
            dtype={"thread_id": str, "event_id": str},
        )
        common.validate_prediction_frame(inner, inner=True)
        common.validate_prediction_frame(outer, inner=False)
        inner = common.attach_outcomes(inner, outcomes)
        outer = common.attach_outcomes(outer, outcomes)
        outcomes = outcomes.loc[outcomes.checkpoint_sec.isin(common.CHECKPOINTS)].copy()

        print("[2/6] Selecting fold-specific thresholds from inner OOF only...", flush=True)
        frontier_rows, threshold_rows = [], []
        for outer_index, outer_event in enumerate(sorted(outer.event_id.unique()), 1):
            print(f"  outer event {outer_index}/7: {outer_event}", flush=True)
            inner_fold = inner.loc[inner.outer_event.eq(outer_event)]
            for checkpoint in common.CHECKPOINTS:
                frontier = common.threshold_frontier(inner_fold, checkpoint)
                frontier["outer_event"] = outer_event
                frontier_rows.append(frontier)
                for strategy, worst_fraction in (
                    ("macro", None), ("robust", common.WORST_EVENT_FRACTION)
                ):
                    for target in common.RECALL_TARGETS:
                        chosen = common.choose_threshold(
                            frontier, target, worst_event_fraction=worst_fraction,
                        )
                        threshold_rows.append({
                            "outer_event": outer_event,
                            "policy_name": f"{strategy}_recall_{int(target * 100)}",
                            "threshold_strategy": strategy,
                            "target_positive_recall": target,
                            **chosen.to_dict(),
                        })
        frontiers = pd.concat(frontier_rows, ignore_index=True)
        thresholds = pd.DataFrame(threshold_rows)

        print("[3/6] Replaying threshold-only qualification at 20 then 30 minutes...", flush=True)
        metric_rows, selected_rows, checkpoint_rows = [], [], []
        for outer_event in sorted(outer.event_id.unique()):
            event = outer.loc[
                outer.event_id.eq(outer_event)
                & outer.checkpoint_sec.isin(common.CHECKPOINTS)
            ].copy()
            all_event = outcomes.loc[outcomes.event_id.eq(outer_event)].copy()
            oracle_union = common.oracle_top50_union(all_event)
            quiet_oracle_union = oracle_union & set(event.thread_id)
            all_quiet_selected, _ = common.apply_thresholds(
                event, {checkpoint: -np.inf for checkpoint in common.CHECKPOINTS},
                "all_quiet_reference",
            )
            all_quiet_impact = float(all_quiet_selected.dynamic_preventable_impact.sum())
            horizon = float(
                all_event.loc[all_event.checkpoint_sec.eq(1200), "future_growth"].sum()
            )
            for threshold_row in thresholds.loc[
                thresholds.outer_event.eq(outer_event)
            ][["policy_name", "threshold_strategy", "target_positive_recall"]].drop_duplicates().itertuples(index=False):
                target = float(threshold_row.target_positive_recall)
                policy_name = str(threshold_row.policy_name)
                chosen_thresholds = thresholds.loc[
                    thresholds.outer_event.eq(outer_event)
                    & thresholds.policy_name.eq(policy_name)
                ]
                choices = dict(zip(
                    chosen_thresholds.checkpoint_sec.astype(int),
                    chosen_thresholds.threshold.astype(float),
                ))
                selected, checkpoint_metrics = common.apply_thresholds(
                    event, choices, policy_name,
                )
                oracle_ceiling = common.same_count_quiet_oracle(event, selected)
                selected_ids = set(selected.thread_id)
                blocked = float(selected.dynamic_preventable_impact.sum())
                ceiling = float(oracle_ceiling.dynamic_preventable_impact.sum())
                hits = len(selected_ids & quiet_oracle_union)
                metric_rows.append({
                    "outer_event": outer_event, "policy_name": policy_name,
                    "threshold_strategy": str(threshold_row.threshold_strategy),
                    "target_positive_recall": target,
                    "interventions": len(selected),
                    "interventions_at_20m": int(selected.action_checkpoint_sec.eq(1200).sum()),
                    "interventions_at_30m": int(selected.action_checkpoint_sec.eq(1800).sum()),
                    "positive_interventions": int(selected.dynamic_preventable_impact.gt(0).sum()),
                    "blocked_impact": blocked,
                    "horizon_future_at_20m": horizon,
                    "horizon_reduction": blocked / horizon if horizon else 0.0,
                    "all_quiet_reference_interventions": len(all_quiet_selected),
                    "all_quiet_reference_impact": all_quiet_impact,
                    "quiet_impact_capture": blocked / all_quiet_impact if all_quiet_impact else 0.0,
                    "quiet_oracle_top50_union": len(quiet_oracle_union),
                    "oracle_top50_hits": hits,
                    "quiet_oracle_top50_recall": hits / len(quiet_oracle_union)
                    if quiet_oracle_union else 0.0,
                    "same_count_quiet_oracle_impact": ceiling,
                    "same_count_oracle_efficiency": blocked / ceiling if ceiling else 0.0,
                })
                checkpoint_metrics["outer_event"] = outer_event
                checkpoint_metrics["threshold_strategy"] = str(threshold_row.threshold_strategy)
                checkpoint_metrics["target_positive_recall"] = target
                checkpoint_rows.append(checkpoint_metrics)
                if len(selected):
                    selected_rows.append(selected[[
                        "thread_id", "event_id", "policy_name", "action_checkpoint_sec",
                        "qualification_threshold", "prediction",
                        "predicted_positive_acceleration", "next10_positive_acceleration",
                        "dynamic_preventable_impact", "future_growth",
                    ]].assign(outer_event=outer_event))

        print("[4/6] Summarizing held-out efficiency, Oracle coverage, and future impact...", flush=True)
        metrics = pd.DataFrame(metric_rows)
        checkpoint_metrics = pd.concat(checkpoint_rows, ignore_index=True)
        selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
        macro = metrics.groupby(
            ["policy_name", "threshold_strategy", "target_positive_recall"], as_index=False
        ).agg(
            outer_events=("outer_event", "nunique"),
            mean_interventions=("interventions", "mean"),
            total_interventions=("interventions", "sum"),
            mean_positive_interventions=("positive_interventions", "mean"),
            mean_blocked_impact=("blocked_impact", "mean"),
            mean_horizon_reduction=("horizon_reduction", "mean"),
            mean_quiet_impact_capture=("quiet_impact_capture", "mean"),
            total_oracle_top50_hits=("oracle_top50_hits", "sum"),
            total_quiet_oracle_top50_union=("quiet_oracle_top50_union", "sum"),
            mean_quiet_oracle_top50_recall=("quiet_oracle_top50_recall", "mean"),
            mean_same_count_oracle_efficiency=("same_count_oracle_efficiency", "mean"),
        )
        macro["micro_quiet_oracle_top50_recall"] = (
            macro.total_oracle_top50_hits / macro.total_quiet_oracle_top50_union
        )

        print("[5/6] Auditing split provenance, identities, and evaluation-only outcomes...", flush=True)
        if set(inner.outer_event) != set(outer.event_id):
            raise ValueError("inner/outer fold coverage mismatch")
        if selected.duplicated(["outer_event", "policy_name", "thread_id"]).any():
            raise ValueError("duplicate intervention identity")
        if thresholds.groupby(["outer_event", "policy_name"]).checkpoint_sec.nunique().ne(2).any():
            raise ValueError("missing inner-selected checkpoint threshold")

        print("[6/6] Writing complete diagnostic and run record...", flush=True)
        frontiers.to_csv(output / "inner_threshold_frontier.csv", index=False)
        thresholds.to_csv(output / "inner_selected_thresholds.csv", index=False)
        metrics.to_csv(output / "outer_policy_per_event.csv", index=False)
        macro.to_csv(output / "outer_policy_macro.csv", index=False)
        checkpoint_metrics.to_csv(output / "outer_checkpoint_qualification.csv", index=False)
        selected.to_csv(output / "outer_selected_threads.csv", index=False)
        result = {
            "policy_macro": macro.to_dict(orient="records"),
            "interpretation_boundary": (
                "Exploratory saved-OOF qualification diagnostic. Threshold selection is inner-only; "
                "Oracle Top-50 and future impact are held-out evaluation outcomes, not policy inputs."
            ),
        }
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        record.update({
            "status": "complete", "completed_at": now(),
            "outer_events": sorted(outer.event_id.unique()),
            "input_rows": {"inner_oof": len(inner), "outer_oof": len(outer)},
            "metrics": macro.to_dict(orient="records"),
            "output_files": [
                "inner_threshold_frontier.csv", "inner_selected_thresholds.csv",
                "outer_policy_per_event.csv", "outer_policy_macro.csv",
                "outer_checkpoint_qualification.csv", "outer_selected_threads.csv",
                "result.json", "run_record.json",
            ],
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"SUCCESS: saved quiet acceleration qualification saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({
            "status": "failed", "failed_at": now(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "output_files": sorted(path.name for path in output.iterdir()),
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"FAILURE: saved quiet acceleration qualification preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
