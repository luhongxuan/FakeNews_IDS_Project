"""Evaluate one frozen AI Top-50 submission against the separated answer key."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
MODEL_ROOT = HERE.parent
ROOT = HERE.parents[2]
OUT_ROOT = MODEL_ROOT / "experiments"
PREDICTIONS = Path(
    r"C:\Users\luhon\.codex\attachments\a733cb6b-5308-4669-a1fc-8fec365f117c\pasted-text.txt"
)
DATASET_RUN = (
    MODEL_ROOT / "experiments" / "20260903_144246_041266_strict30_ai_blind_dataset_full"
)
ANSWER_KEY = DATASET_RUN / "answer_key_DO_NOT_GIVE_TO_AI.csv"
V5_SCORES = (
    ROOT / "effective_models" / "pheme_v5_text_rf" / "reference_result"
    / "20260831_223827_909201_v5_best_text_oof_min_interventions"
    / "oof_thread_scores.csv"
)
CORRECTED_V5_SCORES = (
    ROOT / "effective_models" / "pheme_v5_corrected_target" / "experiments"
    / "20260903_143643_168807_corrected_v5_nested_full" / "oof_thread_scores.csv"
)
CORRECTED_GRAPHS = (
    ROOT / "effective_models" / "pheme_v5_corrected_target" / "experiments"
    / "20260903_143643_168807_corrected_v5_nested_full"
    / "pheme_v5_strict30_corrected_all_roots_semantic.pt"
)
HYBRID_SCORES = (
    ROOT / "effective_models" / "pheme_active_quiet_calibrated_rf" / "experiments"
    / "20260902_132700_413954_quiet_tier_hybrid_policy_full" / "outer_scores.csv"
)
BUDGETS = (10, 20, 50)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def validate_submission(prediction: pd.DataFrame, answer: pd.DataFrame) -> None:
    required = {
        "case_id", "event_group", "predicted_priority", "confidence_0_to_1", "short_reason"
    }
    if not required.issubset(prediction.columns):
        raise ValueError(f"prediction missing columns: {sorted(required - set(prediction.columns))}")
    if prediction.case_id.isna().any() or prediction.case_id.duplicated().any():
        raise ValueError("prediction contains missing or duplicate case IDs")
    if not set(prediction.case_id).issubset(set(answer.case_id)):
        raise ValueError("prediction contains unknown case IDs")
    prediction["predicted_priority"] = pd.to_numeric(
        prediction.predicted_priority, errors="raise"
    ).astype(int)
    prediction["confidence_0_to_1"] = pd.to_numeric(
        prediction.confidence_0_to_1, errors="raise"
    )
    if not prediction.confidence_0_to_1.between(0, 1).all():
        raise ValueError("confidence outside [0, 1]")
    expected_group = answer.set_index("case_id").event_group
    if not prediction.event_group.eq(prediction.case_id.map(expected_group)).all():
        raise ValueError("case/event group mismatch")
    for event_group, group in prediction.groupby("event_group", sort=True):
        priorities = sorted(group.predicted_priority.tolist())
        if priorities != list(range(1, len(group) + 1)):
            raise ValueError(f"{event_group}: priority is not consecutive from 1")
        eligible = bool(answer.loc[answer.event_group.eq(event_group), "official_eligible_event"].iloc[0])
        if eligible and len(group) != 50:
            raise ValueError(f"{event_group}: official group does not contain exactly 50 selections")


def baseline_selection(
    score_frame: pd.DataFrame, answer: pd.DataFrame, score_column: str, budget: int
) -> pd.DataFrame:
    merged = answer.merge(
        score_frame[["thread_id", score_column]], on="thread_id", validate="one_to_one"
    )
    return (
        merged.sort_values(
            ["event_group", score_column, "thread_id"],
            ascending=[True, False, True], kind="stable",
        )
        .groupby("event_group", sort=True, group_keys=False)
        .head(budget)
    )


def metric_row(
    method: str, event_group: str, budget: int, selected: pd.DataFrame, event: pd.DataFrame
) -> dict:
    oracle_impact = event.sort_values(
        ["corrected_preventable_impact", "thread_id"],
        ascending=[False, True], kind="stable",
    ).head(budget)
    oracle_volume = event.sort_values(
        ["post30_reply_count", "thread_id"], ascending=[False, True], kind="stable"
    ).head(budget)
    blocked = float(selected.corrected_preventable_impact.sum())
    volume = float(selected.post30_reply_count.sum())
    total_impact = float(event.corrected_preventable_impact.sum())
    total_future = float(event.project_future_growth.sum())
    total_volume = float(event.post30_reply_count.sum())
    optimal_impact = float(oracle_impact.corrected_preventable_impact.sum())
    optimal_volume = float(oracle_volume.post30_reply_count.sum())
    return {
        "method": method, "event_group": event_group, "budget": budget,
        "selected": len(selected),
        "intervention_top50_hits": int(selected.oracle_intervention_top50.sum()),
        "post30_volume_top50_hits": int(selected.oracle_post30_volume_top50.sum()),
        "blocked_impact": blocked,
        "crr": blocked / total_future if total_future else 0.0,
        "preventable_recall": blocked / total_impact if total_impact else 0.0,
        "intervention_oracle_efficiency": blocked / optimal_impact if optimal_impact else 0.0,
        "captured_post30_replies": volume,
        "post30_volume_recall": volume / total_volume if total_volume else 0.0,
        "post30_volume_oracle_efficiency": volume / optimal_volume if optimal_volume else 0.0,
    }


def main() -> None:
    output = OUT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_ai_blind_prediction_evaluation"
    )
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "started_at": now(),
        "purpose": "evaluate frozen AI strict-30 Top-50 predictions",
        "inputs": {
            "predictions": str(PREDICTIONS.resolve()), "answer_key": str(ANSWER_KEY.resolve()),
            "v5_scores": str(V5_SCORES.resolve()), "hybrid_scores": str(HYBRID_SCORES.resolve()),
            "corrected_v5_scores": str(CORRECTED_V5_SCORES.resolve()),
            "corrected_graphs": str(CORRECTED_GRAPHS.resolve()),
        },
        "split": "seven official event groups with at least 100 candidates",
        "cutoff_seconds": 1800, "budgets": list(BUDGETS), "seed": None,
        "research_safety": (
            "Predictions were frozen before answer-key access. All methods are re-evaluated on the "
            "same corrected all-roots outcome and official event groups; no model is fitted or tuned."
        ),
    }
    write_record(output, record)
    try:
        print("Starting frozen AI strict-30 prediction evaluation.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/5] Loading frozen predictions and separate answer key...", flush=True)
        prediction = pd.read_csv(PREDICTIONS, dtype={"case_id": str, "event_group": str})
        answer = pd.read_csv(
            ANSWER_KEY, dtype={"case_id": str, "event_group": str, "thread_id": str, "event_id": str}
        )
        graphs = torch.load(CORRECTED_GRAPHS, weights_only=False)
        future_by_thread = {
            str(graph.thread_id): int(round(float(torch.expm1(graph.y).item())))
            for graph in graphs
        }
        answer["project_future_growth"] = answer.thread_id.map(future_by_thread)
        if answer.project_future_growth.isna().any():
            raise ValueError("answer key is missing project future-growth denominators")
        print("[2/5] Validating submission coverage, identity, ranks, and confidence...", flush=True)
        validate_submission(prediction, answer)
        official = answer.loc[answer.official_eligible_event].copy()
        frozen = prediction.merge(answer, on=["case_id", "event_group"], validate="one_to_one")

        print("[3/5] Evaluating AI and corrected-target v5/two-expert comparators...", flush=True)
        v5 = pd.read_csv(V5_SCORES, dtype={"thread_id": str})
        corrected_v5 = pd.read_csv(CORRECTED_V5_SCORES, dtype={"thread_id": str})
        hybrid = pd.read_csv(HYBRID_SCORES, dtype={"thread_id": str})
        metric_rows = []
        selections = []
        for budget in BUDGETS:
            ai_selected = frozen.loc[frozen.predicted_priority.le(budget)].copy()
            comparators = {
                "ai_blind": ai_selected,
                "v5_text_rf": baseline_selection(v5, official, "prediction", budget),
                "corrected_v5_text_rf": baseline_selection(
                    corrected_v5, official, "prediction", budget
                ),
                "quiet_tier_hybrid": baseline_selection(
                    hybrid, official, "hybrid_calibrated_score", budget
                ),
            }
            for method, selected_all in comparators.items():
                for event_group, event in official.groupby("event_group", sort=True):
                    selected = selected_all.loc[selected_all.event_group.eq(event_group)].copy()
                    metric_rows.append(metric_row(method, event_group, budget, selected, event))
                    selections.append(
                        selected.assign(method=method, budget=budget)[
                            ["method", "budget", "case_id", "event_group", "thread_id",
                             "corrected_preventable_impact", "post30_reply_count",
                             "oracle_intervention_top50", "oracle_post30_volume_top50"]
                        ]
                    )
        metrics = pd.DataFrame(metric_rows)
        macro = metrics.groupby(["method", "budget"], as_index=False).agg(
            eligible_events=("event_group", "nunique"),
            total_selected=("selected", "sum"),
            total_intervention_top50_hits=("intervention_top50_hits", "sum"),
            total_post30_volume_top50_hits=("post30_volume_top50_hits", "sum"),
            mean_crr=("crr", "mean"),
            mean_preventable_recall=("preventable_recall", "mean"),
            mean_intervention_oracle_efficiency=("intervention_oracle_efficiency", "mean"),
            mean_post30_volume_recall=("post30_volume_recall", "mean"),
            mean_post30_volume_oracle_efficiency=("post30_volume_oracle_efficiency", "mean"),
        )

        quiet_map = hybrid.set_index("thread_id").quiet.astype(bool)
        official_with_gate = official.assign(quiet=official.thread_id.map(quiet_map))
        if official_with_gate.quiet.isna().any():
            raise ValueError("missing saved fold-specific quiet gate")
        quiet_oracle = official_with_gate.loc[official_with_gate.oracle_intervention_top50]
        expert_rows = []
        selected_at_50 = pd.concat(selections, ignore_index=True)
        selected_at_50 = selected_at_50.loc[selected_at_50.budget.eq(50)].copy()
        selected_at_50["quiet"] = selected_at_50.thread_id.map(quiet_map)
        for method, selected in selected_at_50.groupby("method", sort=True):
            selected_quiet = selected.loc[selected.quiet]
            selected_active = selected.loc[~selected.quiet]
            quiet_hits = int(selected_quiet.oracle_intervention_top50.sum())
            active_hits = int(selected_active.oracle_intervention_top50.sum())
            quiet_total = int(quiet_oracle.quiet.sum())
            active_total = int((~quiet_oracle.quiet).sum())
            expert_rows.extend([
                {
                    "method": method, "expert": "quiet", "selected": len(selected_quiet),
                    "oracle_top50_total": quiet_total, "oracle_top50_hits": quiet_hits,
                    "precision": quiet_hits / len(selected_quiet) if len(selected_quiet) else 0.0,
                    "recall": quiet_hits / quiet_total if quiet_total else 0.0,
                },
                {
                    "method": method, "expert": "active", "selected": len(selected_active),
                    "oracle_top50_total": active_total, "oracle_top50_hits": active_hits,
                    "precision": active_hits / len(selected_active) if len(selected_active) else 0.0,
                    "recall": active_hits / active_total if active_total else 0.0,
                },
            ])
        expert_breakdown = pd.DataFrame(expert_rows)
        quiet_hit_sets = {
            method: set(group.loc[group.quiet & group.oracle_intervention_top50, "thread_id"])
            for method, group in selected_at_50.groupby("method", sort=True)
        }
        ai_quiet_hits = quiet_hit_sets["ai_blind"]
        hybrid_quiet_hits = quiet_hit_sets["quiet_tier_hybrid"]
        quiet_complementarity = {
            "ai_quiet_oracle_hits": len(ai_quiet_hits),
            "hybrid_quiet_oracle_hits": len(hybrid_quiet_hits),
            "shared_hits": len(ai_quiet_hits & hybrid_quiet_hits),
            "ai_only_hits": len(ai_quiet_hits - hybrid_quiet_hits),
            "hybrid_only_hits": len(hybrid_quiet_hits - ai_quiet_hits),
            "union_hits": len(ai_quiet_hits | hybrid_quiet_hits),
        }

        print("[4/5] Auditing AI reason signals among its selected cases...", flush=True)
        ai50 = frozen.loc[
            frozen.official_eligible_event & frozen.predicted_priority.le(50)
        ].copy()
        reason_rows = []
        for row in ai50.itertuples(index=False):
            for signal in (part.strip() for part in str(row.short_reason).split("，") if part.strip()):
                reason_rows.append({
                    "signal": signal,
                    "intervention_top50": bool(row.oracle_intervention_top50),
                    "post30_volume_top50": bool(row.oracle_post30_volume_top50),
                })
        reasons = pd.DataFrame(reason_rows).groupby("signal", as_index=False).agg(
            selected_mentions=("signal", "size"),
            intervention_top50_hits=("intervention_top50", "sum"),
            post30_volume_top50_hits=("post30_volume_top50", "sum"),
        )
        reasons["intervention_hit_rate"] = (
            reasons.intervention_top50_hits / reasons.selected_mentions
        )
        reasons["post30_volume_hit_rate"] = (
            reasons.post30_volume_top50_hits / reasons.selected_mentions
        )

        hit = ai50.oracle_intervention_top50.astype(float)
        confidence = ai50.confidence_0_to_1.astype(float)
        confidence_audit = {
            "interpretation": (
                "Diagnostic only: the prompt did not formally define confidence as a calibrated "
                "probability of Oracle Top-50 membership."
            ),
            "mean_confidence": float(confidence.mean()),
            "actual_intervention_top50_rate": float(hit.mean()),
            "mean_confidence_when_hit": float(confidence.loc[hit.eq(1)].mean()),
            "mean_confidence_when_miss": float(confidence.loc[hit.eq(0)].mean()),
            "brier_if_interpreted_as_membership_probability": float(np.square(confidence - hit).mean()),
        }
        group_sizes = official.groupby("event_group").size().astype(float)
        random_expectation_at_50 = {
            "expected_total_intervention_top50_hits": float((2500.0 / group_sizes).sum()),
            "expected_mean_preventable_recall": float((50.0 / group_sizes).mean()),
            "note": "Analytical expectation under uniform random selection of 50 per event.",
        }

        print("[5/5] Writing evaluation evidence and complete run record...", flush=True)
        metrics.to_csv(output / "per_event_metrics.csv", index=False)
        macro.to_csv(output / "macro_comparison.csv", index=False)
        pd.concat(selections, ignore_index=True).to_csv(output / "selected_cases.csv", index=False)
        expert_breakdown.to_csv(output / "top50_expert_breakdown.csv", index=False)
        reasons.sort_values("selected_mentions", ascending=False).to_csv(
            output / "reason_signal_summary.csv", index=False
        )
        summary = {
            "submission_rows": len(prediction),
            "official_ai_selections_at_50": len(ai50),
            "confidence_audit": confidence_audit,
            "random_expectation_at_50": random_expectation_at_50,
            "top50_expert_breakdown": expert_breakdown.to_dict(orient="records"),
            "quiet_hit_complementarity": quiet_complementarity,
            "macro": macro.to_dict(orient="records"),
        }
        (output / "result.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        record.update({
            "status": "complete", "completed_at": now(), "metrics": summary,
            "output_files": [
                "per_event_metrics.csv", "macro_comparison.csv", "selected_cases.csv",
                "top50_expert_breakdown.csv", "reason_signal_summary.csv", "result.json",
                "run_record.json",
            ],
        })
        write_record(output, record)
        print(f"SUCCESS: AI blind prediction evaluation saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({
            "status": "failed", "failed_at": now(),
            "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
            "output_files": sorted(path.name for path in output.iterdir()),
        })
        write_record(output, record)
        print(f"FAILURE: AI prediction evaluation preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
