"""Long manual paired OOF test of uncalibrated versus train-only calibrated experts."""
from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from pheme_account_age_v5_common import BASE, DATASET, ELIGIBLE_MIN_THREADS, MODEL_PARAMS, SEED, load_graphs, load_node_offsets, load_safe_nodes
from pheme_quiet_expert_common import gated_predict, safety_statement
from pheme_quiet_expert_calibration_common import calibrated_scores, calibration_safety_statement, fit_calibrators, inner_oof_predictions


OUT = BASE / "experiments" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_paired_oof_pheme_quiet_expert_calibration")
BUDGETS = (1, 3, 5, 10, 20, 50, 100)
MODELS = ("uncalibrated_two_expert_rf", "train_only_calibrated_two_expert_rf")


def curve_rows(pool: pd.DataFrame, score_column: str, model: str) -> list[dict]:
    ranked = pool.sort_values([score_column, "thread_id"], ascending=[False, True], kind="stable")
    oracle = pool.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable")
    total = float(pool.preventable_impact.sum())
    rows = []
    for budget in BUDGETS:
        actual = min(budget, len(pool))
        blocked = float(ranked.head(actual).preventable_impact.sum())
        optimal = float(oracle.head(actual).preventable_impact.sum())
        rows.append({"model": model, "event_id": str(pool.event_id.iloc[0]), "budget": budget, "actual_budget": actual, "model_blocked_impact": blocked, "oracle_blocked_impact": optimal, "model_reduction": blocked / total if total else 0.0, "oracle_efficiency": blocked / optimal if optimal else 0.0})
    return rows


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUT}")
    print("Starting paired PHEME OOF quiet-expert calibration experiment...", flush=True)
    print(f"Output directory: {OUT.resolve()}", flush=True)
    OUT.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "dataset": str(DATASET), "split": "outer LOEO by PHEME event", "cutoff_seconds": 1800,
        "candidate_pool": "protected v5 rumour-only graph artifact", "target": "unchanged preventable_y = log1p(preventable_impact)",
        "models": list(MODELS), "model_params": MODEL_PARAMS, "seed": SEED, "fixed_budgets": list(BUDGETS), "eligible_min_threads": ELIGIBLE_MIN_THREADS,
        "hypothesis": "Train-only expert-specific linear calibration improves cross-group priority comparability without changing expert inputs, labels, or gate membership.",
        "research_safety": safety_statement() + " " + calibration_safety_statement(),
    }
    (OUT / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        print("[1/5] Loading protected snapshots, safe fields, and cutoff offsets...", flush=True)
        graphs, events, eligible = load_graphs()
        depth_nodes, account_age = load_safe_nodes()
        node_offsets = load_node_offsets()
        print(f"  graphs={len(graphs)}, events={len(events)}, eligible events={len(eligible)}.", flush=True)
        print("[2/5] Running outer LOEO; each fold creates calibration data only from its outer-train events...", flush=True)
        scores_all, calibration_all, details_all, gates, outer = [], [], [], [], {}
        for index, event in enumerate(events, 1):
            print(f"  outer {index}/{len(events)}: held-out={event}; constructing train-only calibration OOF...", flush=True)
            outer_train = [graph for graph in graphs if str(graph.event_id) != event]
            held_out = [graph for graph in graphs if str(graph.event_id) == event]
            oof = inner_oof_predictions(outer_train, depth_nodes, account_age, node_offsets, progress_prefix="    ")
            calibrators, details = fit_calibrators(oof)
            raw_score, gate, _ = gated_predict(outer_train, held_out, depth_nodes, account_age, node_offsets)
            calibrated = calibrated_scores(raw_score, held_out, gate, calibrators)
            score_frame = pd.DataFrame({"thread_id": [str(graph.thread_id) for graph in held_out], "event_id": [str(graph.event_id) for graph in held_out], "uncalibrated_two_expert_score": raw_score, "calibrated_two_expert_score": calibrated, "preventable_impact": [float(graph.preventable_impact.item()) for graph in held_out]})
            raw_curve = pd.DataFrame(curve_rows(score_frame, "uncalibrated_two_expert_score", "uncalibrated_two_expert_rf"))
            calibrated_curve = pd.DataFrame(curve_rows(score_frame, "calibrated_two_expert_score", "train_only_calibrated_two_expert_rf"))
            delta = float(calibrated_curve.oracle_efficiency.mean() - raw_curve.oracle_efficiency.mean())
            outer[event] = {"uncalibrated_mean_oracle_efficiency": float(raw_curve.oracle_efficiency.mean()), "calibrated_mean_oracle_efficiency": float(calibrated_curve.oracle_efficiency.mean()), "calibrated_minus_uncalibrated": delta, "gate": gate, "calibrators": details}
            scores_all.append(score_frame)
            calibration_all.append(oof.assign(outer_test_event=event))
            details_all.extend({"outer_test_event": event, **row} for row in details)
            gates.append({"outer_test_event": event, **gate})
            pd.concat(scores_all, ignore_index=True).to_csv(OUT / "paired_oof_scores_partial.csv", index=False)
            (OUT / "partial_result.json").write_text(json.dumps({"config": record, "outer_events": outer}, indent=2), encoding="utf-8")
            print(f"    uncalibrated={outer[event]['uncalibrated_mean_oracle_efficiency']:.4f}; calibrated={outer[event]['calibrated_mean_oracle_efficiency']:.4f}; delta={delta:+.4f}.", flush=True)
        print("[3/5] Computing fixed-budget paired OOF curves...", flush=True)
        scores = pd.concat(scores_all, ignore_index=True)
        if scores.thread_id.duplicated().any() or not np.isfinite(scores[["uncalibrated_two_expert_score", "calibrated_two_expert_score", "preventable_impact"]].to_numpy(dtype=float)).all():
            raise ValueError("Paired calibration OOF integrity failure")
        curves = pd.concat([
            pd.concat([
                pd.DataFrame(curve_rows(pool, "uncalibrated_two_expert_score", "uncalibrated_two_expert_rf")),
                pd.DataFrame(curve_rows(pool, "calibrated_two_expert_score", "train_only_calibrated_two_expert_rf")),
            ], ignore_index=True)
            for _, pool in scores.groupby("event_id", sort=True)
        ], ignore_index=True)
        summary = curves[curves.event_id.isin(eligible)].groupby(["model", "budget"], as_index=False).agg(eligible_events=("event_id", "nunique"), mean_oracle_efficiency=("oracle_efficiency", "mean"), mean_model_reduction=("model_reduction", "mean"))
        difference = summary.pivot(index="budget", columns="model", values="mean_oracle_efficiency").reset_index()
        difference["calibrated_minus_uncalibrated"] = difference["train_only_calibrated_two_expert_rf"] - difference["uncalibrated_two_expert_rf"]
        print("[4/5] Writing paired scores, calibration evidence, and curves...", flush=True)
        scores.to_csv(OUT / "paired_oof_scores.csv", index=False)
        curves.to_csv(OUT / "per_event_paired_budget_curve.csv", index=False)
        summary.to_csv(OUT / "eligible_event_paired_budget_summary.csv", index=False)
        difference.to_csv(OUT / "paired_oracle_efficiency_difference.csv", index=False)
        pd.concat(calibration_all, ignore_index=True).to_csv(OUT / "inner_oof_calibration_predictions.csv", index=False)
        pd.DataFrame(details_all).to_csv(OUT / "outer_fold_calibration_details.csv", index=False)
        pd.DataFrame(gates).to_csv(OUT / "outer_fold_gate_details.csv", index=False)
        print("[5/5] Finalizing run record...", flush=True)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "outer_events": outer, "output_files": ["paired_oof_scores.csv", "per_event_paired_budget_curve.csv", "eligible_event_paired_budget_summary.csv", "paired_oracle_efficiency_difference.csv", "inner_oof_calibration_predictions.csv", "outer_fold_calibration_details.csv", "outer_fold_gate_details.csv"]})
        (OUT / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"SUCCESS: paired PHEME quiet-expert calibration OOF result saved to: {OUT.resolve()}", flush=True)
    except Exception as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}"})
        (OUT / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"FAILURE: output preserved at: {OUT.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
