"""Full eligible-event outer-LOEO active/quiet + quiet-tier hybrid policy."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import traceback

import numpy as np
import pandas as pd

import run_quiet_tier_hybrid_smoke as common


common.RUN_MODE = "full"
common.TIER_TREES = 300


def append_oof_rows(
    graphs: list, event: str, baseline_raw: np.ndarray, gate: dict,
    candidates: dict[float, np.ndarray], baseline_rows: list[dict],
    candidate_rows: dict[float, list[dict]],
) -> None:
    quiet = common.recency_values(graphs) >= gate["threshold"]
    for index, graph in enumerate(graphs):
        shared = {
            "validation_event": event,
            "thread_id": str(graph.thread_id),
            "expert": "quiet" if quiet[index] else "active",
            "target": float(graph.preventable_y.item()),
        }
        baseline_rows.append({**shared, "raw_score": float(baseline_raw[index])})
        for weight in common.UTILITY_WEIGHTS:
            candidate_rows[weight].append({**shared, "raw_score": float(candidates[weight][index])})


def calibrate_oof(oof: pd.DataFrame, calibrators: dict) -> pd.DataFrame:
    result = oof.copy()
    quiet = result.expert.eq("quiet").to_numpy()
    result["score"] = common.apply_calibration(result.raw_score.to_numpy(float), quiet, calibrators)
    result["event_id"] = result.validation_event
    result["preventable_impact"] = np.expm1(result.target)
    return result


def main() -> None:
    output = common.OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_quiet_tier_hybrid_policy_full")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": "full",
        "dataset": str(common.DATASET),
        "split": "seven eligible outer LOEO events; all remaining events form outer train and inner OOF",
        "cutoff_seconds": 1800,
        "candidate_pool": "protected v5 rumour-only strict-30 snapshots",
        "source_weight": common.SOURCE_WEIGHT,
        "utility_weights": list(common.UTILITY_WEIGHTS),
        "tier_trees": common.TIER_TREES,
        "selection_metric": "outer-train inner-OOF quiet high-tier overlap; full-policy reduction@10 breaks ties",
        "budgets": list(common.BUDGETS),
        "seed": common.SEED,
        "research_safety": "No saved outer predictions are used for fitting. Gate, PCA, tier models, utility models, lambda selection, and expert calibration are re-fitted inside each outer fold using outer-train events only.",
    }
    common.write_record(output, record)
    try:
        print("Starting full active/quiet + quiet-tier hybrid policy.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/6] Loading and validating protected inputs...", flush=True)
        graphs, events, eligible = common.load_graphs()
        depth_nodes, account_age = common.load_safe_nodes()
        offsets = common.load_node_offsets()
        frame, temporal_columns, embeddings = common.graph_inputs(graphs)
        all_scores, all_curves, all_selections, gates, fold_details = [], [], [], [], []
        print("[2/6] Running eligible outer LOEO folds...", flush=True)
        for outer_index, outer_event in enumerate(eligible, 1):
            print(f"  outer {outer_index}/{len(eligible)}: {outer_event}", flush=True)
            outer_train = [graph for graph in graphs if str(graph.event_id) != outer_event]
            held_out = [graph for graph in graphs if str(graph.event_id) == outer_event]
            inner_events = sorted({str(graph.event_id) for graph in outer_train})
            baseline_rows: list[dict] = []
            candidate_rows = {weight: [] for weight in common.UTILITY_WEIGHTS}
            for inner_index, inner_event in enumerate(inner_events, 1):
                print(f"    inner {inner_index}/{len(inner_events)}: validation={inner_event}", flush=True)
                inner_train = [graph for graph in outer_train if str(graph.event_id) != inner_event]
                validation = [graph for graph in outer_train if str(graph.event_id) == inner_event]
                baseline_raw, gate, candidates = common.raw_components(
                    inner_train, validation, depth_nodes, account_age, offsets,
                    frame, temporal_columns, embeddings,
                    common.SEED + outer_index * 1000 + inner_index,
                )
                append_oof_rows(validation, inner_event, baseline_raw, gate, candidates, baseline_rows, candidate_rows)
            print("    selecting lambda and fitting expert calibrators...", flush=True)
            baseline_oof = common.calibrator_frame(baseline_rows)
            baseline_calibrators, baseline_details = common.fit_calibrators(baseline_oof)
            selection_rows = []
            hybrid_calibrators = {}
            hybrid_details = {}
            for weight in common.UTILITY_WEIGHTS:
                oof = common.calibrator_frame(candidate_rows[weight])
                calibrators, details = common.fit_calibrators(oof)
                hybrid_calibrators[weight] = calibrators
                hybrid_details[weight] = details
                scored_oof = calibrate_oof(oof, calibrators)
                selection_rows.append({
                    "outer_event": outer_event,
                    "utility_weight": weight,
                    "inner_quiet_high_overlap": common.quiet_high_overlap(oof),
                    "inner_oof_reduction_at_10": common.reduction_at_10(scored_oof, "score"),
                })
            selection = pd.DataFrame(selection_rows).sort_values(
                ["inner_quiet_high_overlap", "inner_oof_reduction_at_10", "utility_weight"],
                ascending=[False, False, False], kind="stable",
            )
            selected_weight = float(selection.iloc[0].utility_weight)
            all_selections.append(selection)
            print(f"    selected utility_weight={selected_weight:.2f}; scoring held-out event...", flush=True)
            baseline_outer, outer_gate, outer_candidates = common.raw_components(
                outer_train, held_out, depth_nodes, account_age, offsets,
                frame, temporal_columns, embeddings,
                common.SEED + 100000 + outer_index,
            )
            outer_quiet = common.recency_values(held_out) >= outer_gate["threshold"]
            baseline_score = common.apply_calibration(baseline_outer, outer_quiet, baseline_calibrators)
            hybrid_score = common.apply_calibration(outer_candidates[selected_weight], outer_quiet, hybrid_calibrators[selected_weight])
            scores = pd.DataFrame({
                "thread_id": [str(graph.thread_id) for graph in held_out],
                "event_id": [str(graph.event_id) for graph in held_out],
                "preventable_impact": [float(graph.preventable_impact.item()) for graph in held_out],
                "baseline_calibrated_score": baseline_score,
                "hybrid_calibrated_score": hybrid_score,
                "quiet": outer_quiet,
                "selected_utility_weight": selected_weight,
            })
            all_scores.append(scores)
            all_curves.extend(common.curve_rows(scores, "baseline_calibrated_score", "calibrated_active_quiet_rf"))
            all_curves.extend(common.curve_rows(scores, "hybrid_calibrated_score", "quiet_tier_hybrid"))
            gates.append({"outer_event": outer_event, **outer_gate})
            fold_details.append({"outer_event": outer_event, "selected_utility_weight": selected_weight, "baseline_calibrators": baseline_details, "hybrid_calibrators": hybrid_details[selected_weight]})
            pd.concat(all_scores, ignore_index=True).to_csv(output / "outer_scores_partial.csv", index=False)
            (output / "partial_result.json").write_text(json.dumps({"completed_outer_events": fold_details}, indent=2), encoding="utf-8")
        print("[3/6] Computing eligible-event budget summaries...", flush=True)
        scores_frame = pd.concat(all_scores, ignore_index=True)
        curves = pd.DataFrame(all_curves)
        summary = curves.groupby(["model", "budget"], as_index=False).agg(
            eligible_events=("event_id", "nunique"),
            mean_oracle_efficiency=("oracle_efficiency", "mean"),
            mean_model_reduction=("model_reduction", "mean"),
        )
        comparison = summary.pivot(index="budget", columns="model", values=["mean_model_reduction", "mean_oracle_efficiency"])
        comparison.columns = [f"{metric}_{model}" for metric, model in comparison.columns]
        comparison = comparison.reset_index()
        comparison["hybrid_minus_baseline_reduction"] = comparison["mean_model_reduction_quiet_tier_hybrid"] - comparison["mean_model_reduction_calibrated_active_quiet_rf"]
        comparison["hybrid_minus_baseline_efficiency"] = comparison["mean_oracle_efficiency_quiet_tier_hybrid"] - comparison["mean_oracle_efficiency_calibrated_active_quiet_rf"]
        print("[4/6] Validating score/output integrity...", flush=True)
        if scores_frame.thread_id.duplicated().any() or not np.isfinite(scores_frame[["baseline_calibrated_score", "hybrid_calibrated_score", "preventable_impact"]].to_numpy(float)).all():
            raise ValueError("Full hybrid OOF integrity failure")
        print("[5/6] Writing full policy evidence...", flush=True)
        scores_frame.to_csv(output / "outer_scores.csv", index=False)
        curves.to_csv(output / "per_event_budget_curve.csv", index=False)
        summary.to_csv(output / "eligible_event_budget_summary.csv", index=False)
        comparison.to_csv(output / "hybrid_vs_baseline_budget_comparison.csv", index=False)
        pd.concat(all_selections, ignore_index=True).to_csv(output / "inner_weight_selection.csv", index=False)
        pd.DataFrame(gates).to_csv(output / "outer_gate_details.csv", index=False)
        print("[6/6] Finalizing run record...", flush=True)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "eligible_events": eligible, "fold_details": fold_details, "output_files": ["outer_scores.csv", "per_event_budget_curve.csv", "eligible_event_budget_summary.csv", "hybrid_vs_baseline_budget_comparison.csv", "inner_weight_selection.csv", "outer_gate_details.csv"]})
        common.write_record(output, record)
        print(f"SUCCESS: full active/quiet + quiet-tier hybrid policy saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
        common.write_record(output, record)
        print(f"FAILURE: full hybrid policy preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
