"""Full LOEO evaluation of a bounded soft boost for quiet tier candidate pools."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import traceback

import numpy as np
import pandas as pd

import run_quiet_tier_soft_boost_smoke as common
from pheme_quiet_expert_calibration_common import fit_calibrators


common.base.TIER_TREES = 300


def append_oof_rows(graphs: list, event: str, raw: np.ndarray, gate: dict, candidates: dict, baseline_rows: list[dict], candidate_rows: dict) -> None:
    common.append_rows(graphs, event, raw, gate, candidates, baseline_rows, candidate_rows)


def main() -> None:
    output = common.OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_quiet_tier_soft_boost_full")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "mode": "full",
        "dataset": str(common.base.DATASET), "split": "seven eligible outer LOEO events; inner LOEO chooses pool and bonus",
        "cutoff_seconds": 1800, "tier_trees": common.base.TIER_TREES, "configs": [list(config) for config in common.CONFIGS],
        "selection_metric": "outer-train inner-OOF full-policy reduction@50; quiet global Oracle Top-50 coverage then smaller bonus break ties",
        "budgets": list(common.base.BUDGETS), "seed": common.base.SEED,
        "research_safety": "For each outer fold, gate, tier models, utility models, candidate selection, and expert calibration are fitted using outer-train events only. No outer outcomes determine any configuration.",
    }
    common.write_record(output, record)
    try:
        print("Starting full active/quiet + quiet-tier soft-boost policy.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/6] Loading and validating protected inputs...", flush=True)
        graphs, _, eligible = common.base.load_graphs()
        depth_nodes, account_age = common.base.load_safe_nodes(); offsets = common.base.load_node_offsets()
        frame, temporal_columns, embeddings = common.base.graph_inputs(graphs)
        all_scores: list[pd.DataFrame] = []; all_curves: list[dict] = []; all_selection: list[pd.DataFrame] = []; gates = []; fold_details = []
        print("[2/6] Running eligible outer LOEO folds...", flush=True)
        for outer_index, outer_event in enumerate(eligible, 1):
            print(f"  outer {outer_index}/{len(eligible)}: {outer_event}", flush=True)
            outer_train = [graph for graph in graphs if str(graph.event_id) != outer_event]
            held_out = [graph for graph in graphs if str(graph.event_id) == outer_event]
            inner_events = sorted({str(graph.event_id) for graph in outer_train})
            baseline_rows: list[dict] = []; candidate_rows = {config: [] for config in common.CONFIGS}
            for inner_index, inner_event in enumerate(inner_events, 1):
                print(f"    inner {inner_index}/{len(inner_events)}: validation={inner_event}", flush=True)
                inner_train = [graph for graph in outer_train if str(graph.event_id) != inner_event]
                validation = [graph for graph in outer_train if str(graph.event_id) == inner_event]
                raw, gate, candidates = common.raw_components(inner_train, validation, depth_nodes, account_age, offsets, frame, temporal_columns, embeddings, common.base.SEED + outer_index * 1000 + inner_index)
                append_oof_rows(validation, inner_event, raw, gate, candidates, baseline_rows, candidate_rows)
            print("    selecting soft-boost configuration and fitting calibrators...", flush=True)
            baseline_oof = common.base.calibrator_frame(baseline_rows); baseline_calibrators, baseline_details = fit_calibrators(baseline_oof)
            selection_rows = []; candidate_calibrators = {}; candidate_details = {}
            for config in common.CONFIGS:
                oof = common.base.calibrator_frame(candidate_rows[config]); calibrators, details = fit_calibrators(oof)
                candidate_calibrators[config] = calibrators; candidate_details[config] = details
                scored = common.calibrate(oof, calibrators)
                selection_rows.append({"outer_event": outer_event, "pool_fraction": config[0], "bonus": config[1], "inner_oof_reduction_at_50": common.reduction_at_budget(scored, "score", 50), "inner_quiet_global_top50_coverage": common.quiet_global_top50_coverage(scored)})
            selection = pd.DataFrame(selection_rows).sort_values(["inner_oof_reduction_at_50", "inner_quiet_global_top50_coverage", "bonus", "pool_fraction"], ascending=[False, False, True, True], kind="stable")
            chosen = (float(selection.iloc[0].pool_fraction), float(selection.iloc[0].bonus)); all_selection.append(selection)
            print(f"    selected pool={chosen[0]:.2f}, bonus={chosen[1]:.2f}; scoring held-out event...", flush=True)
            baseline_raw, outer_gate, outer_candidates = common.raw_components(outer_train, held_out, depth_nodes, account_age, offsets, frame, temporal_columns, embeddings, common.base.SEED + 100000 + outer_index)
            outer_quiet = common.recency_values(held_out) >= outer_gate["threshold"]
            baseline_score = common.base.apply_calibration(baseline_raw, outer_quiet, baseline_calibrators)
            boosted_score = common.base.apply_calibration(outer_candidates[chosen], outer_quiet, candidate_calibrators[chosen])
            scores = pd.DataFrame({"thread_id": [str(graph.thread_id) for graph in held_out], "event_id": [str(graph.event_id) for graph in held_out], "preventable_impact": [float(graph.preventable_impact.item()) for graph in held_out], "baseline_calibrated_score": baseline_score, "soft_boost_calibrated_score": boosted_score, "quiet": outer_quiet, "selected_pool_fraction": chosen[0], "selected_bonus": chosen[1]})
            all_scores.append(scores)
            all_curves.extend(common.base.curve_rows(scores, "baseline_calibrated_score", "calibrated_active_quiet_rf"))
            all_curves.extend(common.base.curve_rows(scores, "soft_boost_calibrated_score", "quiet_tier_soft_boost"))
            gates.append({"outer_event": outer_event, **outer_gate})
            fold_details.append({"outer_event": outer_event, "selected_pool_fraction": chosen[0], "selected_bonus": chosen[1], "baseline_calibrators": baseline_details, "soft_boost_calibrators": candidate_details[chosen]})
            pd.concat(all_scores, ignore_index=True).to_csv(output / "outer_scores_partial.csv", index=False)
            (output / "partial_result.json").write_text(json.dumps({"completed_outer_events": fold_details}, indent=2), encoding="utf-8")
        print("[3/6] Computing eligible-event budget summaries...", flush=True)
        scores_frame = pd.concat(all_scores, ignore_index=True); curves = pd.DataFrame(all_curves)
        summary = curves.groupby(["model", "budget"], as_index=False).agg(eligible_events=("event_id", "nunique"), mean_oracle_efficiency=("oracle_efficiency", "mean"), mean_model_reduction=("model_reduction", "mean"))
        comparison = summary.pivot(index="budget", columns="model", values=["mean_model_reduction", "mean_oracle_efficiency"])
        comparison.columns = [f"{metric}_{model}" for metric, model in comparison.columns]; comparison = comparison.reset_index()
        comparison["soft_boost_minus_baseline_reduction"] = comparison["mean_model_reduction_quiet_tier_soft_boost"] - comparison["mean_model_reduction_calibrated_active_quiet_rf"]
        comparison["soft_boost_minus_baseline_efficiency"] = comparison["mean_oracle_efficiency_quiet_tier_soft_boost"] - comparison["mean_oracle_efficiency_calibrated_active_quiet_rf"]
        print("[4/6] Validating score/output integrity...", flush=True)
        if scores_frame.thread_id.duplicated().any() or not np.isfinite(scores_frame[["baseline_calibrated_score", "soft_boost_calibrated_score", "preventable_impact"]].to_numpy(float)).all(): raise ValueError("full soft-boost integrity failure")
        print("[5/6] Writing full policy evidence...", flush=True)
        scores_frame.to_csv(output / "outer_scores.csv", index=False); curves.to_csv(output / "per_event_budget_curve.csv", index=False); summary.to_csv(output / "eligible_event_budget_summary.csv", index=False); comparison.to_csv(output / "soft_boost_vs_baseline_budget_comparison.csv", index=False); pd.concat(all_selection, ignore_index=True).to_csv(output / "inner_config_selection.csv", index=False); pd.DataFrame(gates).to_csv(output / "outer_gate_details.csv", index=False)
        print("[6/6] Finalizing run record...", flush=True)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "eligible_events": eligible, "fold_details": fold_details, "output_files": ["outer_scores.csv", "per_event_budget_curve.csv", "eligible_event_budget_summary.csv", "soft_boost_vs_baseline_budget_comparison.csv", "inner_config_selection.csv", "outer_gate_details.csv"]}); common.write_record(output, record)
        print(f"SUCCESS: full quiet-tier soft-boost policy saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}); common.write_record(output, record); print(f"FAILURE: full soft-boost policy preserved at: {output.resolve()}", flush=True); raise


if __name__ == "__main__":
    main()
