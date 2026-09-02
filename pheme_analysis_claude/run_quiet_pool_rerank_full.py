"""Full eligible-event outer-LOEO two-stage quiet pool + within-pool rerank.

Do not launch this from an automated agent. This is the long-running
foreground script referenced by PROJECT_HANDOFF_20260902_QUIET_POLICY.md's
successor experiment; run it manually after reviewing the smoke evidence in
``experiments/*_quiet_pool_rerank_smoke/``.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import traceback

import numpy as np
import pandas as pd

import quiet_pool_rerank_common as common
import run_quiet_pool_rerank_smoke as smoke


smoke.RUN_MODE = "full"
common.hybrid_common.TIER_TREES = 300


def main() -> None:
    output = smoke.OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_quiet_pool_rerank_full")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "mode": "full",
        "dataset": str(common.DATASET),
        "split": "seven eligible outer LOEO events; inner LOEO within outer-train selects pool_fraction and utility_weight",
        "cutoff_seconds": 1800,
        "pool_fractions": list(common.POOL_FRACTIONS), "utility_weights": list(common.UTILITY_WEIGHTS),
        "guard_budgets": list(common.GUARD_BUDGETS), "guard_tolerance": common.GUARD_TOLERANCE,
        "baseline_config": {"pool_fraction": common.BASELINE_CONFIG[0], "utility_weight": common.BASELINE_CONFIG[1]},
        "selection_metric": "per-inner-fold MEDIAN quiet efficiency; guard on every budget (K=1,3,5,10,20,50,100); maximize median(K=1,K=3) gain over baseline",
        "research_safety": common.safety_statement(),
        "seed": common.SEED,
    }
    smoke.write_record(output, record)
    try:
        print("Starting full two-stage quiet pool + rerank policy.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/6] Loading and validating protected inputs...", flush=True)
        graphs, events, eligible = common.load_graphs()
        depth_nodes, account_age = common.load_safe_nodes()
        offsets = common.load_node_offsets()
        frame, temporal_columns, embeddings = common.graph_inputs(graphs)
        all_curves, all_scores, all_selection, fold_details = [], [], [], []
        print("[2/6] Running eligible outer LOEO folds...", flush=True)
        for outer_index, outer_event in enumerate(eligible, 1):
            print(f"  outer {outer_index}/{len(eligible)}: {outer_event}", flush=True)
            outer_train = [graph for graph in graphs if str(graph.event_id) != outer_event]
            held_out = [graph for graph in graphs if str(graph.event_id) == outer_event]
            inner_events = sorted({str(graph.event_id) for graph in outer_train})
            inner_rows = []
            for inner_index, inner_event in enumerate(inner_events, 1):
                print(f"    inner {inner_index}/{len(inner_events)}: validation={inner_event}", flush=True)
                inner_train = [graph for graph in outer_train if str(graph.event_id) != inner_event]
                validation = [graph for graph in outer_train if str(graph.event_id) == inner_event]
                rows, *_ = smoke.fold_candidate_rows(
                    inner_train, validation, depth_nodes, account_age, offsets,
                    frame, temporal_columns, embeddings,
                    common.SEED + outer_index * 1000 + inner_index, inner_event,
                )
                if rows:
                    inner_rows.extend(rows)
            print("    selecting pool_fraction/utility_weight from inner OOF efficiency (median across folds)...", flush=True)
            inner_frame = pd.DataFrame(inner_rows)
            pool_fraction, utility_weight, selection_summary = common.select_config(inner_frame)
            selection_summary = selection_summary.copy()
            selection_summary["outer_event"] = outer_event
            all_selection.append(selection_summary)
            print(f"    selected pool_fraction={pool_fraction:.2f} utility_weight={utility_weight:.2f}; scoring held-out event...", flush=True)
            _, ids, impact, tier_prob_high, quiet_utility = smoke.fold_candidate_rows(
                outer_train, held_out, depth_nodes, account_age, offsets,
                frame, temporal_columns, embeddings, common.SEED + 100000 + outer_index, outer_event,
            )
            score, pool_mask = common.pool_rerank_scores(quiet_utility, tier_prob_high, ids, pool_fraction, utility_weight)
            baseline_score, _ = common.pool_rerank_scores(quiet_utility, tier_prob_high, ids, *common.BASELINE_CONFIG)
            selected_curve = common.quiet_efficiency_curve(ids, impact, score)
            baseline_curve = common.quiet_efficiency_curve(ids, impact, baseline_score)
            for budget in common.BUDGETS:
                all_curves.append({"outer_event": outer_event, "budget": budget, "model": "quiet_pool_rerank", "efficiency": selected_curve[budget]})
                all_curves.append({"outer_event": outer_event, "budget": budget, "model": "quiet_utility_rf_only", "efficiency": baseline_curve[budget]})
            all_scores.append(pd.DataFrame({
                "thread_id": ids, "event_id": outer_event, "preventable_impact": impact,
                "tier_prob_high": tier_prob_high, "quiet_utility_raw": quiet_utility, "pool_member": pool_mask,
                "selected_pool_fraction": pool_fraction, "selected_utility_weight": utility_weight,
                "selected_pool_rerank_score": score, "baseline_utility_only_score": baseline_score,
            }))
            fold_details.append({
                "outer_event": outer_event, "selected_pool_fraction": pool_fraction, "selected_utility_weight": utility_weight,
                "outer_quiet_threads": len(ids), "outer_pool_threads": int(pool_mask.sum()),
            })
            pd.concat(all_scores, ignore_index=True).to_csv(output / "outer_scores_partial.csv", index=False)
            (output / "partial_result.json").write_text(json.dumps({"completed_outer_events": fold_details}, indent=2), encoding="utf-8")
        print("[3/6] Aggregating eligible-event efficiency curves...", flush=True)
        curves = pd.DataFrame(all_curves)
        summary = curves.groupby(["model", "budget"], as_index=False)["efficiency"].mean().rename(columns={"efficiency": "mean_quiet_efficiency"})
        comparison = summary.pivot(index="budget", columns="model", values="mean_quiet_efficiency").reset_index()
        comparison["pool_rerank_minus_utility_only"] = comparison["quiet_pool_rerank"] - comparison["quiet_utility_rf_only"]
        print("[4/6] Validating score/output integrity...", flush=True)
        scores_frame = pd.concat(all_scores, ignore_index=True)
        if scores_frame.thread_id.duplicated().any() or not np.isfinite(
            scores_frame[["selected_pool_rerank_score", "baseline_utility_only_score", "preventable_impact"]].to_numpy(float)
        ).all():
            raise ValueError("Full quiet pool-rerank integrity failure")
        print("[5/6] Writing full policy evidence...", flush=True)
        scores_frame.to_csv(output / "outer_scores.csv", index=False)
        curves.to_csv(output / "per_event_efficiency_curve.csv", index=False)
        summary.to_csv(output / "eligible_event_efficiency_summary.csv", index=False)
        comparison.to_csv(output / "pool_rerank_vs_utility_only_comparison.csv", index=False)
        pd.concat(all_selection, ignore_index=True).to_csv(output / "inner_selection_summary.csv", index=False)
        print("[6/6] Finalizing run record...", flush=True)
        record.update({
            "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
            "eligible_events": eligible, "fold_details": fold_details,
            "output_files": [
                "outer_scores.csv", "per_event_efficiency_curve.csv", "eligible_event_efficiency_summary.csv",
                "pool_rerank_vs_utility_only_comparison.csv", "inner_selection_summary.csv",
            ],
        })
        smoke.write_record(output, record)
        print(f"SUCCESS: full quiet pool + rerank policy saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({
            "status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
        })
        smoke.write_record(output, record)
        print(f"FAILURE: full quiet pool + rerank preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
