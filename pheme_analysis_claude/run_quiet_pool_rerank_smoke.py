"""Two-stage quiet pool + within-pool rerank policy smoke.

One outer event, a small number of inner validation events. Fits the quiet
utility RF and tier P(high) model once per fold, then cheaply evaluates
every (pool_fraction, utility_weight) combination from
quiet_pool_rerank_common on top of that single fit.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

import quiet_pool_rerank_common as common


RUN_MODE = "smoke"
OUTER_EVENT = "charliehebdo"
SMOKE_INNER_EVENTS = 2
OUT_ROOT = Path(__file__).resolve().parent / "experiments"


def write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def fold_candidate_rows(
    train: list, test: list, depth_nodes, account_age, offsets,
    frame: pd.DataFrame, temporal_columns: list[str], embeddings: dict[str, np.ndarray],
    seed: int, fold_label: str,
):
    """Fit once per fold; return per-(config, budget) inner rows plus raw fold outputs."""
    quiet_train, quiet_test, quiet_utility, gate = common.quiet_split(train, test, depth_nodes, account_age, offsets)
    if not quiet_test:
        return None, [], np.empty(0), np.empty(0), np.empty(0)
    ids = [str(graph.thread_id) for graph in quiet_test]
    impact = np.asarray([float(graph.preventable_impact.item()) for graph in quiet_test], dtype=np.float64)
    tier_probability = common.hybrid_common.tier_probabilities(
        quiet_train, quiet_test, frame, temporal_columns, embeddings, seed
    )
    tier_prob_high = tier_probability[:, 0]
    rows = []
    for pool_fraction in common.POOL_FRACTIONS:
        for utility_weight in common.UTILITY_WEIGHTS:
            score, pool_mask = common.pool_rerank_scores(quiet_utility, tier_prob_high, ids, pool_fraction, utility_weight)
            curve = common.quiet_efficiency_curve(ids, impact, score)
            for budget, efficiency in curve.items():
                rows.append({
                    "fold": fold_label, "pool_fraction": pool_fraction, "utility_weight": utility_weight,
                    "budget": budget, "efficiency": efficiency,
                    "quiet_threads": len(ids), "pool_threads": int(pool_mask.sum()),
                })
    return rows, ids, impact, tier_prob_high, quiet_utility


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_quiet_pool_rerank_smoke")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "mode": RUN_MODE,
        "dataset": str(common.DATASET), "outer_event": OUTER_EVENT,
        "cutoff_seconds": 1800,
        "pool_fractions": list(common.POOL_FRACTIONS), "utility_weights": list(common.UTILITY_WEIGHTS),
        "guard_budgets": list(common.GUARD_BUDGETS), "guard_tolerance": common.GUARD_TOLERANCE,
        "baseline_config": {"pool_fraction": common.BASELINE_CONFIG[0], "utility_weight": common.BASELINE_CONFIG[1]},
        "selection_metric": "per-inner-fold MEDIAN quiet efficiency; guard on every budget (K=1,3,5,10,20,50,100); maximize median(K=1,K=3) gain",
        "hypothesis": (
            "A hard top-20% tier cut only enriches the high tier ~1.76x over random and misses "
            "roughly two-thirds of true high-impact quiet threads. Widening the tier-formed "
            "candidate pool to 30-80% of quiet threads, then re-ranking within that pool by the "
            "existing quiet utility RF (optionally fused with tier P(high)), should keep more "
            "true high-impact threads reachable while still concentrating them near the top of "
            "the final ranking."
        ),
        "seed": common.SEED,
    }
    write_record(output, record)
    try:
        print("Starting two-stage quiet pool + rerank smoke.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/6] Loading and validating protected inputs...", flush=True)
        graphs, events, eligible = common.load_graphs()
        depth_nodes, account_age = common.load_safe_nodes()
        offsets = common.load_node_offsets()
        frame, temporal_columns, embeddings = common.graph_inputs(graphs)
        outer_train = [graph for graph in graphs if str(graph.event_id) != OUTER_EVENT]
        held_out = [graph for graph in graphs if str(graph.event_id) == OUTER_EVENT]
        inner_events = sorted({str(graph.event_id) for graph in outer_train})[:SMOKE_INNER_EVENTS]
        print(f"[2/6] Running {len(inner_events)} inner fold(s): {inner_events}...", flush=True)
        inner_rows = []
        for index, inner_event in enumerate(inner_events, 1):
            print(f"  inner {index}/{len(inner_events)}: validation={inner_event}", flush=True)
            inner_train = [graph for graph in outer_train if str(graph.event_id) != inner_event]
            validation = [graph for graph in outer_train if str(graph.event_id) == inner_event]
            rows, *_ = fold_candidate_rows(
                inner_train, validation, depth_nodes, account_age, offsets,
                frame, temporal_columns, embeddings, common.SEED + index * 10, inner_event,
            )
            if rows:
                inner_rows.extend(rows)
        print("[3/6] Selecting pool_fraction/utility_weight from inner OOF efficiency (median across folds)...", flush=True)
        inner_frame = pd.DataFrame(inner_rows)
        pool_fraction, utility_weight, selection_summary = common.select_config(inner_frame)
        print(f"  selected pool_fraction={pool_fraction:.2f} utility_weight={utility_weight:.2f}", flush=True)
        print("[4/6] Fitting final outer models and scoring held-out event...", flush=True)
        _, ids, impact, tier_prob_high, quiet_utility = fold_candidate_rows(
            outer_train, held_out, depth_nodes, account_age, offsets,
            frame, temporal_columns, embeddings, common.SEED + 1000, OUTER_EVENT,
        )
        score, pool_mask = common.pool_rerank_scores(quiet_utility, tier_prob_high, ids, pool_fraction, utility_weight)
        baseline_score, _ = common.pool_rerank_scores(quiet_utility, tier_prob_high, ids, *common.BASELINE_CONFIG)
        print("[5/6] Computing held-out quiet efficiency curves...", flush=True)
        selected_curve = common.quiet_efficiency_curve(ids, impact, score)
        baseline_curve = common.quiet_efficiency_curve(ids, impact, baseline_score)
        scores_frame = pd.DataFrame({
            "thread_id": ids, "event_id": OUTER_EVENT, "preventable_impact": impact,
            "tier_prob_high": tier_prob_high, "quiet_utility_raw": quiet_utility, "pool_member": pool_mask,
            "selected_pool_rerank_score": score, "baseline_utility_only_score": baseline_score,
        })
        print("[6/6] Writing smoke evidence...", flush=True)
        inner_frame.to_csv(output / "inner_fold_candidate_efficiency.csv", index=False)
        selection_summary.to_csv(output / "inner_selection_summary.csv", index=False)
        scores_frame.sort_values("selected_pool_rerank_score", ascending=False).to_csv(output / "outer_scores.csv", index=False)
        pd.DataFrame([
            {"budget": budget, "selected_pool_rerank": selected_curve[budget], "baseline_utility_only": baseline_curve[budget]}
            for budget in common.BUDGETS
        ]).to_csv(output / "outer_efficiency_curve.csv", index=False)
        record.update({
            "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
            "inner_events": inner_events, "selected_pool_fraction": pool_fraction, "selected_utility_weight": utility_weight,
            "outer_quiet_threads": len(ids), "outer_pool_threads": int(pool_mask.sum()),
            "outer_efficiency_curve_selected": selected_curve, "outer_efficiency_curve_baseline_utility_only": baseline_curve,
            "research_safety": common.safety_statement(),
            "output_files": ["inner_fold_candidate_efficiency.csv", "inner_selection_summary.csv", "outer_scores.csv", "outer_efficiency_curve.csv"],
        })
        write_record(output, record)
        print(f"SUCCESS: quiet pool + rerank smoke saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({
            "status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
        })
        write_record(output, record)
        print(f"FAILURE: quiet pool + rerank smoke preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
