"""Leakage-safe smoke for the bounded quiet-RF plus quiet-head blend.

The head feature combination is fixed in this smoke.  Its purpose is to verify
the complete train-only score path before the separately prepared nested search
selects any head combination or blend setting.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from pheme_account_age_v5_common import DATASET, SEED, load_graphs, load_node_offsets, load_safe_nodes
from pheme_quiet_expert_calibration_common import fit_calibrators
from pheme_quiet_expert_common import gated_predict, recency_values
from quiet_head_blend_score import bounded_head_blend
from quiet_head_combination_common import columns_by_group, fit_score
from run_quiet_tier_hybrid_smoke import apply_calibration, calibrator_frame, curve_rows, graph_inputs


HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parent / "experiments"
OUTER_EVENT = "charliehebdo"
# Fixed before execution.  Three OOF events supply enough active/quiet rows for
# the project's existing two-expert calibrator without using the outer event.
SMOKE_VALIDATION_EVENTS = ("ferguson", "germanwings-crash", "putinmissing")
HEAD_GROUPS = ("source_reply_embedding", "activity_level", "temporal_dynamics", "topology")
HEAD_TREES = 50
BUDGETS_FOR_SELECTION = (1, 3, 5, 10, 20, 50)
BLEND_CONFIGURATIONS = ((0.0, 0.0), (0.25, 0.10), (0.50, 0.10), (0.50, 0.20), (0.75, 0.20))


def write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def ids(graphs: list) -> list[str]:
    return [str(graph.thread_id) for graph in graphs]


def quiet_graphs(graphs: list, quiet: np.ndarray) -> list:
    return [graph for graph, include in zip(graphs, quiet) if include]


def head_predictions(train: list, test: list, train_quiet: np.ndarray, test_quiet: np.ndarray, frame: pd.DataFrame, columns: dict, embeddings: dict[str, np.ndarray], seed: int) -> np.ndarray:
    """Fit only on quiet outer/inner-train rows and return aligned quiet scores."""
    train_ids, test_ids = ids(quiet_graphs(train, train_quiet)), ids(quiet_graphs(test, test_quiet))
    if len(train_ids) < 33:
        raise ValueError(f"Too few quiet-train rows for head PCA: {len(train_ids)}")
    if not test_ids:
        return np.empty(0, dtype=np.float32)
    indexed = frame.set_index("thread_id", verify_integrity=True)
    train_frame = indexed.loc[train_ids].reset_index()
    test_frame = indexed.loc[test_ids].reset_index()
    return fit_score(train_frame, test_frame, HEAD_GROUPS, columns, embeddings, seed, HEAD_TREES)


def raw_scores(train: list, test: list, depth_nodes, account_age, offsets, frame: pd.DataFrame, columns: dict, embeddings: dict[str, np.ndarray], seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    baseline, gate, _ = gated_predict(train, test, depth_nodes, account_age, offsets)
    train_quiet = recency_values(train) >= float(gate["threshold"])
    test_quiet = recency_values(test) >= float(gate["threshold"])
    head = head_predictions(train, test, train_quiet, test_quiet, frame, columns, embeddings, seed)
    result = np.asarray(baseline, dtype=np.float32).copy()
    if test_quiet.any():
        result[test_quiet] = head
    return np.asarray(baseline, dtype=np.float32), result, {"gate": gate, "quiet": test_quiet}


def oof_frame(graphs: list, raw: np.ndarray, quiet: np.ndarray, validation_event: str) -> pd.DataFrame:
    rows = []
    for graph, score, is_quiet in zip(graphs, raw, quiet):
        rows.append({"validation_event": validation_event, "thread_id": str(graph.thread_id), "expert": "quiet" if is_quiet else "active", "raw_score": float(score), "target": float(graph.preventable_y.item())})
    return calibrator_frame(rows)


def selection_metrics(oof: pd.DataFrame, calibrators: dict) -> dict[int, float]:
    scored = oof.copy()
    scored["score"] = apply_calibration(scored.raw_score.to_numpy(), scored.expert.eq("quiet").to_numpy(), calibrators)
    scored["event_id"] = scored.validation_event
    scored["preventable_impact"] = np.expm1(scored.target)
    results = {budget: [] for budget in BUDGETS_FOR_SELECTION}
    for _, event in scored.groupby("validation_event", sort=True):
        ranked = event.sort_values(["score", "thread_id"], ascending=[False, True], kind="stable")
        total = float(event.preventable_impact.sum())
        for budget in BUDGETS_FOR_SELECTION:
            results[budget].append(float(ranked.head(min(budget, len(event))).preventable_impact.sum()) / total if total else 0.0)
    return {budget: float(np.mean(values)) for budget, values in results.items()}


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_quiet_head_blend_smoke")
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "mode": "smoke", "model": "active_quiet_rf_bounded_quiet_head_blend", "dataset": str(DATASET), "split": "one fixed outer event and three fixed outer-train OOF validation events", "outer_event": OUTER_EVENT, "validation_events": list(SMOKE_VALIDATION_EVENTS), "cutoff_seconds": 1800, "seed": SEED, "head_groups_fixed_for_smoke": list(HEAD_GROUPS), "head_trees": HEAD_TREES, "blend_configurations": [list(value) for value in BLEND_CONFIGURATIONS], "safety": "All head, gate, calibration, and blend-selection fits use only inner-train or outer-train data; outer labels are evaluation-only."}
    write_record(output, record)
    try:
        print("Starting bounded quiet-head blend smoke (foreground job).", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/6] Loading protected strict-30 snapshots and validating identities...", flush=True)
        graphs, _, _ = load_graphs()
        depth_nodes, account_age = load_safe_nodes()
        offsets = load_node_offsets()
        frame, _, embeddings = graph_inputs(graphs)
        columns = columns_by_group(frame)
        outer_train = [graph for graph in graphs if str(graph.event_id) != OUTER_EVENT]
        held_out = [graph for graph in graphs if str(graph.event_id) == OUTER_EVENT]
        if not held_out or any(not any(str(graph.event_id) == event for graph in outer_train) for event in SMOKE_VALIDATION_EVENTS):
            raise ValueError("Configured smoke events are unavailable")
        print("[2/6] Producing train-only inner OOF scores for 3 fixed validation events...", flush=True)
        inner_payloads, baseline_rows = [], []
        for index, event in enumerate(SMOKE_VALIDATION_EVENTS, 1):
            print(f"  OOF {index}/{len(SMOKE_VALIDATION_EVENTS)}: validation={event}", flush=True)
            validation = [graph for graph in outer_train if str(graph.event_id) == event]
            inner_train = [graph for graph in outer_train if str(graph.event_id) != event]
            baseline_inner, head_inner, inner = raw_scores(inner_train, validation, depth_nodes, account_age, offsets, frame, columns, embeddings, SEED + index)
            inner_payloads.append((event, validation, baseline_inner, head_inner, inner["quiet"]))
            baseline_rows.extend(oof_frame(validation, baseline_inner, inner["quiet"], event).drop(columns="event_weight").to_dict(orient="records"))
        baseline_oof = calibrator_frame(baseline_rows)
        baseline_calibrators, baseline_calibration = fit_calibrators(baseline_oof)
        baseline_metrics = selection_metrics(baseline_oof, baseline_calibrators)
        print("[3/6] Testing predeclared bounded blend configurations on inner OOF only...", flush=True)
        candidates, calibrators = [], {}
        for weight, max_shift in BLEND_CONFIGURATIONS:
            rows = []
            for event, validation, baseline_inner, head_inner, quiet in inner_payloads:
                raw = baseline_inner.copy()
                if quiet.any():
                    raw[quiet] = bounded_head_blend(baseline_inner[quiet], head_inner[quiet], ids(quiet_graphs(validation, quiet)), weight, max_shift)
                rows.extend(oof_frame(validation, raw, quiet, event).drop(columns="event_weight").to_dict(orient="records"))
            oof = calibrator_frame(rows)
            candidate_calibrators, details = fit_calibrators(oof)
            metrics = selection_metrics(oof, candidate_calibrators)
            feasible = all(metrics[budget] >= baseline_metrics[budget] - 1e-12 for budget in (5, 10, 20, 50))
            candidates.append({"weight": weight, "maximum_rank_shift": max_shift, "feasible_non_degrading_k5_to_k50": feasible, **{f"reduction_at_{budget}": metrics[budget] for budget in BUDGETS_FOR_SELECTION}})
            calibrators[(weight, max_shift)] = candidate_calibrators
        selection = pd.DataFrame(candidates)
        feasible = selection.loc[selection.feasible_non_degrading_k5_to_k50]
        selected = (feasible if len(feasible) else selection.iloc[[0]]).sort_values(["reduction_at_1", "reduction_at_3", "reduction_at_50", "weight"], ascending=[False, False, False, False], kind="stable").iloc[0]
        selected_key = (float(selected.weight), float(selected.maximum_rank_shift))
        print(f"  selected inner-only configuration: weight={selected_key[0]:.2f}, max_shift={selected_key[1]:.2f}", flush=True)
        print("[4/6] Fitting outer-train models and scoring the held-out event...", flush=True)
        baseline_outer, head_outer, outer = raw_scores(outer_train, held_out, depth_nodes, account_age, offsets, frame, columns, embeddings, SEED + 1000)
        outer_quiet = outer["quiet"]
        blend_outer = baseline_outer.copy()
        if outer_quiet.any():
            outer_ids = ids(quiet_graphs(held_out, outer_quiet))
            blend_outer[outer_quiet] = bounded_head_blend(baseline_outer[outer_quiet], head_outer[outer_quiet], outer_ids, *selected_key)
        baseline_score = apply_calibration(baseline_outer, outer_quiet, baseline_calibrators)
        blend_score = apply_calibration(blend_outer, outer_quiet, calibrators[selected_key])
        scores = pd.DataFrame({"thread_id": ids(held_out), "event_id": [str(graph.event_id) for graph in held_out], "preventable_impact": [float(graph.preventable_impact.item()) for graph in held_out], "expert": np.where(outer_quiet, "quiet", "active"), "baseline_calibrated_score": baseline_score, "blend_calibrated_score": blend_score})
        print("[5/6] Computing held-out intervention curves without using held-out labels for selection...", flush=True)
        curves = pd.concat([pd.DataFrame(curve_rows(scores, "baseline_calibrated_score", "calibrated_active_quiet_rf")), pd.DataFrame(curve_rows(scores, "blend_calibrated_score", "bounded_quiet_head_blend"))], ignore_index=True)
        print("[6/6] Writing run record and audit artifacts...", flush=True)
        selection.to_csv(output / "inner_blend_selection.csv", index=False)
        scores.to_csv(output / "outer_scores.csv", index=False)
        curves.to_csv(output / "outer_budget_curve.csv", index=False)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "outer_gate": outer["gate"], "baseline_calibration": baseline_calibration, "selected_configuration": {"weight": selected_key[0], "maximum_rank_shift": selected_key[1]}, "baseline_inner_metrics": baseline_metrics, "outer_metrics": curves.to_dict(orient="records"), "output_files": ["inner_blend_selection.csv", "outer_scores.csv", "outer_budget_curve.csv"]})
        write_record(output, record)
        print(f"SUCCESS: bounded quiet-head blend smoke saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
        write_record(output, record)
        print(f"FAILURE: bounded quiet-head blend smoke preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
