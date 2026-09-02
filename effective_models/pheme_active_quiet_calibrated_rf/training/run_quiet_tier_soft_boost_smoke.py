"""Smoke test for leakage-safe active/quiet policy with a soft quiet-tier boost."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

import run_quiet_tier_hybrid_smoke as base
from pheme_quiet_expert_calibration_common import fit_calibrators
from pheme_quiet_expert_common import gated_predict, recency_values
from quiet_tier_hybrid_score import soft_pool_boost_quiet_score


RUN_MODE = "smoke"
OUTER_EVENT = "charliehebdo"
POOL_FRACTIONS = (0.40, 0.50, 0.60)
BONUSES = (0.05, 0.10, 0.20)
CONFIGS = ((1.00, 0.00),) + tuple((pool, bonus) for pool in POOL_FRACTIONS for bonus in BONUSES)
OUT_ROOT = base.OUT_ROOT


def write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def raw_components(train: list, test: list, depth_nodes, account_age, offsets, frame: pd.DataFrame, temporal_columns: list[str], embeddings: dict[str, np.ndarray], seed: int) -> tuple[np.ndarray, dict, dict[tuple[float, float], np.ndarray]]:
    utility, gate, _ = gated_predict(train, test, depth_nodes, account_age, offsets)
    train_quiet = recency_values(train) >= gate["threshold"]
    test_quiet = recency_values(test) >= gate["threshold"]
    quiet_train = [graph for graph, include in zip(train, train_quiet) if include]
    quiet_test = [graph for graph, include in zip(test, test_quiet) if include]
    probability = base.tier_probabilities(quiet_train, quiet_test, frame, temporal_columns, embeddings, seed) if quiet_test else np.empty((0, 3))
    candidates = {}
    for config in CONFIGS:
        score = np.asarray(utility, dtype=np.float32).copy()
        if quiet_test:
            pool, bonus = config
            score[test_quiet] = soft_pool_boost_quiet_score(score[test_quiet], probability, [str(graph.thread_id) for graph in quiet_test], pool, bonus)
        candidates[config] = score
    return utility, gate, candidates


def calibrate(frame: pd.DataFrame, calibrators: dict) -> pd.DataFrame:
    result = frame.copy()
    quiet = result.expert.eq("quiet").to_numpy()
    result["score"] = base.apply_calibration(result.raw_score.to_numpy(float), quiet, calibrators)
    result["event_id"] = result.validation_event
    result["preventable_impact"] = np.expm1(result.target)
    return result


def quiet_global_top50_coverage(frame: pd.DataFrame) -> float:
    values = []
    for _, event in frame.groupby("validation_event", sort=True):
        oracle = event.sort_values(["target", "thread_id"], ascending=[False, True], kind="stable").head(50)
        quiet_oracle = set(oracle.loc[oracle.expert.eq("quiet"), "thread_id"])
        if not quiet_oracle:
            continue
        quiet = event.loc[event.expert.eq("quiet")].sort_values(["score", "thread_id"], ascending=[False, True], kind="stable")
        selected = set(quiet.head(len(quiet_oracle)).thread_id)
        values.append(len(selected & quiet_oracle) / len(quiet_oracle))
    return float(np.mean(values)) if values else 0.0


def reduction_at_budget(frame: pd.DataFrame, score_column: str, budget: int) -> float:
    values = []
    for _, event in frame.groupby("event_id", sort=True):
        ranked = event.sort_values([score_column, "thread_id"], ascending=[False, True], kind="stable")
        total = float(event.preventable_impact.sum())
        values.append(float(ranked.head(min(budget, len(event))).preventable_impact.sum()) / total if total else 0.0)
    return float(np.mean(values)) if values else 0.0


def append_rows(graphs: list, event: str, raw: np.ndarray, gate: dict, candidates: dict, baseline_rows: list[dict], candidate_rows: dict) -> None:
    quiet = recency_values(graphs) >= gate["threshold"]
    for index, graph in enumerate(graphs):
        shared = {"validation_event": event, "thread_id": str(graph.thread_id), "expert": "quiet" if quiet[index] else "active", "target": float(graph.preventable_y.item())}
        baseline_rows.append({**shared, "raw_score": float(raw[index])})
        for config in CONFIGS:
            candidate_rows[config].append({**shared, "raw_score": float(candidates[config][index])})


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_quiet_tier_soft_boost_smoke")
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "mode": RUN_MODE, "outer_event": OUTER_EVENT, "cutoff_seconds": 1800, "configs": [list(c) for c in CONFIGS], "selection_metric": "inner reduction@50, then quiet global Oracle Top-50 coverage", "research_safety": "All gate, tier, utility, and calibration fits use inner-train events only."}
    write_record(output, record)
    try:
        print("Starting quiet-tier soft-boost policy smoke.", flush=True); print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/6] Loading protected inputs...", flush=True)
        graphs, _, eligible = base.load_graphs(); depth_nodes, account_age = base.load_safe_nodes(); offsets = base.load_node_offsets(); frame, temporal_columns, embeddings = base.graph_inputs(graphs)
        outer_train = [g for g in graphs if str(g.event_id) != OUTER_EVENT]; held_out = [g for g in graphs if str(g.event_id) == OUTER_EVENT]
        inner_event = next(event for event in eligible if event != OUTER_EVENT); inner_train = [g for g in outer_train if str(g.event_id) != inner_event]; validation = [g for g in outer_train if str(g.event_id) == inner_event]
        print(f"[2/6] Building one inner held-out fold: {inner_event}...", flush=True)
        raw, gate, candidates = raw_components(inner_train, validation, depth_nodes, account_age, offsets, frame, temporal_columns, embeddings, base.SEED)
        baseline_rows: list[dict] = []; candidate_rows = {config: [] for config in CONFIGS}; append_rows(validation, inner_event, raw, gate, candidates, baseline_rows, candidate_rows)
        print("[3/6] Fitting inner-only calibrators and selecting soft-boost configuration...", flush=True)
        base_calibrators, _ = fit_calibrators(base.calibrator_frame(baseline_rows)); selections = []; calibrators = {}
        for config in CONFIGS:
            oof = base.calibrator_frame(candidate_rows[config]); calibrators[config], _ = fit_calibrators(oof); scored = calibrate(oof, calibrators[config])
            selections.append({"pool_fraction": config[0], "bonus": config[1], "inner_reduction_at_50": reduction_at_budget(scored, "score", 50), "inner_quiet_global_top50_coverage": quiet_global_top50_coverage(scored)})
        selection = pd.DataFrame(selections).sort_values(["inner_reduction_at_50", "inner_quiet_global_top50_coverage", "bonus"], ascending=[False, False, True], kind="stable"); chosen = (float(selection.iloc[0].pool_fraction), float(selection.iloc[0].bonus)); print(f"  selected pool={chosen[0]:.2f}, bonus={chosen[1]:.2f}", flush=True)
        print("[4/6] Fitting outer-train models and scoring held-out event...", flush=True)
        baseline_raw, outer_gate, outer_candidates = raw_components(outer_train, held_out, depth_nodes, account_age, offsets, frame, temporal_columns, embeddings, base.SEED + 1000); outer_quiet = recency_values(held_out) >= outer_gate["threshold"]
        baseline_score = base.apply_calibration(baseline_raw, outer_quiet, base_calibrators); boosted_score = base.apply_calibration(outer_candidates[chosen], outer_quiet, calibrators[chosen])
        scores = pd.DataFrame({"thread_id": [str(g.thread_id) for g in held_out], "event_id": [str(g.event_id) for g in held_out], "preventable_impact": [float(g.preventable_impact.item()) for g in held_out], "baseline_calibrated_score": baseline_score, "soft_boost_calibrated_score": boosted_score, "quiet": outer_quiet})
        print("[5/6] Computing budget curves...", flush=True); curves = pd.concat([pd.DataFrame(base.curve_rows(scores, "baseline_calibrated_score", "calibrated_active_quiet_rf")), pd.DataFrame(base.curve_rows(scores, "soft_boost_calibrated_score", "quiet_tier_soft_boost"))], ignore_index=True)
        print("[6/6] Writing smoke evidence...", flush=True); selection.to_csv(output / "inner_config_selection.csv", index=False); scores.to_csv(output / "outer_scores.csv", index=False); curves.to_csv(output / "outer_budget_curve.csv", index=False)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "inner_event": inner_event, "selected_config": list(chosen), "outer_gate": outer_gate, "output_files": ["inner_config_selection.csv", "outer_scores.csv", "outer_budget_curve.csv"]}); write_record(output, record); print(f"SUCCESS: quiet-tier soft-boost smoke saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}); write_record(output, record); print(f"FAILURE: soft-boost smoke preserved at: {output.resolve()}", flush=True); raise


if __name__ == "__main__":
    main()
