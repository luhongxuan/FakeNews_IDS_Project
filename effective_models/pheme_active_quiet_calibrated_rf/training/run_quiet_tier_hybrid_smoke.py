"""Leakage-safe active/quiet utility + quiet-tier hybrid policy smoke.

This bounded smoke runs one outer event and one inner validation fold.  The
same outer-train-only path fits the recency gate, quiet tier models, hybrid
weight, expert calibrators, and final held-out policy scores.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
import torch

from pheme_account_age_v5_common import (
    DATASET, ELIGIBLE_MIN_THREADS, SEED, load_graphs, load_node_offsets,
    load_safe_nodes,
)
from pheme_quiet_expert_common import gated_predict, recency_values
from pheme_quiet_expert_calibration_common import fit_calibrators
from quiet_tier_hybrid_score import hybrid_quiet_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TIER_ROOT = ROOT / "effective_models" / "pheme_quiet_tier_group_search"
sys.path.insert(0, str(TIER_ROOT))
from strict30_protected_feature_registry import feature_group  # noqa: E402


RUN_MODE = "smoke"
TIER_TREES = 50
OUTER_EVENT = "charliehebdo"
SOURCE_WEIGHT = 0.50  # predeclared before this full-policy experiment
UTILITY_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)
BUDGETS = (1, 3, 5, 10, 20, 50, 100)
MATERIALIZATION = TIER_ROOT / "experiments" / "20260902_030441_849111_strict30_protected_feature_materialization"
FEATURE_TABLE = MATERIALIZATION / "strict30_protected_snapshot_features.csv"
MATERIALIZATION_RECORD = MATERIALIZATION / "run_record.json"
OUT_ROOT = HERE.parent / "experiments"
LABELS = np.asarray(["high", "low", "middle"])


def write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def add_tiers(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, event in frame.groupby("event_id", sort=True):
        ranked = event.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable").copy()
        high = max(1, math.ceil(len(ranked) * 0.20))
        middle = max(high + 1, math.ceil(len(ranked) * 0.50))
        ranked["tier"] = "low"
        ranked.iloc[:high, ranked.columns.get_loc("tier")] = "high"
        ranked.iloc[high:middle, ranked.columns.get_loc("tier")] = "middle"
        parts.append(ranked)
    return pd.concat(parts, ignore_index=True)


def aligned_probabilities(model: ExtraTreesClassifier, matrix: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(matrix)
    result = np.zeros((len(matrix), len(LABELS)), dtype=float)
    for index, label in enumerate(model.classes_):
        result[:, int(np.where(LABELS == label)[0][0])] = raw[:, index]
    return result


def tier_probabilities(
    train_graphs: list, test_graphs: list, frame: pd.DataFrame,
    temporal_columns: list[str], embeddings: dict[str, np.ndarray], seed: int,
) -> np.ndarray:
    train_ids = [str(graph.thread_id) for graph in train_graphs]
    test_ids = [str(graph.thread_id) for graph in test_graphs]
    train = add_tiers(frame.set_index("thread_id").loc[train_ids].reset_index())
    test = frame.set_index("thread_id").loc[test_ids].reset_index()
    train_embedding = np.vstack([embeddings[value] for value in train.thread_id])
    test_embedding = np.vstack([embeddings[value] for value in test.thread_id])
    scaler = StandardScaler().fit(train_embedding)
    pca = PCA(n_components=16, random_state=seed).fit(scaler.transform(train_embedding))
    source_train = pca.transform(scaler.transform(train_embedding))
    source_test = pca.transform(scaler.transform(test_embedding))
    source = ExtraTreesClassifier(n_estimators=TIER_TREES, min_samples_leaf=3, max_features=0.7, class_weight="balanced", n_jobs=1, random_state=seed)
    temporal = ExtraTreesClassifier(n_estimators=TIER_TREES, min_samples_leaf=3, max_features=0.7, class_weight="balanced", n_jobs=1, random_state=seed + 1)
    source.fit(source_train, train.tier)
    temporal.fit(train[temporal_columns].to_numpy(np.float32), train.tier)
    source_probability = aligned_probabilities(source, source_test)
    temporal_probability = aligned_probabilities(temporal, test[temporal_columns].to_numpy(np.float32))
    return SOURCE_WEIGHT * source_probability + (1.0 - SOURCE_WEIGHT) * temporal_probability


def graph_inputs(graphs: list) -> tuple[pd.DataFrame, list[str], dict[str, np.ndarray]]:
    materialization = json.loads(MATERIALIZATION_RECORD.read_text(encoding="utf-8"))
    if materialization.get("status") != "complete":
        raise ValueError("Protected feature materialization is not complete")
    frame = pd.read_csv(FEATURE_TABLE, dtype={"thread_id": str, "event_id": str})
    if len(frame) != 2402 or frame.thread_id.duplicated().any():
        raise ValueError("Invalid protected feature identities")
    temporal_columns = [column for column in frame if feature_group(column) in {"activity_level", "temporal_dynamics"}]
    embeddings = {}
    for graph in graphs:
        source = torch.nonzero(graph.is_source_mask, as_tuple=False).flatten()
        if source.numel() != 1 or graph.x.shape[1] <= 13:
            raise ValueError(f"Invalid source embedding for {graph.thread_id}")
        root = int(source.item())
        vectors = graph.x[:, 13:].detach().cpu().numpy().astype(np.float32)
        replies = np.delete(vectors, root, axis=0)
        embeddings[str(graph.thread_id)] = np.concatenate([vectors[root], replies.mean(axis=0) if len(replies) else np.zeros_like(vectors[root])])
    return frame, temporal_columns, embeddings


def raw_components(
    train: list, test: list, depth_nodes, account_age, offsets,
    frame: pd.DataFrame, temporal_columns: list[str], embeddings: dict[str, np.ndarray], seed: int,
) -> tuple[np.ndarray, dict, dict[float, np.ndarray]]:
    utility, gate, _ = gated_predict(train, test, depth_nodes, account_age, offsets)
    train_quiet = recency_values(train) >= gate["threshold"]
    test_quiet = recency_values(test) >= gate["threshold"]
    quiet_train = [graph for graph, include in zip(train, train_quiet) if include]
    quiet_test = [graph for graph, include in zip(test, test_quiet) if include]
    probability = tier_probabilities(quiet_train, quiet_test, frame, temporal_columns, embeddings, seed) if quiet_test else np.empty((0, 3))
    candidates = {}
    for weight in UTILITY_WEIGHTS:
        score = np.asarray(utility, dtype=np.float32).copy()
        if quiet_test:
            ids = [str(graph.thread_id) for graph in quiet_test]
            score[test_quiet] = hybrid_quiet_score(score[test_quiet], probability, ids, weight)
        candidates[weight] = score
    return utility, gate, candidates


def calibrator_frame(rows: list[dict]) -> pd.DataFrame:
    result = pd.DataFrame(rows)
    event_count = result.validation_event.nunique()
    result["event_weight"] = result.groupby("validation_event")["thread_id"].transform(
        lambda values: len(result) / (event_count * len(values))
    )
    return result


def apply_calibration(raw: np.ndarray, quiet: np.ndarray, calibrators: dict) -> np.ndarray:
    output = np.empty(len(raw), dtype=np.float32)
    for expert, mask in (("quiet", quiet), ("active", ~quiet)):
        if mask.any():
            mapping = calibrators[expert]
            output[mask] = mapping["coefficient"] * raw[mask] + mapping["intercept"]
    return output


def reduction_at_10(frame: pd.DataFrame, score_column: str) -> float:
    values = []
    for _, event in frame.groupby("event_id", sort=True):
        ranked = event.sort_values([score_column, "thread_id"], ascending=[False, True], kind="stable")
        total = float(event.preventable_impact.sum())
        values.append(float(ranked.head(min(10, len(event))).preventable_impact.sum()) / total if total else 0.0)
    return float(np.mean(values))


def quiet_high_overlap(frame: pd.DataFrame) -> float:
    """Outcome-only inner selection metric for quiet candidate prioritization."""
    quiet = frame.loc[frame.expert.eq("quiet")].copy()
    values = []
    for _, event in quiet.groupby("validation_event", sort=True):
        k = max(1, math.ceil(len(event) * 0.20))
        predicted = set(event.sort_values(["raw_score", "thread_id"], ascending=[False, True], kind="stable").head(k).thread_id)
        oracle = set(event.sort_values(["target", "thread_id"], ascending=[False, True], kind="stable").head(k).thread_id)
        values.append(len(predicted & oracle) / k)
    return float(np.mean(values)) if values else 0.0


def curve_rows(frame: pd.DataFrame, score_column: str, model: str) -> list[dict]:
    ranked = frame.sort_values([score_column, "thread_id"], ascending=[False, True], kind="stable")
    oracle = frame.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable")
    total = float(frame.preventable_impact.sum())
    rows = []
    for budget in BUDGETS:
        actual = min(budget, len(frame))
        blocked = float(ranked.head(actual).preventable_impact.sum())
        optimal = float(oracle.head(actual).preventable_impact.sum())
        rows.append({"model": model, "event_id": str(frame.event_id.iloc[0]), "budget": budget, "model_blocked_impact": blocked, "model_reduction": blocked / total if total else 0.0, "oracle_efficiency": blocked / optimal if optimal else 0.0})
    return rows


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_quiet_tier_hybrid_policy_smoke")
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "mode": RUN_MODE, "dataset": str(DATASET), "split": "one outer event and one outer-train inner validation event smoke", "outer_event": OUTER_EVENT, "cutoff_seconds": 1800, "source_weight": SOURCE_WEIGHT, "utility_weights": list(UTILITY_WEIGHTS), "selection_metric": "inner OOF full-policy reduction@10", "seed": SEED}
    write_record(output, record)
    try:
        print("Starting active/quiet + quiet-tier hybrid policy smoke.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/6] Loading and validating protected inputs...", flush=True)
        graphs, events, eligible = load_graphs(); depth_nodes, account_age = load_safe_nodes(); offsets = load_node_offsets()
        frame, temporal_columns, embeddings = graph_inputs(graphs)
        outer_train = [graph for graph in graphs if str(graph.event_id) != OUTER_EVENT]
        held_out = [graph for graph in graphs if str(graph.event_id) == OUTER_EVENT]
        inner_event = next(event for event in eligible if event != OUTER_EVENT)
        inner_train = [graph for graph in outer_train if str(graph.event_id) != inner_event]
        validation = [graph for graph in outer_train if str(graph.event_id) == inner_event]
        print(f"[2/6] Building inner OOF components: validation={inner_event}...", flush=True)
        baseline_inner, inner_gate, candidates = raw_components(inner_train, validation, depth_nodes, account_age, offsets, frame, temporal_columns, embeddings, SEED)
        inner_quiet = recency_values(validation) >= inner_gate["threshold"]
        base_rows = []
        candidate_rows = {weight: [] for weight in UTILITY_WEIGHTS}
        for index, graph in enumerate(validation):
            common = {"validation_event": inner_event, "thread_id": str(graph.thread_id), "expert": "quiet" if inner_quiet[index] else "active", "target": float(graph.preventable_y.item())}
            base_rows.append({**common, "raw_score": float(baseline_inner[index])})
            for weight in UTILITY_WEIGHTS: candidate_rows[weight].append({**common, "raw_score": float(candidates[weight][index])})
        baseline_oof = calibrator_frame(base_rows); baseline_calibrators, _ = fit_calibrators(baseline_oof)
        print("[3/6] Selecting hybrid utility weight from inner OOF reduction@10...", flush=True)
        selection = []
        hybrid_calibrators = {}
        for weight in UTILITY_WEIGHTS:
            oof = calibrator_frame(candidate_rows[weight]); calibrators, _ = fit_calibrators(oof); hybrid_calibrators[weight] = calibrators
            quiet = oof.expert.eq("quiet").to_numpy(); oof["score"] = apply_calibration(oof.raw_score.to_numpy(float), quiet, calibrators)
            oof["event_id"] = oof.validation_event; oof["preventable_impact"] = np.expm1(oof.target)
            selection.append({"utility_weight": weight, "inner_quiet_high_overlap": quiet_high_overlap(oof), "inner_reduction_at_10": reduction_at_10(oof, "score")})
        selection_frame = pd.DataFrame(selection).sort_values(["inner_quiet_high_overlap", "inner_reduction_at_10", "utility_weight"], ascending=[False, False, False], kind="stable")
        selected_weight = float(selection_frame.iloc[0].utility_weight)
        print(f"  selected utility_weight={selected_weight:.2f}", flush=True)
        print("[4/6] Fitting final outer models and calibrating held-out scores...", flush=True)
        baseline_outer, outer_gate, outer_candidates = raw_components(outer_train, held_out, depth_nodes, account_age, offsets, frame, temporal_columns, embeddings, SEED + 1000)
        outer_quiet = recency_values(held_out) >= outer_gate["threshold"]
        baseline_score = apply_calibration(baseline_outer, outer_quiet, baseline_calibrators)
        hybrid_score = apply_calibration(outer_candidates[selected_weight], outer_quiet, hybrid_calibrators[selected_weight])
        scores = pd.DataFrame({"thread_id": [str(graph.thread_id) for graph in held_out], "event_id": [str(graph.event_id) for graph in held_out], "preventable_impact": [float(graph.preventable_impact.item()) for graph in held_out], "baseline_calibrated_score": baseline_score, "hybrid_calibrated_score": hybrid_score, "quiet": outer_quiet})
        print("[5/6] Computing held-out budget curves...", flush=True)
        curves = pd.concat([pd.DataFrame(curve_rows(scores, "baseline_calibrated_score", "calibrated_active_quiet_rf")), pd.DataFrame(curve_rows(scores, "hybrid_calibrated_score", "quiet_tier_hybrid"))], ignore_index=True)
        print("[6/6] Writing smoke evidence...", flush=True)
        selection_frame.to_csv(output / "inner_weight_selection.csv", index=False); scores.to_csv(output / "outer_scores.csv", index=False); curves.to_csv(output / "outer_budget_curve.csv", index=False)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "inner_validation_event": inner_event, "selected_utility_weight": selected_weight, "outer_gate": outer_gate, "metrics": curves.to_dict(orient="records"), "output_files": ["inner_weight_selection.csv", "outer_scores.csv", "outer_budget_curve.csv"]})
        write_record(output, record)
        print(f"SUCCESS: active/quiet + quiet-tier hybrid policy smoke saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}); write_record(output, record)
        print(f"FAILURE: hybrid policy smoke preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
