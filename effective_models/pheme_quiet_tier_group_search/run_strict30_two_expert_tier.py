"""Strict-30 quiet three-tier classifier with source and temporal experts.

The two experts are deliberately narrow: (1) source plus observed-reply
semantic embeddings, and (2) observable activity/temporal dynamics.  Inner
LOEO chooses only the predeclared probability-fusion weight; no outer-event
label is used for that choice.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import torch

from strict30_protected_feature_registry import STRICT30_ROOT, feature_group


RUN_MODE = "full"  # Explicitly authorized by the user on 2026-09-02.
SEED = 42
QUIET_QUANTILE = 0.60
HIGH_FRACTION = 0.20
MIDDLE_CUMULATIVE_FRACTION = 0.50
FUSION_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)  # weight assigned to source expert
SMOKE_OUTER_EVENT = "charliehebdo"
SMOKE_INNER_EVENTS = 2
EXPERIMENT_NAME = "strict30_quiet_two_expert_tier"
EXTRA_TEMPORAL_PREFIXES: tuple[str, ...] = ()
EMBEDDING_MODE = "source_reply_mean"

BASE = Path(__file__).resolve().parent
MATERIALIZATION = BASE / "experiments" / "20260902_030441_849111_strict30_protected_feature_materialization"
FEATURE_TABLE = MATERIALIZATION / "strict30_protected_snapshot_features.csv"
MATERIALIZATION_RECORD = MATERIALIZATION / "run_record.json"
DATASET = STRICT30_ROOT / "pheme_graphs_roberta_30min_replyv4_preventableimpact_semantic.pt"
OUT_ROOT = BASE / "experiments"
LABELS = np.asarray(["high", "low", "middle"])


def _write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def _compose_embedding(source: np.ndarray, replies: np.ndarray) -> np.ndarray:
    """Produce an outcome-free source/reply representation from a strict snapshot."""
    mean = replies.mean(axis=0) if len(replies) else np.zeros_like(source)
    if EMBEDDING_MODE == "source_reply_mean":
        return np.concatenate([source, mean])
    if EMBEDDING_MODE == "source_reply_interactions":
        dispersion = replies.std(axis=0) if len(replies) else np.zeros_like(source)
        return np.concatenate([source, mean, source - mean, source * mean, dispersion])
    raise ValueError(f"unknown embedding mode: {EMBEDDING_MODE}")


def _event_tiers(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, event in frame.groupby("event_id", sort=True):
        ranked = event.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable").copy()
        high = max(1, math.ceil(len(ranked) * HIGH_FRACTION))
        middle = max(high + 1, math.ceil(len(ranked) * MIDDLE_CUMULATIVE_FRACTION))
        ranked["tier"] = "low"
        ranked.iloc[:high, ranked.columns.get_loc("tier")] = "high"
        ranked.iloc[high:middle, ranked.columns.get_loc("tier")] = "middle"
        parts.append(ranked)
    return pd.concat(parts, ignore_index=True)


def _quiet_partitions(frame: pd.DataFrame, recency: dict[str, float], train_events: list[str], test_event: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    raw_train = frame.loc[frame.event_id.isin(train_events)].copy()
    raw_test = frame.loc[frame.event_id.eq(test_event)].copy()
    threshold = float(np.quantile(raw_train.thread_id.map(recency), QUIET_QUANTILE))
    train = raw_train.loc[raw_train.thread_id.map(recency) >= threshold].copy()
    test = raw_test.loc[raw_test.thread_id.map(recency) >= threshold].copy()
    if train.empty or test.empty:
        raise ValueError(f"{test_event}: empty quiet partition")
    return train, test, threshold


def _high_overlap(frame: pd.DataFrame) -> float:
    values = []
    for _, event in frame.groupby("event_id", sort=True):
        k = max(1, math.ceil(len(event) * HIGH_FRACTION))
        selected = event.sort_values(["prob_high", "thread_id"], ascending=[False, True], kind="stable").head(k)
        values.append(float(selected.tier.eq("high").mean()))
    return float(np.mean(values))


def _probabilities(model: ExtraTreesClassifier, matrix: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(matrix)
    result = np.zeros((len(matrix), len(LABELS)), dtype=float)
    for index, label in enumerate(model.classes_):
        result[:, int(np.where(LABELS == label)[0][0])] = raw[:, index]
    return result


def _fit_experts(train: pd.DataFrame, test: pd.DataFrame, temporal_columns: list[str], embeddings: dict[str, np.ndarray], seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    train = _event_tiers(train)
    evaluated = _event_tiers(test)
    train_embeddings = np.vstack([embeddings[thread_id] for thread_id in train.thread_id])
    test_embeddings = np.vstack([embeddings[thread_id] for thread_id in evaluated.thread_id])
    scaler = StandardScaler().fit(train_embeddings)
    pca = PCA(n_components=16, random_state=seed).fit(scaler.transform(train_embeddings))
    source_train = pca.transform(scaler.transform(train_embeddings))
    source_test = pca.transform(scaler.transform(test_embeddings))
    temporal_train = train[temporal_columns].to_numpy(np.float32)
    temporal_test = evaluated[temporal_columns].to_numpy(np.float32)
    trees = 50 if RUN_MODE == "smoke" else 300
    source_model = ExtraTreesClassifier(n_estimators=trees, min_samples_leaf=3, max_features=0.7, class_weight="balanced", n_jobs=1, random_state=seed)
    temporal_model = ExtraTreesClassifier(n_estimators=trees, min_samples_leaf=3, max_features=0.7, class_weight="balanced", n_jobs=1, random_state=seed + 1)
    source_model.fit(source_train, train.tier)
    temporal_model.fit(temporal_train, train.tier)
    return evaluated, _probabilities(source_model, source_test), _probabilities(temporal_model, temporal_test)


def _evaluate_weight(evaluated: pd.DataFrame, source_prob: np.ndarray, temporal_prob: np.ndarray, source_weight: float) -> tuple[dict[str, float], pd.DataFrame]:
    probabilities = source_weight * source_prob + (1.0 - source_weight) * temporal_prob
    result = evaluated.copy()
    result["prob_high"] = probabilities[:, 0]
    result["predicted_tier"] = LABELS[probabilities.argmax(axis=1)]
    metrics = {
        "high_top_fraction_overlap": _high_overlap(result),
        "tier_macro_f1": float(f1_score(result.tier, result.predicted_tier, labels=LABELS.tolist(), average="macro", zero_division=0)),
    }
    return metrics, result


def _select_weight(frame: pd.DataFrame, recency: dict[str, float], embeddings: dict[str, np.ndarray], temporal_columns: list[str], outer_event: str) -> tuple[float, pd.DataFrame]:
    inner_events = sorted(event for event in frame.event_id.unique() if event != outer_event)
    if RUN_MODE == "smoke":
        inner_events = inner_events[:SMOKE_INNER_EVENTS]
    rows = []
    for fold_index, inner_event in enumerate(inner_events, 1):
        train_events = [event for event in inner_events if event != inner_event]
        train, test, _ = _quiet_partitions(frame, recency, train_events, inner_event)
        evaluated, source_prob, temporal_prob = _fit_experts(train, test, temporal_columns, embeddings, SEED + fold_index * 10)
        for weight in FUSION_WEIGHTS:
            metrics, _ = _evaluate_weight(evaluated, source_prob, temporal_prob, weight)
            rows.append({"inner_event": inner_event, "source_weight": weight, **metrics})
    detail = pd.DataFrame(rows)
    summary = (detail.groupby("source_weight", as_index=False)
               .agg(inner_folds=("inner_event", "nunique"), inner_high_top_fraction_overlap=("high_top_fraction_overlap", "mean"), inner_tier_macro_f1=("tier_macro_f1", "mean"))
               .sort_values(["inner_high_top_fraction_overlap", "inner_tier_macro_f1", "source_weight"], ascending=[False, False, True], kind="stable"))
    return float(summary.iloc[0].source_weight), summary


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{EXPERIMENT_NAME}_{RUN_MODE}")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "mode": RUN_MODE,
        "inputs": {"clean_feature_table": str(FEATURE_TABLE.resolve()), "materialization_record": str(MATERIALIZATION_RECORD.resolve()), "protected_graphs": str(DATASET.resolve())},
        "split": "eligible-event LOEO; inner LOEO selects source-probability fusion weight; outer event is held out",
        "cutoff_seconds": 1800, "quiet_gate": "training events' 60th percentile protected recency", "tiers": "quiet event high top 20%, middle next 30%, low remaining 50% by preventable impact",
        "experts": {"source": "train-only PCA16 of predeclared source/reply snapshot embedding", "temporal": "activity_level + temporal_dynamics plus predeclared extra feature prefixes"},
        "embedding_mode": EMBEDDING_MODE,
        "extra_temporal_prefixes": list(EXTRA_TEMPORAL_PREFIXES),
        "fusion_weights": list(FUSION_WEIGHTS), "selection_metric": "inner deterministic high-top-fraction overlap, then tier macro-F1", "seed": SEED,
    }
    _write_record(output, record)
    try:
        print(f"Starting strict-30 quiet two-expert tier {RUN_MODE}.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/5] Validating clean features and eligible events...", flush=True)
        materialization = json.loads(MATERIALIZATION_RECORD.read_text(encoding="utf-8"))
        if materialization.get("status") != "complete": raise ValueError("clean materialization is not complete")
        frame = pd.read_csv(FEATURE_TABLE, dtype={"thread_id": str, "event_id": str})
        if len(frame) != 2402 or frame.thread_id.duplicated().any(): raise ValueError("invalid clean feature identities")
        temporal_columns = [column for column in frame if feature_group(column) in {"activity_level", "temporal_dynamics"} or column.startswith(EXTRA_TEMPORAL_PREFIXES)]
        if not temporal_columns or not np.isfinite(frame[temporal_columns].to_numpy(float)).all(): raise ValueError("invalid temporal/activity inputs")
        sizes = frame.groupby("event_id").size(); eligible = sorted(sizes[sizes >= 100].index.tolist())
        if len(eligible) != 7: raise ValueError("expected seven eligible events")
        frame = frame.loc[frame.event_id.isin(eligible)].copy()
        print("[2/5] Auditing protected recency and building snapshot embeddings...", flush=True)
        graphs = torch.load(DATASET, weights_only=False); recency = {}; embeddings = {}
        for index, graph in enumerate(graphs, 1):
            temporal = graph.temporal_features.detach().cpu().numpy().reshape(-1)
            if len(temporal) != 11 or not np.isfinite(temporal).all(): raise ValueError(f"{graph.thread_id}: invalid temporal snapshot")
            source_indices = torch.nonzero(graph.is_source_mask, as_tuple=False).flatten()
            if source_indices.numel() != 1 or graph.x.ndim != 2 or graph.x.shape[1] <= 13: raise ValueError(f"{graph.thread_id}: invalid source embedding")
            root = int(source_indices.item()); vectors = graph.x[:, 13:].detach().cpu().numpy().astype(np.float32); replies = np.delete(vectors, root, axis=0)
            recency[str(graph.thread_id)] = float(temporal[5]); embeddings[str(graph.thread_id)] = _compose_embedding(vectors[root], replies)
            if index % 500 == 0 or index == len(graphs): print(f"  audited {index}/{len(graphs)} snapshots", flush=True)
        if set(frame.thread_id) - set(embeddings): raise ValueError("feature threads missing snapshot embedding")
        outer_events = [SMOKE_OUTER_EVENT] if RUN_MODE == "smoke" else eligible
        all_metrics, all_predictions = [], []
        print("[3/5] Selecting fusion weights with inner LOEO and evaluating held-out events...", flush=True)
        for index, outer_event in enumerate(outer_events, 1):
            print(f"  outer event {index}/{len(outer_events)}: {outer_event}", flush=True)
            weight, summary = _select_weight(frame, recency, embeddings, temporal_columns, outer_event)
            train_events = [event for event in eligible if event != outer_event]
            train, test, threshold = _quiet_partitions(frame, recency, train_events, outer_event)
            evaluated, source_prob, temporal_prob = _fit_experts(train, test, temporal_columns, embeddings, SEED + 1000 + index)
            metrics, predictions = _evaluate_weight(evaluated, source_prob, temporal_prob, weight)
            metrics.update({"outer_event": outer_event, "selected_source_weight": weight, "selected_temporal_weight": 1.0 - weight, "quiet_threshold": threshold, "train_quiet_threads": len(train), "test_quiet_threads": len(test), "temporal_feature_count": len(temporal_columns), "source_pca_components": 16})
            summary.to_csv(output / f"inner_fusion_selection_{outer_event}.csv", index=False)
            predictions["outer_event"] = outer_event; predictions["selected_source_weight"] = weight
            all_metrics.append(metrics); all_predictions.append(predictions)
        print("[4/5] Writing fold predictions and metrics...", flush=True)
        metrics_frame = pd.DataFrame(all_metrics); predictions_frame = pd.concat(all_predictions, ignore_index=True)
        metrics_frame.to_csv(output / "outer_fold_metrics.csv", index=False); predictions_frame.to_csv(output / "outer_tier_predictions.csv", index=False)
        print("[5/5] Finalizing run record...", flush=True)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "eligible_events": eligible, "outer_events_run": outer_events, "metrics": all_metrics, "output_files": ["outer_fold_metrics.csv", "outer_tier_predictions.csv"] + [f"inner_fusion_selection_{event}.csv" for event in outer_events]})
        _write_record(output, record)
        print(f"SUCCESS: strict-30 quiet two-expert tier {RUN_MODE} saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
        _write_record(output, record)
        print(f"FAILURE: two-expert tier run preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
