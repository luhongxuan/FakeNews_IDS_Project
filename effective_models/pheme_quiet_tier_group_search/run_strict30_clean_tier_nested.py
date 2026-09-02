"""Nested-LOEO quiet Oracle-tier experiment using clean protected features.

Set RUN_MODE to ``full`` only for a foreground user-run experiment.  The
default smoke exercises the identical cutoff/gate/tier/model path with a
small, predeclared set of feature combinations.
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
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

from strict30_protected_feature_registry import FEATURE_GROUPS, STRICT30_ROOT, feature_group


RUN_MODE = "full"  # Explicitly authorized by the user on 2026-09-02.
SEED = 42
QUIET_QUANTILE = 0.60
HIGH_FRACTION = 0.20
MIDDLE_CUMULATIVE_FRACTION = 0.50
SMOKE_OUTER_EVENT = "charliehebdo"
SMOKE_INNER_EVENTS = 2
MODEL_GROUPS = FEATURE_GROUPS + ("source_reply_embedding",)
SMOKE_CONFIGS = (
    ("source_reply_embedding", ("source_reply_embedding",)),
    ("activity_temporal_topology", ("activity_level", "temporal_dynamics", "topology")),
    ("source_embedding_plus_dynamics", ("source_reply_embedding", "activity_level", "temporal_dynamics", "topology")),
    ("all_clean_groups", MODEL_GROUPS),
)

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
MATERIALIZATION = BASE / "experiments" / "20260902_030441_849111_strict30_protected_feature_materialization"
FEATURE_TABLE = MATERIALIZATION / "strict30_protected_snapshot_features.csv"
MATERIALIZATION_RECORD = MATERIALIZATION / "run_record.json"
DATASET = STRICT30_ROOT / "pheme_graphs_roberta_30min_replyv4_preventableimpact_semantic.pt"
OUT_ROOT = BASE / "experiments"


def _write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


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


def _ranked_high_overlap(evaluated: pd.DataFrame) -> float:
    overlaps = []
    for _, event in evaluated.groupby("event_id", sort=True):
        k = max(1, math.ceil(len(event) * HIGH_FRACTION))
        # Equal class probabilities are common in tree ensembles.  Resolve
        # the deployment top-K deterministically with an identifier-only rule;
        # never let input row order (or an outcome) decide an Oracle overlap.
        ranked = event.sort_values(["prob_high", "thread_id"], ascending=[False, True], kind="stable")
        chosen = set(ranked.head(k).thread_id)
        oracle = set(event.loc[event.tier.eq("high"), "thread_id"])
        overlaps.append(len(chosen & oracle) / k)
    return float(np.mean(overlaps)) if overlaps else 0.0


def _fit_and_score(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], use_embedding: bool, embeddings: dict[str, np.ndarray], seed: int) -> tuple[dict[str, float], pd.DataFrame]:
    train = _event_tiers(train)
    evaluated = _event_tiers(test)
    x_train = train[columns].to_numpy(np.float32)
    x_test = evaluated[columns].to_numpy(np.float32)
    if use_embedding:
        train_embeddings = np.vstack([embeddings[thread_id] for thread_id in train.thread_id])
        test_embeddings = np.vstack([embeddings[thread_id] for thread_id in evaluated.thread_id])
        scaler = StandardScaler().fit(train_embeddings)
        pca = PCA(n_components=16, random_state=seed).fit(scaler.transform(train_embeddings))
        x_train = np.hstack([x_train, pca.transform(scaler.transform(train_embeddings))])
        x_test = np.hstack([x_test, pca.transform(scaler.transform(test_embeddings))])
    model = ExtraTreesClassifier(
        n_estimators=50 if RUN_MODE == "smoke" else 300,
        min_samples_leaf=3,
        max_features=0.7,
        class_weight="balanced",
        n_jobs=1,
        random_state=seed,
    )
    model.fit(x_train, train.tier)
    probabilities = model.predict_proba(x_test)
    evaluated["predicted_tier"] = model.classes_[probabilities.argmax(axis=1)]
    high_index = int(np.where(model.classes_ == "high")[0][0])
    evaluated["prob_high"] = probabilities[:, high_index]
    labels = ["high", "middle", "low"]
    return {
        "tier_macro_f1": float(f1_score(evaluated.tier, evaluated.predicted_tier, labels=labels, average="macro", zero_division=0)),
        "high_top_fraction_overlap": _ranked_high_overlap(evaluated),
    }, evaluated


def _quiet_partitions(frame: pd.DataFrame, recency: dict[str, float], train_events: list[str], test_event: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    raw_train = frame.loc[frame.event_id.isin(train_events)].copy()
    raw_test = frame.loc[frame.event_id.eq(test_event)].copy()
    threshold = float(np.quantile(raw_train.thread_id.map(recency), QUIET_QUANTILE))
    train = raw_train.loc[raw_train.thread_id.map(recency) >= threshold].copy()
    test = raw_test.loc[raw_test.thread_id.map(recency) >= threshold].copy()
    if train.empty or test.empty:
        raise ValueError(f"{test_event}: quiet partition is empty")
    return train, test, threshold


def _configs() -> list[tuple[str, tuple[str, ...]]]:
    if RUN_MODE == "smoke":
        return list(SMOKE_CONFIGS)
    output = []
    for length in range(1, len(MODEL_GROUPS) + 1):
        for groups in combinations(MODEL_GROUPS, length):
            output.append(("+".join(groups), groups))
    return output


def _inner_select(frame: pd.DataFrame, recency: dict[str, float], embeddings: dict[str, np.ndarray], outer_event: str, configs: list[tuple[str, tuple[str, ...]]], columns_by_group: dict[str, list[str]]) -> tuple[tuple[str, tuple[str, ...]], pd.DataFrame]:
    inner_events = sorted(event for event in frame.event_id.unique() if event != outer_event)
    if RUN_MODE == "smoke":
        inner_events = inner_events[:SMOKE_INNER_EVENTS]
    rows = []
    for config_index, (name, groups) in enumerate(configs, 1):
        columns = [column for group in groups if group != "source_reply_embedding" for column in columns_by_group[group]]
        use_embedding = "source_reply_embedding" in groups
        fold_metrics = []
        for fold_index, inner_event in enumerate(inner_events, 1):
            train_events = [event for event in inner_events if event != inner_event]
            train, test, threshold = _quiet_partitions(frame, recency, train_events, inner_event)
            metrics, _ = _fit_and_score(train, test, columns, use_embedding, embeddings, SEED + config_index * 100 + fold_index)
            fold_metrics.append(metrics)
        rows.append({
            "config": name, "groups": "+".join(groups), "feature_count": len(columns) + (16 if use_embedding else 0),
            "inner_folds": len(fold_metrics),
            "inner_high_top_fraction_overlap": float(np.mean([item["high_top_fraction_overlap"] for item in fold_metrics])),
            "inner_tier_macro_f1": float(np.mean([item["tier_macro_f1"] for item in fold_metrics])),
        })
        print(f"  inner config {config_index}/{len(configs)}: {name}", flush=True)
    summary = pd.DataFrame(rows).sort_values(
        ["inner_high_top_fraction_overlap", "inner_tier_macro_f1", "feature_count", "config"],
        ascending=[False, False, True, True], kind="stable",
    ).reset_index(drop=True)
    selected = summary.iloc[0]
    selected_groups = tuple(selected.groups.split("+"))
    return (str(selected.config), selected_groups), summary


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_strict30_clean_quiet_tier_{RUN_MODE}")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "mode": RUN_MODE,
        "inputs": {"clean_feature_table": str(FEATURE_TABLE.resolve()), "materialization_record": str(MATERIALIZATION_RECORD.resolve()), "protected_graphs": str(DATASET.resolve())},
        "split": "eligible-event LOEO; inner LOEO selects feature group combinations; outer held-out event is evaluation only",
        "cutoff_seconds": 1800, "quiet_gate": "outer/inner-train 60th percentile of protected log1p seconds since last activity",
        "tiers": "within each quiet event: high top 20%, middle next 30%, low remaining 50% by preventable impact",
        "selection_metric": "inner mean high_top_fraction_overlap; tier_macro_f1 then smaller feature count break ties",
        "seed": SEED,
    }
    _write_record(output, record)
    try:
        print(f"Starting strict-30 clean quiet tier {RUN_MODE}.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/5] Validating materialization provenance and loading features...", flush=True)
        source_record = json.loads(MATERIALIZATION_RECORD.read_text(encoding="utf-8"))
        if source_record.get("status") != "complete":
            raise ValueError("clean feature materialization is not complete")
        frame = pd.read_csv(FEATURE_TABLE, dtype={"thread_id": str, "event_id": str})
        if len(frame) != 2402 or frame.thread_id.duplicated().any():
            raise ValueError("clean feature table does not contain exactly 2402 unique threads")
        columns_by_group = {group: [column for column in frame if feature_group(column) == group] for group in FEATURE_GROUPS}
        if any(not columns for columns in columns_by_group.values()):
            raise ValueError("at least one declared feature group is missing")
        model_columns = [column for columns in columns_by_group.values() for column in columns]
        if not np.isfinite(frame[model_columns].to_numpy(float)).all():
            raise ValueError("clean model feature table contains non-finite values")
        event_sizes = frame.groupby("event_id").size(); eligible = sorted(event_sizes[event_sizes >= 100].index.tolist())
        if len(eligible) != 7:
            raise ValueError(f"expected seven eligible events, found {len(eligible)}")
        frame = frame.loc[frame.event_id.isin(eligible)].copy()
        print("[2/5] Auditing protected recency values and snapshot identities...", flush=True)
        graphs = torch.load(DATASET, weights_only=False); recency = {}; embeddings = {}
        for index, graph in enumerate(graphs, 1):
            values = graph.temporal_features.detach().cpu().numpy().reshape(-1)
            if len(values) != 11 or not np.isfinite(values).all():
                raise ValueError(f"{graph.thread_id}: invalid strict-30 temporal features")
            recency[str(graph.thread_id)] = float(values[5])
            source_indices = torch.nonzero(graph.is_source_mask, as_tuple=False).flatten()
            if source_indices.numel() != 1 or graph.x.ndim != 2 or graph.x.shape[1] <= 13:
                raise ValueError(f"{graph.thread_id}: invalid strict-30 source/reply embedding input")
            source_index = int(source_indices.item())
            vectors = graph.x[:, 13:].detach().cpu().numpy().astype(np.float32)
            replies = np.delete(vectors, source_index, axis=0)
            reply_centroid = replies.mean(axis=0) if len(replies) else np.zeros_like(vectors[source_index])
            embeddings[str(graph.thread_id)] = np.concatenate([vectors[source_index], reply_centroid])
            if index % 500 == 0 or index == len(graphs): print(f"  audited recency {index}/{len(graphs)}", flush=True)
        if set(frame.thread_id) - set(recency): raise ValueError("feature table thread missing protected graph")
        configs = _configs()
        outer_events = [SMOKE_OUTER_EVENT] if RUN_MODE == "smoke" else eligible
        all_metrics, all_predictions = [], []
        print("[3/5] Running inner LOEO group selection and held-out tier scoring...", flush=True)
        for outer_index, outer_event in enumerate(outer_events, 1):
            print(f"Outer event {outer_index}/{len(outer_events)}: {outer_event}", flush=True)
            selected, inner_summary = _inner_select(frame, recency, embeddings, outer_event, configs, columns_by_group)
            selected_name, selected_groups = selected; columns = [column for group in selected_groups if group != "source_reply_embedding" for column in columns_by_group[group]]
            use_embedding = "source_reply_embedding" in selected_groups
            train_events = [event for event in eligible if event != outer_event]
            train, test, threshold = _quiet_partitions(frame, recency, train_events, outer_event)
            metrics, predictions = _fit_and_score(train, test, columns, use_embedding, embeddings, SEED + outer_index)
            metrics.update({"outer_event": outer_event, "selected_config": selected_name, "selected_groups": "+".join(selected_groups), "selected_feature_count": len(columns) + (16 if use_embedding else 0), "quiet_threshold": threshold, "train_quiet_threads": len(train), "test_quiet_threads": len(test)})
            inner_summary.to_csv(output / f"inner_selection_{outer_event}.csv", index=False)
            predictions["outer_event"] = outer_event; all_predictions.append(predictions); all_metrics.append(metrics)
        print("[4/5] Writing predictions and fold metrics...", flush=True)
        metrics_frame = pd.DataFrame(all_metrics); predictions_frame = pd.concat(all_predictions, ignore_index=True)
        metrics_frame.to_csv(output / "outer_fold_metrics.csv", index=False); predictions_frame.to_csv(output / "outer_tier_predictions.csv", index=False)
        print("[5/5] Writing completion record...", flush=True)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "eligible_events": eligible, "config_count": len(configs), "outer_events_run": outer_events, "metrics": all_metrics, "output_files": ["outer_fold_metrics.csv", "outer_tier_predictions.csv"] + [f"inner_selection_{event}.csv" for event in outer_events]})
        _write_record(output, record)
        print(f"SUCCESS: strict-30 clean quiet tier {RUN_MODE} saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
        _write_record(output, record)
        print(f"FAILURE: output preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
