"""Leakage-safe LOEO Random Forest baseline for preventable impact.

Unlike the earlier baseline, this script never reads the globally normalized
numeric columns in ``graph.x``.  It reconstructs the small tabular feature
set from the protected, unnormalized node-feature table and uses only nodes
that are already present in each 30-minute graph.  The scaler is fitted anew
on each LOEO training fold, then applied unchanged to its test event.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import torch

from run_temporal_tabular_baseline import DATASET, SEED, selection_metrics


BASE_DIR = Path(__file__).resolve().parent
RAW_NODE_FEATURES = BASE_DIR.parents[2] / "data" / "protected_research_assets" / "pheme_v5_strict30" / "pheme_node_features_roberta_replyv4.csv"
RAW_COLUMNS = ["thread_id", "tweet_id", "event_id", "depth", "followers_log"]
TEMPORAL_FEATURE_NAMES = [
    "log1p_count_0_10m", "log1p_count_10_20m", "log1p_count_20_30m",
    "log1p_count_recent_5m", "log1p_count_recent_10m",
    "log1p_seconds_since_last_activity", "log1p_median_interarrival_sec",
    "log1p_std_interarrival_sec", "observed_leaf_fraction",
    "mean_observed_children", "max_observed_children",
]
AUX_FEATURE_NAMES = [
    "log1p_observed_nodes", "late_activity_frac", "max_followers_log",
    "mean_depth", "max_depth",
]
FEATURE_NAMES = TEMPORAL_FEATURE_NAMES + AUX_FEATURE_NAMES
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 3,
    "max_features": 0.8,
    "n_jobs": 1,
    "random_state": SEED,
}


def load_raw_node_features() -> dict[tuple[str, str], tuple[str, float, float]]:
    """Load only original numeric attributes needed by this baseline."""
    frame = pd.read_csv(
        RAW_NODE_FEATURES,
        usecols=RAW_COLUMNS,
        dtype={"thread_id": str, "tweet_id": str, "event_id": str},
        low_memory=False,
    )
    if frame[["depth", "followers_log"]].isna().any().any():
        raise ValueError("Raw node features contain missing depth/followers values")
    if frame.duplicated(["thread_id", "tweet_id"]).any():
        raise ValueError("Raw node features are not unique by (thread_id, tweet_id)")
    return {
        (row.thread_id, row.tweet_id): (row.event_id, float(row.depth), float(row.followers_log))
        for row in frame.itertuples(index=False)
    }


def graph_features(graph, raw_nodes: dict[tuple[str, str], tuple[str, float, float]]) -> np.ndarray:
    """Create one feature vector using data observable by the cutoff only."""
    thread_id = str(graph.thread_id)
    rows = []
    for tweet_id in graph.node_ids:
        key = (thread_id, str(tweet_id))
        if key not in raw_nodes:
            raise KeyError(f"Missing raw feature row for observable node {key}")
        event_id, depth, followers_log = raw_nodes[key]
        if event_id != graph.event_id:
            raise ValueError(f"Event mismatch for {key}: {event_id} != {graph.event_id}")
        rows.append((depth, followers_log))
    node_values = np.asarray(rows, dtype=np.float32)
    temporal = graph.temporal_features.detach().cpu().numpy().reshape(-1)
    if temporal.size != len(TEMPORAL_FEATURE_NAMES):
        raise ValueError(f"Unexpected temporal feature length for {thread_id}")
    aux = np.asarray([
        np.log1p(graph.num_nodes),
        float(graph.late_activity_frac.item()),
        float(node_values[:, 1].max()),
        float(node_values[:, 0].mean()),
        float(node_values[:, 0].max()),
    ], dtype=np.float32)
    return np.concatenate([temporal, aux]).astype(np.float32)


def validate_fold(train, test, raw_nodes) -> None:
    train_events = {graph.event_id for graph in train}
    test_events = {graph.event_id for graph in test}
    if train_events & test_events:
        raise ValueError(f"Event overlap between train and test: {train_events & test_events}")
    # Materialize features to prove every graph references only its observed node IDs.
    for graph in (*train, *test):
        vector = graph_features(graph, raw_nodes)
        if not np.isfinite(vector).all():
            raise ValueError(f"Non-finite input feature for {graph.thread_id}")


def main() -> None:
    graphs = torch.load(DATASET, weights_only=False)
    raw_nodes = load_raw_node_features()
    events = sorted({graph.event_id for graph in graphs})
    output_dir = BASE_DIR / "experiments" / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_fold_safe_temporal_rf"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    results = {}
    for test_event in events:
        train = [graph for graph in graphs if graph.event_id != test_event]
        test = [graph for graph in graphs if graph.event_id == test_event]
        validate_fold(train, test, raw_nodes)
        x_train = np.vstack([graph_features(graph, raw_nodes) for graph in train])
        x_test = np.vstack([graph_features(graph, raw_nodes) for graph in test])
        y_train = np.asarray([graph.preventable_y.item() for graph in train])
        scaler = StandardScaler().fit(x_train)
        model = RandomForestRegressor(**MODEL_PARAMS).fit(scaler.transform(x_train), y_train)
        prediction = model.predict(scaler.transform(x_test))
        results[test_event] = selection_metrics(test, prediction)[test_event]
        print(
            f"{test_event:<22} crr={results[test_event]['crr_model']:.4f} "
            f"size={results[test_event]['crr_size']:.4f}"
        )
    included = [row for row in results.values() if row["n_test"] >= 100]
    report = {
        "config": {
            "dataset": DATASET.name,
            "raw_node_feature_source": str(RAW_NODE_FEATURES),
            "task": "log1p(preventable_future_impact)",
            "split": "LOEO",
            "cutoff_sec": 1800,
            "model": "RandomForestRegressor",
            "model_params": MODEL_PARAMS,
            "feature_names": FEATURE_NAMES,
            "normalization": "StandardScaler fitted on each fold's train events only; test events are transform-only",
            "research_safety": (
                "Every fold asserts train/test event disjointness. Features use only graph.node_ids "
                "from the 30-minute snapshot and raw per-node attributes; no graph.x numeric value is read."
            ),
        },
        "event_summary": results,
        "mean_excluding_small_events": {
            key: float(np.mean([row[key] for row in included]))
            for key in ("crr_model", "crr_size", "crr_random", "preventable_recall_model")
        },
    }
    (output_dir / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
