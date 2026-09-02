"""Nested LOEO test of adding frozen text embeddings to the impact ranker."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import torch


BASE_DIR = Path(__file__).resolve().parent
MODEL_ROOT = BASE_DIR.parent
PROJECT_ROOT = BASE_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from text_feature_common import PCA_COMPONENTS, text_matrix  # noqa: E402
from run_fold_safe_temporal_rf import (  # noqa: E402
    FEATURE_NAMES, MODEL_PARAMS, SEED, graph_features, load_raw_node_features, validate_fold,
)
from run_temporal_tabular_baseline import selection_metrics  # noqa: E402


DATASET = PROJECT_ROOT / "data" / "protected_research_assets" / "pheme_v5_strict30" / "pheme_graphs_roberta_30min_replyv4_preventableimpact_semantic.pt"
ELIGIBLE_MIN_THREADS = 100
FEATURE_SETS = ("baseline", "text_augmented")


def matrices(train, test, raw_nodes, feature_set: str) -> tuple[np.ndarray, np.ndarray]:
    x_train = np.vstack([graph_features(graph, raw_nodes) for graph in train])
    x_test = np.vstack([graph_features(graph, raw_nodes) for graph in test])
    if feature_set == "baseline":
        return x_train, x_test
    text_train, text_test = text_matrix(train), text_matrix(test)
    text_scaler = StandardScaler().fit(text_train)
    pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED).fit(text_scaler.transform(text_train))
    return (
        np.hstack([x_train, pca.transform(text_scaler.transform(text_train))]).astype(np.float32),
        np.hstack([x_test, pca.transform(text_scaler.transform(text_test))]).astype(np.float32),
    )


def fit_predict(train, test, raw_nodes, feature_set: str) -> np.ndarray:
    validate_fold(train, test, raw_nodes)
    x_train, x_test = matrices(train, test, raw_nodes, feature_set)
    y_train = np.asarray([graph.preventable_y.item() for graph in train])
    scaler = StandardScaler().fit(x_train)
    model = RandomForestRegressor(**MODEL_PARAMS).fit(scaler.transform(x_train), y_train)
    return model.predict(scaler.transform(x_test))


def main() -> None:
    graphs = torch.load(DATASET, weights_only=False)
    raw_nodes = load_raw_node_features()
    events = sorted({graph.event_id for graph in graphs})
    eligible = [event for event in events if sum(graph.event_id == event for graph in graphs) >= ELIGIBLE_MIN_THREADS]
    output_dir = MODEL_ROOT / "experiments" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_nested_text_impact_ranker_rf")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "dataset": DATASET.name, "task": "preventable_future_impact", "split": "nested LOEO",
        "cutoff_sec": 1800, "model": "RandomForestRegressor", "model_params": MODEL_PARAMS,
        "feature_sets": {
            "baseline": FEATURE_NAMES,
            "text_augmented": FEATURE_NAMES + [f"text_pca_{index}" for index in range(PCA_COMPONENTS)],
        },
        "text_input": "raw source RoBERTa embedding + observed early-reply embedding centroid",
        "pca_components": PCA_COMPONENTS, "inner_validation": "mean CRR@10 across eligible non-test events",
        "eligible_min_threads": ELIGIBLE_MIN_THREADS, "seed": SEED,
        "research_safety": "PCA, scalers, model, and feature-set selection fit/use no outer-test data; all text embeddings are from snapshot nodes only.",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    results, selected = {}, {}
    for test_event in events:
        inner_events = [event for event in eligible if event != test_event]
        validation = {feature_set: [] for feature_set in FEATURE_SETS}
        for validation_event in inner_events:
            train = [graph for graph in graphs if graph.event_id not in (test_event, validation_event)]
            valid = [graph for graph in graphs if graph.event_id == validation_event]
            for feature_set in FEATURE_SETS:
                prediction = fit_predict(train, valid, raw_nodes, feature_set)
                validation[feature_set].append(selection_metrics(valid, prediction)[validation_event]["crr_model"])
        means = {feature_set: float(np.mean(values)) for feature_set, values in validation.items()}
        chosen = max(means, key=means.get)
        train = [graph for graph in graphs if graph.event_id != test_event]
        test = [graph for graph in graphs if graph.event_id == test_event]
        row = selection_metrics(test, fit_predict(train, test, raw_nodes, chosen))[test_event]
        row.update({"selected_feature_set": chosen, "inner_validation_mean_crr": means, "inner_validation_events": inner_events})
        results[test_event], selected[test_event] = row, chosen
        print(f"{test_event:<22} selected={chosen:<15} val={means[chosen]:.4f} test_crr={row['crr_model']:.4f}", flush=True)
        (output_dir / "partial_result.json").write_text(json.dumps({"config": config, "event_summary": results, "selected_feature_sets": selected}, indent=2), encoding="utf-8")
    included = [row for event, row in results.items() if event in eligible]
    report = {
        "config": config, "event_summary": results, "selected_feature_sets": selected,
        "mean_eligible_events": {
            key: float(np.mean([row[key] for row in included]))
            for key in ("crr_model", "crr_size", "crr_random", "preventable_recall_model")
        },
    }
    (output_dir / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
