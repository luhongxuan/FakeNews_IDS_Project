"""LOEO tabular baseline for preventable future impact.

This baseline tests whether early temporal/structural summaries contain signal
before attributing a failure to the GNN architecture.  It never uses test data
for fitting or model selection.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import random

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch


BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "pheme_graphs_roberta_30min_replyv4_preventableimpact_temporal.pt"
BUDGET_THREADS = 10
SEED = 42
TEMPORAL_FEATURE_NAMES = [
    "log1p_count_0_10m", "log1p_count_10_20m", "log1p_count_20_30m",
    "log1p_count_recent_5m", "log1p_count_recent_10m",
    "log1p_seconds_since_last_activity", "log1p_median_interarrival_sec",
    "log1p_std_interarrival_sec", "observed_leaf_fraction",
    "mean_observed_children", "max_observed_children",
]
AUX_FEATURE_NAMES = ["log1p_observed_nodes", "late_activity_frac", "max_followers", "mean_depth", "max_depth"]
FEATURE_NAMES = TEMPORAL_FEATURE_NAMES + AUX_FEATURE_NAMES


def graph_features(graph) -> np.ndarray:
    temporal = graph.temporal_features.detach().cpu().numpy().reshape(-1)
    x = graph.x.detach().cpu().numpy()
    aux = np.asarray([
        np.log1p(graph.num_nodes),
        float(graph.late_activity_frac.item()),
        float(x[:, 8].max()),
        float(x[:, 0].mean()),
        float(x[:, 0].max()),
    ], dtype=np.float32)
    return np.concatenate([temporal, aux]).astype(np.float32)


def top_ids(scores: dict[str, float], budget: int) -> set[str]:
    return {thread_id for thread_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:budget]}


def selection_metrics(graphs, predictions: np.ndarray) -> dict:
    records = []
    for graph, prediction in zip(graphs, predictions):
        records.append({
            "thread_id": str(graph.thread_id),
            "event": graph.event_id,
            "prediction": float(prediction),
            "impact": int(graph.preventable_impact.item()),
            "future": int(round(float(torch.expm1(graph.y).item()))),
            "size": int(graph.num_nodes),
        })
    by_event = {}
    for record in records:
        by_event.setdefault(record["event"], []).append(record)
    result = {}
    for event, rows in by_event.items():
        by_id = {row["thread_id"]: row for row in rows}
        budget = min(BUDGET_THREADS, len(rows))
        selected = top_ids({row["thread_id"]: row["prediction"] for row in rows}, budget)
        size_selected = top_ids({row["thread_id"]: row["size"] for row in rows}, budget)
        random_selected = set(random.Random(SEED).sample(list(by_id), budget))
        total_future = sum(row["future"] for row in rows)
        total_impact = sum(row["impact"] for row in rows)
        def score(ids: set[str]) -> tuple[int, float, float]:
            blocked = sum(by_id[thread_id]["impact"] for thread_id in ids)
            return blocked, blocked / total_future if total_future else 0.0, blocked / total_impact if total_impact else 0.0
        model_blocked, model_crr, model_recall = score(selected)
        size_blocked, size_crr, size_recall = score(size_selected)
        random_blocked, random_crr, random_recall = score(random_selected)
        result[event] = {
            "n_test": len(rows), "blocked_model": model_blocked, "crr_model": model_crr,
            "preventable_recall_model": model_recall, "blocked_size": size_blocked,
            "crr_size": size_crr, "preventable_recall_size": size_recall,
            "blocked_random": random_blocked, "crr_random": random_crr,
            "preventable_recall_random": random_recall,
        }
    return result


def main() -> None:
    graphs = torch.load(DATASET, weights_only=False)
    events = sorted({graph.event_id for graph in graphs})
    output_dir = BASE_DIR / "experiments" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_preventable_temporal_tabular")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "dataset": DATASET.name, "task": "log1p(preventable_future_impact)",
        "split": "LOEO", "cutoff_sec": 1800, "budget_threads": BUDGET_THREADS,
        "seed": SEED, "model": "RandomForestRegressor",
        "model_params": {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 3, "max_features": 0.8, "n_jobs": 1},
        "feature_names": FEATURE_NAMES,
        "research_safety": "Fit feature scaling and model on each fold's train events only; final test is evaluation only.",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    results, importances = {}, []
    for event in events:
        train = [graph for graph in graphs if graph.event_id != event]
        test = [graph for graph in graphs if graph.event_id == event]
        x_train = np.vstack([graph_features(graph) for graph in train])
        x_test = np.vstack([graph_features(graph) for graph in test])
        y_train = np.asarray([graph.preventable_y.item() for graph in train])
        y_test = np.asarray([graph.preventable_y.item() for graph in test])
        scaler = StandardScaler().fit(x_train)
        model = RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=3, max_features=0.8,
            n_jobs=1, random_state=SEED,
        )
        model.fit(scaler.transform(x_train), y_train)
        prediction = model.predict(scaler.transform(x_test))
        event_metrics = selection_metrics(test, prediction)[event]
        event_metrics["mae"] = float(mean_absolute_error(y_test, prediction))
        event_metrics["mae_constant_baseline"] = float(mean_absolute_error(y_test, np.full_like(y_test, y_train.mean())))
        results[event] = event_metrics
        importances.append(model.feature_importances_)
        print(f"{event:<22} mae={event_metrics['mae']:.4f} crr={event_metrics['crr_model']:.4f} size={event_metrics['crr_size']:.4f}")
    included = [metrics for metrics in results.values() if metrics["n_test"] >= 50]
    report = {
        "config": config,
        "event_summary": results,
        "mean_excluding_small_events": {
            key: float(np.mean([metrics[key] for metrics in included]))
            for key in ("mae", "mae_constant_baseline", "crr_model", "crr_size", "crr_random",
                        "preventable_recall_model", "preventable_recall_size", "preventable_recall_random")
        },
        "mean_train_only_feature_importance": {
            name: float(value) for name, value in zip(FEATURE_NAMES, np.mean(importances, axis=0))
        },
    }
    (output_dir / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
