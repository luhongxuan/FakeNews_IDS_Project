"""Shared nested smoke/full workflow for the PHEME v5 text RF."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from effective_models.pheme_v5_text_rf import config
from effective_models.pheme_v5_text_rf.evaluation.metrics import selection_metrics
from effective_models.pheme_v5_text_rf.features.v5_features import (
    BASELINE_FEATURE_NAMES,
    load_graphs,
    load_raw_node_features,
    observable_node_keys,
    structural_matrix,
    text_matrix,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def validate_event_split(train: list[object], test: list[object]) -> None:
    overlap = {graph.event_id for graph in train} & {graph.event_id for graph in test}
    if overlap:
        raise ValueError(f"Event overlap between train and test: {sorted(overlap)}")


def fit_predict(
    train: list[object],
    test: list[object],
    raw_nodes: dict[tuple[str, str], tuple[str, float, float]],
    feature_set: str,
    parameters: dict[str, object],
) -> np.ndarray:
    if feature_set not in config.FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    validate_event_split(train, test)
    x_train = structural_matrix(train, raw_nodes)
    x_test = structural_matrix(test, raw_nodes)
    if feature_set == "text_augmented":
        raw_text_train = text_matrix(train)
        raw_text_test = text_matrix(test)
        text_scaler = StandardScaler().fit(raw_text_train)
        pca = PCA(n_components=config.PCA_COMPONENTS, random_state=config.SEED).fit(
            text_scaler.transform(raw_text_train)
        )
        x_train = np.hstack([x_train, pca.transform(text_scaler.transform(raw_text_train))])
        x_test = np.hstack([x_test, pca.transform(text_scaler.transform(raw_text_test))])
    scaler = StandardScaler().fit(x_train)
    y_train = np.asarray([float(graph.preventable_y.item()) for graph in train])
    model = RandomForestRegressor(**parameters).fit(scaler.transform(x_train), y_train)
    return model.predict(scaler.transform(x_test))


def run_experiment(mode: str) -> Path:
    if mode not in {"smoke", "full"}:
        raise ValueError(f"Unsupported mode: {mode}")
    is_smoke = mode == "smoke"
    started_at = datetime.now(timezone.utc)
    suffix = "v5_text_rf_smoke" if is_smoke else "v5_text_rf_nested_full"
    output = config.EXPERIMENTS_DIR / f"{started_at:%Y%m%d_%H%M%S_%f}_{suffix}"
    output.mkdir(parents=True, exist_ok=False)
    record_path = output / "run_record.json"
    parameters = config.SMOKE_PARAMETERS if is_smoke else config.FULL_PARAMETERS
    record: dict[str, object] = {
        "status": "running",
        "started_at": started_at.isoformat(),
        "mode": mode,
        "model": "PHEME v5 nested text RandomForestRegressor",
        "configuration": {
            "cutoff_seconds": config.CUTOFF_SECONDS,
            "top_k": config.TOP_K,
            "eligible_event_min_candidates": config.ELIGIBLE_EVENT_MIN_CANDIDATES,
            "pca_components": config.PCA_COMPONENTS,
            "parameters": parameters,
            "feature_sets": list(config.FEATURE_SETS),
            "smoke_difference": (
                "four-event subset, one outer fold, two inner folds, and five trees; "
                "same loaders, fold-local PCA/scalers, nested selection, RF fit, and metrics"
            ),
        },
        "input_artifacts": {
            "graph_dataset": str(config.DATASET_PATH),
            "raw_node_features": str(config.RAW_NODE_FEATURES_PATH),
        },
        "split": "nested leave-one-complete-event-out",
        "cutoff_seconds": config.CUTOFF_SECONDS,
        "seed": config.SEED,
        "metrics_results": None,
        "output_file_inventory": [],
        "failure_details": None,
    }
    _write_json(record_path, record)
    print(f"Starting PHEME v5 text RF {mode} run.", flush=True)
    print(f"Output directory: {output.resolve()}", flush=True)

    try:
        print("[1/5] Loading and validating protected graph artifact...", flush=True)
        all_graphs = load_graphs()
        all_events = sorted({graph.event_id for graph in all_graphs})
        if is_smoke:
            smoke_events = [
                event for event in ("charliehebdo", "ferguson", "germanwings-crash", "ottawashooting")
                if event in all_events
            ]
            graphs = [graph for graph in all_graphs if graph.event_id in smoke_events]
            outer_events = smoke_events[:1]
        else:
            graphs = all_graphs
            outer_events = all_events

        print("[2/5] Loading raw attributes for observable snapshot nodes...", flush=True)
        raw_nodes = load_raw_node_features(observable_node_keys(graphs))
        event_counts = {event: sum(graph.event_id == event for graph in graphs) for event in all_events}
        eligible_events = [
            event for event in all_events
            if event_counts.get(event, 0) >= config.ELIGIBLE_EVENT_MIN_CANDIDATES
        ]

        print(f"[3/5] Running {len(outer_events)} nested outer fold(s)...", flush=True)
        results: dict[str, object] = {}
        selected: dict[str, str] = {}
        for outer_index, test_event in enumerate(outer_events, start=1):
            inner_events = [event for event in eligible_events if event != test_event]
            if is_smoke:
                inner_events = inner_events[:2]
            print(
                f"  outer {outer_index}/{len(outer_events)}: held_out={test_event}; "
                f"inner_folds={len(inner_events)}",
                flush=True,
            )
            validation: dict[str, list[float]] = {name: [] for name in config.FEATURE_SETS}
            for inner_index, validation_event in enumerate(inner_events, start=1):
                print(
                    f"    inner {inner_index}/{len(inner_events)}: validation={validation_event}",
                    flush=True,
                )
                train = [
                    graph for graph in graphs
                    if graph.event_id not in (test_event, validation_event)
                ]
                valid = [graph for graph in graphs if graph.event_id == validation_event]
                for feature_set in config.FEATURE_SETS:
                    prediction = fit_predict(train, valid, raw_nodes, feature_set, parameters)
                    validation[feature_set].append(
                        float(selection_metrics(valid, prediction)["crr_model"])
                    )
            if any(not values for values in validation.values()):
                raise ValueError(f"No eligible inner score for outer event {test_event}")
            means = {name: float(np.mean(values)) for name, values in validation.items()}
            chosen = max(means, key=means.get)
            train = [graph for graph in graphs if graph.event_id != test_event]
            test = [graph for graph in graphs if graph.event_id == test_event]
            prediction = fit_predict(train, test, raw_nodes, chosen, parameters)
            row = selection_metrics(test, prediction)
            row.update(
                {
                    "selected_feature_set": chosen,
                    "inner_validation_mean_crr": means,
                    "inner_validation_events": inner_events,
                }
            )
            results[test_event] = row
            selected[test_event] = chosen
            print(
                f"    selected={chosen}; CRR@{config.TOP_K}={row['crr_model']:.4f}; "
                f"ImpactCapture@{config.TOP_K}={row['preventable_recall_model']:.4f}",
                flush=True,
            )

        print("[4/5] Aggregating eligible-event metrics...", flush=True)
        included = [
            row for event, row in results.items()
            if event_counts[event] >= config.ELIGIBLE_EVENT_MIN_CANDIDATES
        ]
        report = {
            "config": {
                "split": "nested LOEO",
                "cutoff_seconds": config.CUTOFF_SECONDS,
                "candidate_pool": "rumour graphs in protected artifact; never a model feature",
                "params": parameters,
                "feature_sets": {
                    "baseline": BASELINE_FEATURE_NAMES,
                    "text_augmented": BASELINE_FEATURE_NAMES
                    + [f"text_pca_{index}" for index in range(config.PCA_COMPONENTS)],
                },
                "pca": "64 components fitted on each training fold only",
                "research_result": not is_smoke,
            },
            "event_summary": results,
            "selected_feature_sets": selected,
            "mean_eligible_events": {
                key: float(np.mean([float(row[key]) for row in included])) if included else None
                for key in (
                    "crr_model", "crr_size", "crr_random", "preventable_recall_model",
                    "preventable_recall_size", "preventable_recall_random",
                )
            },
        }
        _write_json(output / "result.json", report)

        print("[5/5] Writing complete run record...", flush=True)
        record.update(
            {
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "metrics_results": report,
                "research_safety": {
                    "nested_event_separation_enforced": True,
                    "pca_and_scalers_fit_on_training_rows_only": True,
                    "features_reconstructed_from_observable_snapshot_nodes": True,
                    "future_target_used_as_feature": False,
                    "smoke_is_not_a_research_result": is_smoke,
                },
                "output_file_inventory": ["result.json", "run_record.json"],
            }
        )
        _write_json(record_path, record)
        print(f"SUCCESS: PHEME v5 text RF {mode} saved to: {output.resolve()}", flush=True)
        return output
    except Exception:
        record.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "failure_details": traceback.format_exc(),
                "output_file_inventory": [path.name for path in output.iterdir() if path.is_file()],
            }
        )
        _write_json(record_path, record)
        print(f"FAILURE: PHEME v5 text RF {mode} preserved at: {output.resolve()}", flush=True)
        raise
