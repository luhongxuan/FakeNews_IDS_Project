"""Shared nested smoke/full execution for the Graph7 discourse/context RF."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from effective_models.pheme_graph7_discourse_rf import config
from effective_models.pheme_graph7_discourse_rf.features.discourse_features import (
    feature_variants, load_schema_locked_data,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def evaluate_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    parameters: dict[str, object],
) -> dict[str, float | int]:
    if set(train["event_id"]) & set(test["event_id"]):
        raise ValueError("Event leakage between fit and evaluation rows")
    model = RandomForestRegressor(**parameters).fit(train[features], train["preventable_y"])
    pool = test.loc[test["category"].eq("rumours")].copy()
    if pool.empty:
        return {"n_candidate_rumours": 0, "crr_at_10": 0.0}
    pool["score"] = model.predict(pool[features])
    total = float(pool["preventable_impact"].sum())
    selected = float(pool.nlargest(config.TOP_K, "score")["preventable_impact"].sum())
    return {
        "n_candidate_rumours": int(len(pool)),
        "crr_at_10": selected / total if total else 0.0,
    }


def choose_variant(
    frame: pd.DataFrame,
    variants: dict[str, list[str]],
    outer_event: str,
    parameters: dict[str, object],
    inner_events: list[str],
) -> tuple[str, dict[str, float]]:
    scores: dict[str, list[float]] = {name: [] for name in variants}
    for index, validation_event in enumerate(inner_events, start=1):
        print(
            f"    inner {index}/{len(inner_events)}: validation={validation_event}",
            flush=True,
        )
        train = frame.loc[~frame["event_id"].isin([outer_event, validation_event])]
        validation = frame.loc[frame["event_id"].eq(validation_event)]
        for name, features in variants.items():
            result = evaluate_split(train, validation, features, parameters)
            if result["n_candidate_rumours"] >= config.ELIGIBLE_EVENT_MIN_CANDIDATES:
                scores[name].append(float(result["crr_at_10"]))
    if any(not values for values in scores.values()):
        raise ValueError(f"No eligible inner score for outer event {outer_event}")
    means = {name: float(np.mean(values)) for name, values in scores.items()}
    chosen = max(variants, key=lambda name: (means[name], name == "role_base"))
    return chosen, means


def run_experiment(mode: str) -> Path:
    if mode not in {"smoke", "full"}:
        raise ValueError(f"Unsupported mode: {mode}")
    is_smoke = mode == "smoke"
    started_at = datetime.now(timezone.utc)
    suffix = "nested_discourse_context_smoke" if is_smoke else "nested_discourse_context_full"
    output = config.EXPERIMENTS_DIR / f"{started_at:%Y%m%d_%H%M%S_%f}_{suffix}"
    output.mkdir(parents=True, exist_ok=False)
    record_path = output / "run_record.json"
    parameters = config.SMOKE_PARAMETERS if is_smoke else config.FULL_PARAMETERS
    run_record: dict[str, object] = {
        "status": "running",
        "started_at": started_at.isoformat(),
        "mode": mode,
        "model": "Graph7 nested discourse/context RandomForestRegressor",
        "configuration": {
            "cutoff_minutes": config.CUTOFF_MINUTES,
            "top_k": config.TOP_K,
            "eligible_event_min_candidates": config.ELIGIBLE_EVENT_MIN_CANDIDATES,
            "parameters": parameters,
            "variants": ["role_base", "plus_event_context", "plus_event_context_discourse"],
            "smoke_difference": (
                "one outer fold, first two eligible inner folds, and five trees; "
                "same artifact loader, feature bundles, nested selection, fit, and metric"
            ),
        },
        "input_artifacts": {
            "dataset_dir": str(config.DATASET_DIR),
            "records": str(config.DATASET_DIR / "thread_level_records.csv"),
            "schema": str(config.DATASET_DIR / "schema.json"),
            "manifest": str(config.DATASET_DIR / "manifest.json"),
        },
        "split": "nested leave-one-complete-event-out",
        "cutoff_minutes": config.CUTOFF_MINUTES,
        "seed": config.SEED,
        "metrics_results": None,
        "output_file_inventory": [],
        "failure_details": None,
    }
    _write_json(record_path, run_record)
    print(f"Starting Graph7 discourse/context RF {mode} run.", flush=True)
    print(f"Output directory: {output.resolve()}", flush=True)

    try:
        print("[1/4] Loading and validating schema-locked data...", flush=True)
        frame, schema = load_schema_locked_data()
        variants = feature_variants(schema)
        events = sorted(frame["event_id"].dropna().unique().tolist())
        outer_events = events[:1] if is_smoke else events

        print(f"[2/4] Running {len(outer_events)} nested outer fold(s)...", flush=True)
        outer_results: dict[str, object] = {}
        for outer_index, outer_event in enumerate(outer_events, start=1):
            eligible_inner = [
                event
                for event in events
                if event != outer_event
                and int(
                    frame.loc[
                        frame["event_id"].eq(event) & frame["category"].eq("rumours")
                    ].shape[0]
                )
                >= config.ELIGIBLE_EVENT_MIN_CANDIDATES
            ]
            inner_events = eligible_inner[:2] if is_smoke else [
                event for event in events if event != outer_event
            ]
            print(
                f"  outer {outer_index}/{len(outer_events)}: held_out={outer_event}; "
                f"inner_folds={len(inner_events)}",
                flush=True,
            )
            chosen, means = choose_variant(
                frame, variants, outer_event, parameters, inner_events
            )
            train = frame.loc[frame["event_id"].ne(outer_event)]
            test = frame.loc[frame["event_id"].eq(outer_event)]
            result = evaluate_split(train, test, variants[chosen], parameters)
            outer_results[outer_event] = {
                "chosen_variant": chosen,
                "inner_mean_crr": means,
                "outer": result,
            }
            print(
                f"    chose={chosen}; outer CRR@{config.TOP_K}={result['crr_at_10']:.4f}",
                flush=True,
            )

        print("[3/4] Aggregating eligible-event metrics...", flush=True)
        eligible_values = [
            float(value["outer"]["crr_at_10"])
            for value in outer_results.values()
            if value["outer"]["n_candidate_rumours"]
            >= config.ELIGIBLE_EVENT_MIN_CANDIDATES
        ]
        report = {
            "config": {
                "split": "nested LOEO",
                "cutoff_minutes": config.CUTOFF_MINUTES,
                "variants": list(variants),
                "candidate_pool": "oracle rumour category",
                "params": parameters,
                "research_result": not is_smoke,
                "safety": (
                    "Representation selection uses inner events only; frozen discourse/context "
                    "features are cutoff-safe and label-free."
                ),
            },
            "outer_events": outer_results,
            "nested_mean_crr_at_10_excluding_small_events": (
                float(np.mean(eligible_values)) if eligible_values else None
            ),
        }
        _write_json(output / "result.json", report)

        print("[4/4] Writing complete run record...", flush=True)
        run_record.update(
            {
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "metrics_results": report,
                "research_safety": {
                    "outer_and_inner_event_separation_enforced": True,
                    "schema_feature_boundary_enforced": True,
                    "future_target_used_as_feature": False,
                    "smoke_is_not_a_research_result": is_smoke,
                },
                "output_file_inventory": ["result.json", "run_record.json"],
            }
        )
        _write_json(record_path, run_record)
        print(f"SUCCESS: Graph7 discourse/context RF {mode} saved to: {output.resolve()}", flush=True)
        return output
    except Exception:
        run_record.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "failure_details": traceback.format_exc(),
                "output_file_inventory": [path.name for path in output.iterdir() if path.is_file()],
            }
        )
        _write_json(record_path, run_record)
        print(f"FAILURE: Graph7 discourse/context RF {mode} preserved at: {output.resolve()}", flush=True)
        raise
