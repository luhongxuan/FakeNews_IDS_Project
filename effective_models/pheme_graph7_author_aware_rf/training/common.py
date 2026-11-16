"""Shared smoke/full execution for the Graph7 author-aware RF."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from effective_models.pheme_graph7_author_aware_rf import config
from effective_models.pheme_graph7_author_aware_rf.features.author_aware_features import load_schema_locked_data


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def evaluate_fold(
    model: RandomForestRegressor,
    test: pd.DataFrame,
    features: list[str],
    rng: np.random.Generator,
) -> dict[str, float | int]:
    pool = test.loc[test["category"].eq("rumours")].copy()
    if pool.empty:
        return {
            "n_candidate_rumours": 0,
            "crr_model": 0.0,
            "crr_size": 0.0,
            "crr_random": 0.0,
            "blocked_model": 0.0,
            "candidate_future_total": 0.0,
        }
    pool["prediction"] = model.predict(pool[features])
    total = float(pool["preventable_impact"].sum())
    model_value = float(pool.nlargest(config.TOP_K, "prediction")["preventable_impact"].sum())
    size_value = float(pool.nlargest(config.TOP_K, "observed_actions")["preventable_impact"].sum())
    sample_size = min(config.TOP_K, len(pool))
    random_values = [
        float(pool.iloc[rng.choice(len(pool), size=sample_size, replace=False)]["preventable_impact"].sum())
        for _ in range(200)
    ]
    return {
        "n_candidate_rumours": int(len(pool)),
        "crr_model": model_value / total if total else 0.0,
        "crr_size": size_value / total if total else 0.0,
        "crr_random": float(np.mean(random_values)) / total if total else 0.0,
        "blocked_model": model_value,
        "candidate_future_total": total,
    }


def run_experiment(mode: str) -> Path:
    if mode not in {"smoke", "full"}:
        raise ValueError(f"Unsupported mode: {mode}")
    is_smoke = mode == "smoke"
    started_at = datetime.now(timezone.utc)
    suffix = "author_aware_rf_smoke" if is_smoke else "author_aware_rf_partial_formal"
    output = config.EXPERIMENTS_DIR / f"{started_at:%Y%m%d_%H%M%S_%f}_{suffix}"
    output.mkdir(parents=True, exist_ok=False)
    run_record_path = output / "run_record.json"
    parameters = config.SMOKE_PARAMETERS if is_smoke else config.FULL_PARAMETERS
    run_record: dict[str, object] = {
        "status": "running",
        "started_at": started_at.isoformat(),
        "mode": mode,
        "model": "Graph7 author-aware RandomForestRegressor",
        "configuration": {
            "cutoff_minutes": config.CUTOFF_MINUTES,
            "top_k": config.TOP_K,
            "eligible_event_min_candidates": config.ELIGIBLE_EVENT_MIN_CANDIDATES,
            "parameters": parameters,
            "smoke_difference": "one outer fold and five trees; same loader, features, fit, and metric",
        },
        "input_artifacts": {
            "dataset_dir": str(config.DATASET_DIR),
            "records": str(config.DATASET_DIR / "thread_level_records.csv"),
            "schema": str(config.DATASET_DIR / "schema.json"),
        },
        "split": "leave one complete PHEME event out",
        "cutoff_minutes": config.CUTOFF_MINUTES,
        "seed": config.SEED,
        "metrics_results": None,
        "output_file_inventory": [],
        "failure_details": None,
    }
    _write_json(run_record_path, run_record)
    print(f"Starting Graph7 author-aware RF {mode} run.", flush=True)
    print(f"Output directory: {output.resolve()}", flush=True)

    try:
        print("[1/4] Loading and validating schema-locked data...", flush=True)
        frame, schema = load_schema_locked_data()
        features = list(schema["feature_columns"])
        events = sorted(frame["event_id"].dropna().unique().tolist())
        selected_events = events[:1] if is_smoke else events

        print(f"[2/4] Running {len(selected_events)} event-disjoint fold(s)...", flush=True)
        rows: dict[str, dict[str, float | int]] = {}
        rng = np.random.default_rng(config.SEED)
        for index, event in enumerate(selected_events, start=1):
            train = frame.loc[frame["event_id"].ne(event)]
            test = frame.loc[frame["event_id"].eq(event)]
            if set(train["event_id"]) & set(test["event_id"]):
                raise ValueError("Event leakage")
            print(f"  fold {index}/{len(selected_events)}: held_out={event}", flush=True)
            model = RandomForestRegressor(**parameters).fit(train[features], train["preventable_y"])
            rows[event] = evaluate_fold(model, test, features, rng)
            print(
                f"    CRR@{config.TOP_K}: model={rows[event]['crr_model']:.4f}, "
                f"size={rows[event]['crr_size']:.4f}, random={rows[event]['crr_random']:.4f}",
                flush=True,
            )

        print("[3/4] Aggregating eligible-event metrics...", flush=True)
        eligible = [
            value for value in rows.values()
            if value["n_candidate_rumours"] >= config.ELIGIBLE_EVENT_MIN_CANDIDATES
        ]
        macro = {
            key: float(np.mean([float(value[key]) for value in eligible])) if eligible else None
            for key in ("crr_model", "crr_size", "crr_random")
        }
        result = {
            "config": {
                "protocol": "partial_observation",
                "split": "LOEO by event",
                "candidate_pool": "oracle rumour category; never a feature",
                "cutoff_minutes": config.CUTOFF_MINUTES,
                "model": "RandomForestRegressor",
                "params": parameters,
                "features": features,
                "selection": f"top {config.TOP_K}",
                "research_result": not is_smoke,
            },
            "events": rows,
            "mean_excluding_small_events": macro,
        }
        _write_json(output / "result.json", result)

        print("[4/4] Writing complete run record...", flush=True)
        run_record.update(
            {
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "metrics_results": result,
                "research_safety": {
                    "event_disjoint_folds": True,
                    "schema_feature_boundary_enforced": True,
                    "future_target_used_as_feature": False,
                    "smoke_is_not_a_research_result": is_smoke,
                },
                "output_file_inventory": ["result.json", "run_record.json"],
            }
        )
        _write_json(run_record_path, run_record)
        print(f"SUCCESS: Graph7 author-aware RF {mode} saved to: {output.resolve()}", flush=True)
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
        _write_json(run_record_path, run_record)
        print(f"FAILURE: Graph7 author-aware RF {mode} preserved at: {output.resolve()}", flush=True)
        raise
