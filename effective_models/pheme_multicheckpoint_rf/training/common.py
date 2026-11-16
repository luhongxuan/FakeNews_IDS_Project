"""Shared smoke/full workflow for the balanced cumulative dynamic RF."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from effective_models.pheme_multicheckpoint_rf import config
from effective_models.pheme_multicheckpoint_rf.evaluation.sequential_policy import event_metrics
from effective_models.pheme_multicheckpoint_rf.features.cumulative_features import load_frozen_materialization


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    parameters: dict[str, object],
) -> tuple[RandomForestRegressor, np.ndarray]:
    if set(train["event_id"]) & set(test["event_id"]):
        raise ValueError("Event overlap between train and test")
    x_train = train[columns].to_numpy(dtype=np.float32)
    x_test = test[columns].to_numpy(dtype=np.float32)
    y_train = train["dynamic_preventable_y"].to_numpy(dtype=np.float32)
    if not all(np.isfinite(value).all() for value in (x_train, x_test, y_train)):
        raise ValueError("Non-finite model array")
    model = RandomForestRegressor(**parameters).fit(x_train, y_train)
    prediction = model.predict(x_test)
    if prediction.shape != (len(test),) or not np.isfinite(prediction).all():
        raise ValueError("Invalid prediction")
    return model, prediction


def run_experiment(mode: str) -> Path:
    if mode not in {"smoke", "full"}:
        raise ValueError(f"Unsupported mode: {mode}")
    is_smoke = mode == "smoke"
    started = datetime.now(timezone.utc)
    suffix = "balanced_cumulative_smoke" if is_smoke else "balanced_cumulative_full"
    output = config.EXPERIMENTS_DIR / f"{started:%Y%m%d_%H%M%S_%f}_{suffix}"
    output.mkdir(parents=True, exist_ok=False)
    record_path = output / "run_record.json"
    parameters = config.SMOKE_PARAMETERS if is_smoke else config.FULL_PARAMETERS
    record: dict[str, object] = {
        "status": "running",
        "started_at": started.isoformat(),
        "mode": mode,
        "model": "Balanced Cumulative Multi-checkpoint RandomForestRegressor",
        "configuration": {
            "checkpoints_seconds": list(config.CHECKPOINTS),
            "balanced_quotas": list(config.BALANCED_QUOTAS),
            "total_budget": config.TOTAL_BUDGET,
            "parameters": parameters,
            "smoke_difference": "one held-out event and five trees; same frozen rows, features, fit, and sequential policy",
        },
        "input_artifacts": {
            "dataset": str(config.DATASET_PATH),
            "schema": str(config.SCHEMA_PATH),
        },
        "split": "leave one complete event out across all checkpoints",
        "cutoff_seconds": list(config.CHECKPOINTS),
        "seed": config.SEED,
        "metrics_results": None,
        "output_file_inventory": [],
        "failure_details": None,
    }
    _write_json(record_path, record)
    print(f"Starting balanced cumulative RF {mode} run.", flush=True)
    print(f"Output directory: {output.resolve()}", flush=True)
    try:
        print("[1/5] Loading and validating frozen multi-checkpoint rows...", flush=True)
        frame, columns = load_frozen_materialization()
        events = sorted(frame["event_id"].unique())
        thread_counts = frame[["thread_id", "event_id"]].drop_duplicates().groupby("event_id").size()
        eligible = sorted(thread_counts.loc[thread_counts.ge(config.ELIGIBLE_MIN_THREADS)].index)
        outer_events = events[-1:] if is_smoke else events

        print("[2/5] Validating event-disjoint cumulative feature contract...", flush=True)
        if any(token in column.lower() for column in columns for token in ("impact", "future", "target", "oracle")):
            raise ValueError("Outcome-like cumulative feature")

        print(f"[3/5] Running {len(outer_events)} held-out event fold(s)...", flush=True)
        predictions, metric_rows, selected_rows = [], [], []
        for index, event in enumerate(outer_events, start=1):
            print(f"  fold {index}/{len(outer_events)}: held_out={event}", flush=True)
            train = frame.loc[frame["event_id"].ne(event)]
            test = frame.loc[frame["event_id"].eq(event)].copy()
            _, prediction = fit_predict(train, test, columns, parameters)
            test["prediction"] = prediction
            predictions.append(test[[
                "thread_id", "event_id", "checkpoint_sec", "dynamic_preventable_impact",
                "dynamic_preventable_y", "future_growth", "prediction",
            ]])
            metrics, selected = event_metrics(test)
            metric_rows.append(metrics)
            selected_rows.append(selected[[
                "thread_id", "event_id", "action_checkpoint_sec", "dynamic_preventable_impact",
                "future_growth", "prediction", "selection_channel",
            ]])
            print(f"    horizon_reduction={metrics['horizon_reduction']:.4f}", flush=True)

        print("[4/5] Aggregating eligible-event sequential metrics...", flush=True)
        metrics_frame = pd.DataFrame(metric_rows)
        eligible_metrics = metrics_frame.loc[metrics_frame["event_id"].isin(eligible)]
        macro = {
            "eligible_events": int(eligible_metrics["event_id"].nunique()),
            "mean_horizon_reduction": (
                float(eligible_metrics["horizon_reduction"].mean()) if len(eligible_metrics) else None
            ),
            "total_blocked_impact": float(eligible_metrics["blocked_impact"].sum()),
            "total_positive_interventions": int(eligible_metrics["positive_interventions"].sum()),
            "mean_action_minute": (
                float(eligible_metrics["mean_action_minute"].mean()) if len(eligible_metrics) else None
            ),
        }
        pd.concat(predictions, ignore_index=True).to_csv(output / "oof_predictions.csv", index=False)
        metrics_frame.to_csv(output / "sequential_policy_per_event.csv", index=False)
        pd.concat(selected_rows, ignore_index=True).to_csv(output / "sequential_selected_threads.csv", index=False)
        _write_json(output / "result.json", {"macro": macro, "events": metric_rows})

        if not is_smoke:
            print("  fitting all-event deployment artifact after OOF evaluation...", flush=True)
            deployment = RandomForestRegressor(**parameters).fit(
                frame[columns].to_numpy(dtype=np.float32),
                frame["dynamic_preventable_y"].to_numpy(dtype=np.float32),
            )
            joblib.dump({"model": deployment, "feature_columns": columns}, output / "cumulative_all_events.joblib")

        print("[5/5] Writing complete run record...", flush=True)
        inventory = [
            "oof_predictions.csv", "sequential_policy_per_event.csv",
            "sequential_selected_threads.csv", "result.json", "run_record.json",
        ] + ([] if is_smoke else ["cumulative_all_events.joblib"])
        record.update(
            {
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "metrics_results": {"macro": macro, "events": metric_rows},
                "research_safety": {
                    "dataset_and_labels_unchanged": True,
                    "held_out_event_excluded_from_fit_at_all_checkpoints": True,
                    "feature_allowlist_excludes_outcomes": True,
                    "sequential_schedule_predeclared": True,
                    "smoke_is_not_a_research_result": is_smoke,
                },
                "output_file_inventory": inventory,
            }
        )
        _write_json(record_path, record)
        print(f"SUCCESS: balanced cumulative RF {mode} saved to: {output.resolve()}", flush=True)
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
        print(f"FAILURE: balanced cumulative RF {mode} preserved at: {output.resolve()}", flush=True)
        raise
