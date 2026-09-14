from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from effective_models.pheme_hawkes_multicheckpoint_rf import config
from effective_models.pheme_hawkes_multicheckpoint_rf.features.hawkes_features import add_fixed_decay_features, load_inputs
from effective_models.pheme_hawkes_multicheckpoint_rf.evaluation.sequential_policy import metrics


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _fit_score(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], params: dict) -> pd.DataFrame:
    model = RandomForestRegressor(**params)
    model.fit(train[columns].to_numpy(dtype=float), train.dynamic_preventable_y.to_numpy(dtype=float))
    prediction = model.predict(test[columns].to_numpy(dtype=float))
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite prediction")
    scored = test[["thread_id", "event_id", "checkpoint_sec", *(
        "dynamic_preventable_impact", "dynamic_preventable_y", "future_growth"
    )]].copy()
    scored["prediction"] = prediction
    return scored


def run(mode: str) -> Path:
    if mode not in {"smoke", "full"}:
        raise ValueError(mode)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = config.EXPERIMENTS / f"{stamp}_hawkes_fixed_decay_{mode}"
    output.mkdir(parents=True, exist_ok=False)
    params = dict(config.MODEL_PARAMS)
    if mode == "smoke":
        params.update(n_estimators=5, n_jobs=1)
    record = {
        "status": "running", "started_at": _now(), "mode": mode,
        "model": "Hawkes-inspired fixed-decay cumulative RF paired with cumulative baseline",
        "configuration": {"model_params": params, "decays_seconds": config.DECAYS_SECONDS,
                          "checkpoints_seconds": config.CHECKPOINTS_SECONDS,
                          "balanced_quotas": config.BALANCED_QUOTAS},
        "input_artifacts": {"features": str(config.FEATURE_TABLE.resolve()),
                            "schema": str(config.FEATURE_SCHEMA.resolve()),
                            "replies": str(config.REPLY_TABLE.resolve())},
        "split": "leave one complete event out across all checkpoints", "cutoff_seconds": config.CHECKPOINTS_SECONDS,
        "seed": config.SEED,
    }
    _write(output / "run_record.json", record)
    try:
        print(f"Starting Hawkes-inspired {mode} run.\nOutput directory: {output.resolve()}", flush=True)
        print("[1/6] Loading and validating frozen cumulative rows...", flush=True)
        base, base_columns = load_inputs()
        print("[2/6] Computing cutoff-safe fixed-decay reply features...", flush=True)
        frame, hawkes_columns = add_fixed_decay_features(base)
        augmented_columns = base_columns + hawkes_columns
        events = sorted(frame.event_id.unique())
        eligible = [event for event in events if frame.loc[frame.event_id.eq(event), "thread_id"].nunique() >= 100]
        folds = eligible[-1:] if mode == "smoke" else eligible
        print(f"[3/6] Running {len(folds)} paired event fold(s)...", flush=True)
        score_parts, event_rows, selected_parts = [], [], []
        for index, event in enumerate(folds, 1):
            train, test = frame.loc[frame.event_id.ne(event)], frame.loc[frame.event_id.eq(event)]
            if set(train.event_id) & set(test.event_id):
                raise ValueError("Event overlap")
            print(f"  fold {index}/{len(folds)}: held_out={event}", flush=True)
            for name, columns in (("cumulative_baseline", base_columns),
                                  ("cumulative_hawkes_fixed_decay", augmented_columns)):
                scored = _fit_score(train, test, columns, params)
                scored["model_name"] = name
                score_parts.append(scored)
                result, chosen = metrics(scored)
                event_rows.append({"outer_event": event, "model_name": name, **result})
                chosen["outer_event"], chosen["model_name"] = event, name
                selected_parts.append(chosen)
                print(f"    {name}: horizon_reduction={result['horizon_reduction']:.4f}", flush=True)
        print("[4/6] Aggregating paired sequential metrics...", flush=True)
        scores, per_event, selected = pd.concat(score_parts), pd.DataFrame(event_rows), pd.concat(selected_parts)
        macro = per_event.groupby("model_name", as_index=False).agg(
            eligible_events=("outer_event", "nunique"), mean_horizon_reduction=("horizon_reduction", "mean"),
            total_blocked_impact=("blocked_impact", "sum"), mean_action_minute=("mean_action_minute", "mean"))
        baseline = float(macro.loc[macro.model_name.eq("cumulative_baseline"), "mean_horizon_reduction"].iloc[0])
        macro["reduction_delta_vs_baseline"] = macro.mean_horizon_reduction - baseline
        print("[5/6] Writing OOF scores and selections...", flush=True)
        scores.to_csv(output / "oof_predictions.csv", index=False)
        per_event.to_csv(output / "per_event_metrics.csv", index=False)
        selected.to_csv(output / "selected_threads.csv", index=False)
        macro.to_csv(output / "macro_comparison.csv", index=False)
        result = {"macro": macro.to_dict(orient="records"), "smoke_is_not_efficacy_evidence": mode == "smoke"}
        _write(output / "result.json", result)
        print("[6/6] Writing complete run record...", flush=True)
        record.update(status="complete", completed_at=_now(), metrics_results=result,
                      output_file_inventory=["oof_predictions.csv", "per_event_metrics.csv", "selected_threads.csv",
                                             "macro_comparison.csv", "result.json", "run_record.json"],
                      failure_details=None,
                      research_safety={"dataset_and_labels_unchanged": True,
                                       "held_out_event_excluded_from_fit": True,
                                       "reply_offsets_limited_to_each_checkpoint": True,
                                       "outcome_columns_excluded_from_features": True})
        _write(output / "run_record.json", record)
        print(f"SUCCESS: Hawkes-inspired {mode} saved to: {output.resolve()}", flush=True)
        return output
    except BaseException as error:
        record.update(status="failed", completed_at=_now(), failure_details={
            "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
        _write(output / "run_record.json", record)
        print(f"FAILURE: Hawkes-inspired {mode} preserved at: {output.resolve()}", flush=True)
        raise

