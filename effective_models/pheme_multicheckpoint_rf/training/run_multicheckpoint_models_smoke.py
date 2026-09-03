"""Event-disjoint fit/predict smoke for both multi-checkpoint RF families."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import pandas as pd

import multicheckpoint_model_common as common


HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parent / "experiments"
SMOKE_PARAMS = {**common.MODEL_PARAMS, "n_estimators": 30, "max_depth": 5}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_multicheckpoint_models_smoke")
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "started_at": now(), "mode": "smoke", "seed": 42}
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        print("Starting multi-checkpoint cumulative/window-delta model smoke.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/5] Loading latest complete bounded materialization...", flush=True)
        source = common.latest_complete_materialization("smoke")
        frame, schema = common.load_materialization(source)
        events = sorted(frame.event_id.unique())
        test_event = events[-1]
        train = frame.loc[frame.event_id.ne(test_event)].copy()
        test = frame.loc[frame.event_id.eq(test_event)].copy()
        record.update({
            "input": str(source.resolve()), "split": {"train_events": events[:-1], "test_event": test_event},
            "checkpoints_seconds": sorted(frame.checkpoint_sec.unique().astype(int).tolist()),
            "model_params": SMOKE_PARAMS,
            "research_safety": "All rows from the held-out event are excluded from fitting; outcome columns are not model inputs.",
        })
        print("[2/5] Validating event-disjoint rows and feature allowlists...", flush=True)
        if set(train.thread_id) & set(test.thread_id):
            raise ValueError("thread overlap in smoke split")
        predictions, metrics = [], []
        print("[3/5] Fitting cumulative RF through the normal pipeline...", flush=True)
        for index, family in enumerate(common.FAMILIES, 1):
            if index == 2:
                print("[4/5] Fitting window/delta RF through the normal pipeline...", flush=True)
            columns = common.family_columns(schema, family)
            _, prediction = common.fit_predict(train, test, columns, SMOKE_PARAMS)
            scored = test[[
                "thread_id", "event_id", "checkpoint_sec", "dynamic_preventable_impact",
                "dynamic_preventable_y", "future_growth",
            ]].copy()
            scored["family"] = family
            scored["prediction"] = prediction
            predictions.append(scored)
            metrics.extend(common.metric_rows(scored, family))
        print("[5/5] Writing predictions, metrics, and complete smoke record...", flush=True)
        pd.concat(predictions, ignore_index=True).to_csv(output / "smoke_predictions.csv", index=False)
        metric_frame = pd.DataFrame(metrics)
        metric_frame.to_csv(output / "smoke_metrics.csv", index=False)
        record.update({
            "status": "complete", "completed_at": now(), "train_rows": len(train),
            "test_rows": len(test), "metrics_rows": len(metric_frame),
            "output_files": ["smoke_predictions.csv", "smoke_metrics.csv", "run_record.json"],
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"SUCCESS: multi-checkpoint model smoke saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({
            "status": "failed", "failed_at": now(), "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(), "output_files": sorted(path.name for path in output.iterdir()),
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"FAILURE: multi-checkpoint model smoke preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
