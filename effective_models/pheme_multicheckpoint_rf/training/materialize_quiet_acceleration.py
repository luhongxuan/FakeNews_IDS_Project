"""Derive next-window acceleration outcomes without changing early features."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parent / "experiments"
SOURCE = OUT_ROOT / "20260903_153526_240803_multicheckpoint_materialization_full"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    output = OUT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + "_quiet_acceleration_materialization"
    )
    output.mkdir(parents=True, exist_ok=False)
    source_data = SOURCE / "pheme_multicheckpoint_features.csv"
    record = {
        "status": "running", "started_at": now(),
        "mode": "derived acceleration outcomes; no fitting",
        "input": str(source_data.resolve()), "split": "no split; outcomes only",
        "prediction_checkpoints_seconds": [600, 1200, 1800, 2400, 3000],
        "target_definition": (
            "signed acceleration = replies in (T,T+600] minus observed replies in (T-600,T]; "
            "positive target = max(signed acceleration, 0)"
        ),
        "research_safety": (
            "Next-window counts and acceleration are supervision/evaluation only. Quiet status "
            "is not materialized globally; every model fold derives its gate from training events."
        ),
    }
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        print("Starting quiet next-10-minute acceleration materialization.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/5] Loading completed cutoff-specific feature table...", flush=True)
        frame = pd.read_csv(source_data, dtype={"thread_id": str, "event_id": str})
        schema = json.loads((SOURCE / "feature_schema.json").read_text(encoding="utf-8"))
        frame = frame.sort_values(["thread_id", "checkpoint_sec"], kind="stable")
        print("[2/5] Verifying six checkpoints for every protected candidate...", flush=True)
        expected = (600, 1200, 1800, 2400, 3000, 3600)
        sequences = frame.groupby("thread_id").checkpoint_sec.apply(tuple)
        if frame.thread_id.nunique() != 2402 or not sequences.map(lambda x: x == expected).all():
            raise ValueError("unexpected candidate or checkpoint sequence")
        print("[3/5] Deriving next-window growth and acceleration outcomes...", flush=True)
        next_future = frame.groupby("thread_id", sort=False).future_growth.shift(-1)
        next_count = frame.future_growth - next_future
        previous_count = frame["window_delta__win_current_count"]
        frame["previous10_reply_count"] = previous_count
        frame["next10_reply_count"] = next_count
        frame["next10_signed_acceleration"] = next_count - previous_count
        frame["next10_positive_acceleration"] = np.maximum(
            frame.next10_signed_acceleration, 0.0
        )
        frame["next10_positive_acceleration_y"] = np.log1p(
            frame.next10_positive_acceleration
        )
        frame = frame.loc[frame.checkpoint_sec.lt(3600)].copy()
        print("[4/5] Auditing outcome bounds and feature exclusion...", flush=True)
        outcomes = [
            "previous10_reply_count", "next10_reply_count",
            "next10_signed_acceleration", "next10_positive_acceleration",
            "next10_positive_acceleration_y",
        ]
        if frame[outcomes].isna().any().any() or (frame.next10_reply_count < 0).any():
            raise ValueError("invalid acceleration outcome")
        features = list(schema["window_delta_feature_columns"])
        forbidden = ("next10", "future", "acceleration", "impact", "target", "label")
        bad = [c for c in features if any(token in c.lower() for token in forbidden)]
        if bad:
            raise ValueError(f"future-like feature names: {bad}")
        print("[5/5] Writing derived data, schema, statistics, and run record...", flush=True)
        frame.to_csv(output / "pheme_quiet_acceleration_features.csv", index=False)
        derived_schema = {
            "metadata_columns": ["thread_id", "event_id", "checkpoint_sec"],
            "outcome_columns_not_features": [
                "dynamic_preventable_impact", "dynamic_preventable_y", "future_growth",
                *outcomes,
            ],
            "quiet_gate_feature": "cumulative__cum_time_since_last_activity",
            "quiet_gate_quantile": 0.60,
            "window_delta_feature_columns": features,
        }
        (output / "feature_schema.json").write_text(
            json.dumps(derived_schema, indent=2), encoding="utf-8"
        )
        stats = frame.groupby("checkpoint_sec", as_index=False).agg(
            rows=("thread_id", "size"),
            positive_acceleration_rows=(
                "next10_positive_acceleration", lambda x: int(x.gt(0).sum())
            ),
            mean_signed_acceleration=("next10_signed_acceleration", "mean"),
            mean_positive_acceleration=("next10_positive_acceleration", "mean"),
            max_positive_acceleration=("next10_positive_acceleration", "max"),
        )
        stats.to_csv(output / "target_statistics.csv", index=False)
        record.update({
            "status": "complete", "completed_at": now(), "rows": len(frame),
            "threads": frame.thread_id.nunique(), "events": frame.event_id.nunique(),
            "feature_count": len(features),
            "positive_acceleration_rows": int(frame.next10_positive_acceleration.gt(0).sum()),
            "output_files": [
                "pheme_quiet_acceleration_features.csv", "feature_schema.json",
                "target_statistics.csv", "run_record.json",
            ],
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"SUCCESS: quiet acceleration materialization saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({
            "status": "failed", "failed_at": now(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "output_files": sorted(p.name for p in output.iterdir()),
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"FAILURE: quiet acceleration materialization preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
