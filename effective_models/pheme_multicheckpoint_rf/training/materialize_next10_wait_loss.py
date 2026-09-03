"""Derive next-10-minute burst and wait-loss supervision from saved outcomes."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parent / "experiments"
SOURCE = (
    OUT_ROOT / "20260903_153526_240803_multicheckpoint_materialization_full"
)
SOURCE_DATA = SOURCE / "pheme_multicheckpoint_features.csv"
SOURCE_SCHEMA = SOURCE / "feature_schema.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    output = OUT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + "_next10_wait_loss_materialization"
    )
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "started_at": now(),
        "mode": "derived target materialization; no fitting",
        "input": str(SOURCE_DATA.resolve()),
        "split": "no split; outcomes only",
        "input_cutoffs_seconds": [600, 1200, 1800, 2400, 3000, 3600],
        "prediction_cutoffs_seconds": [600, 1200, 1800, 2400, 3000],
        "research_safety": (
            "next10 outcomes are supervision/evaluation only. Model feature columns are copied "
            "unchanged from the protected cutoff-specific cumulative allowlist."
        ),
    }
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        print("Starting next-10-minute wait-loss target materialization.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/5] Loading completed multi-checkpoint features and schema...", flush=True)
        frame = pd.read_csv(SOURCE_DATA, dtype={"thread_id": str, "event_id": str})
        schema = json.loads(SOURCE_SCHEMA.read_text(encoding="utf-8"))
        frame = frame.sort_values(["thread_id", "checkpoint_sec"], kind="stable")
        print("[2/5] Verifying complete deterministic checkpoint sequences...", flush=True)
        expected = (600, 1200, 1800, 2400, 3000, 3600)
        sequences = frame.groupby("thread_id").checkpoint_sec.apply(tuple)
        if not sequences.map(lambda value: value == expected).all() or frame.thread_id.nunique() != 2402:
            raise ValueError("incomplete or unexpected checkpoint sequence")
        print("[3/5] Deriving future-only next-window outcomes...", flush=True)
        grouped = frame.groupby("thread_id", sort=False)
        next_future = grouped.future_growth.shift(-1)
        next_impact = grouped.dynamic_preventable_impact.shift(-1)
        frame["next10_new_replies"] = frame.future_growth - next_future
        frame["next10_wait_loss"] = np.maximum(
            frame.dynamic_preventable_impact - next_impact, 0.0
        )
        frame = frame.loc[frame.checkpoint_sec.lt(3600)].copy()
        frame["next10_new_replies_y"] = np.log1p(frame.next10_new_replies)
        frame["next10_wait_loss_y"] = np.log1p(frame.next10_wait_loss)
        print("[4/5] Auditing target bounds and feature exclusion...", flush=True)
        targets = [
            "next10_new_replies", "next10_wait_loss",
            "next10_new_replies_y", "next10_wait_loss_y",
        ]
        if frame[targets].isna().any().any() or (frame[targets] < 0).any().any():
            raise ValueError("invalid next-window outcome")
        if (frame.next10_new_replies > frame.future_growth).any():
            raise ValueError("next-window replies exceed remaining future")
        feature_columns = list(schema["cumulative_feature_columns"])
        if set(targets) & set(feature_columns):
            raise ValueError("future target included in model feature allowlist")
        print("[5/5] Writing derived dataset, schema, statistics, and record...", flush=True)
        frame.to_csv(output / "pheme_next10_wait_loss_features.csv", index=False)
        derived_schema = {
            "metadata_columns": ["thread_id", "event_id", "checkpoint_sec"],
            "current_outcome_columns_not_features": [
                "dynamic_preventable_impact", "dynamic_preventable_y", "future_growth"
            ],
            "next10_outcome_columns_not_features": targets,
            "cumulative_feature_columns": feature_columns,
        }
        (output / "feature_schema.json").write_text(
            json.dumps(derived_schema, indent=2), encoding="utf-8"
        )
        stats = frame.groupby("checkpoint_sec", as_index=False).agg(
            rows=("thread_id", "size"),
            positive_wait_loss=("next10_wait_loss", lambda x: int(x.gt(0).sum())),
            mean_wait_loss=("next10_wait_loss", "mean"),
            max_wait_loss=("next10_wait_loss", "max"),
            mean_new_replies=("next10_new_replies", "mean"),
            max_new_replies=("next10_new_replies", "max"),
        )
        stats.to_csv(output / "target_statistics.csv", index=False)
        record.update({
            "status": "complete", "completed_at": now(), "rows": len(frame),
            "threads": frame.thread_id.nunique(), "events": frame.event_id.nunique(),
            "feature_count": len(feature_columns),
            "positive_wait_loss_rows": int(frame.next10_wait_loss.gt(0).sum()),
            "output_files": [
                "pheme_next10_wait_loss_features.csv", "feature_schema.json",
                "target_statistics.csv", "run_record.json",
            ],
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"SUCCESS: next10 wait-loss materialization saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({
            "status": "failed", "failed_at": now(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "output_files": sorted(path.name for path in output.iterdir()),
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"FAILURE: next10 wait-loss materialization preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
