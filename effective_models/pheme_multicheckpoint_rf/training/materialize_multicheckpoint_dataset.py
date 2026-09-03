"""Materialize full 10--60 minute cumulative and window/delta dataset."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import torch

import multicheckpoint_common as common


HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parent / "experiments"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(mode: str = "full", max_threads_per_event: int | None = None) -> Path:
    suffix = "multicheckpoint_materialization_smoke" if mode == "smoke" else "multicheckpoint_materialization_full"
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + suffix)
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "started_at": now(), "mode": mode,
        "inputs": {
            "reply_provenance": str(common.REPLIES.resolve()),
            "candidate_allowlist": str(common.CANDIDATE_GRAPHS.resolve()),
        },
        "candidate_pool": "exact 2402 corrected-v5 rumour-only thread IDs",
        "split": "no split; feature materialization only",
        "checkpoints_seconds": list(common.CHECKPOINTS), "window_seconds": common.WINDOW_SECONDS,
        "target": "corrected all-traversable-roots dynamic preventable impact at each checkpoint",
        "research_safety": (
            "Each feature row uses only reply/node fields with offset<=its checkpoint. Window/delta "
            "uses (T-600,T] and the immediately preceding window. Future rows are outcome-only."
        ),
    }
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        print(f"Starting {mode} multi-checkpoint feature materialization.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/4] Loading protected reply provenance...", flush=True)
        raw = common.load_replies()
        graphs = torch.load(common.CANDIDATE_GRAPHS, weights_only=False)
        candidate_ids = {str(graph.thread_id) for graph in graphs}
        if len(candidate_ids) != 2402:
            raise ValueError(f"expected 2402 corrected-v5 candidates, found {len(candidate_ids)}")
        raw = raw.loc[raw.thread_id.isin(candidate_ids)].copy()
        if set(raw.thread_id) != candidate_ids:
            raise ValueError("reply provenance does not exactly cover corrected-v5 candidates")
        print(f"  candidate allowlist applied: {len(candidate_ids)} rumour threads", flush=True)
        print("[2/4] Building cumulative and window/delta rows at 10-minute checkpoints...", flush=True)
        frame = common.build_rows(raw, max_threads_per_event=max_threads_per_event)
        print("[3/4] Validating feature families and cutoff outcomes...", flush=True)
        cumulative = common.feature_columns(frame, "cumulative")
        window = common.feature_columns(frame, "window_delta")
        if (frame.dynamic_preventable_impact > frame.future_growth).any():
            raise ValueError("preventable impact exceeds future growth")
        if not frame.groupby("thread_id").checkpoint_sec.apply(
            lambda values: tuple(values) == common.CHECKPOINTS
        ).all():
            raise ValueError("incomplete checkpoint sequence")
        print("[4/4] Writing dataset, schema, and complete run record...", flush=True)
        frame.to_csv(output / "pheme_multicheckpoint_features.csv", index=False)
        schema = {
            "metadata_columns": ["thread_id", "event_id", "checkpoint_sec"],
            "outcome_columns_not_features": [
                "dynamic_preventable_impact", "dynamic_preventable_y", "future_growth"
            ],
            "cumulative_feature_columns": cumulative,
            "window_delta_feature_columns": window,
        }
        (output / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
        record.update({
            "status": "complete", "completed_at": now(), "threads": frame.thread_id.nunique(),
            "rows": len(frame), "events": frame.event_id.nunique(),
            "cumulative_features": len(cumulative), "window_delta_features": len(window),
            "output_files": ["pheme_multicheckpoint_features.csv", "feature_schema.json", "run_record.json"],
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"SUCCESS: multi-checkpoint dataset saved to: {output.resolve()}", flush=True)
        return output
    except BaseException as error:
        record.update({
            "status": "failed", "failed_at": now(),
            "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
            "output_files": sorted(path.name for path in output.iterdir()),
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"FAILURE: multi-checkpoint materialization preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    run()
