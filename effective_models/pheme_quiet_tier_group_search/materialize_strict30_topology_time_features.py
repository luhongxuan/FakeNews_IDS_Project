"""Materialize safe topology-by-time interactions from strict-30 features."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
SOURCE = BASE / "experiments" / "20260902_030441_849111_strict30_protected_feature_materialization" / "strict30_protected_snapshot_features.csv"
OUT_ROOT = BASE / "experiments"
EXPECTED_THREADS = 2402


def topology_time_block(frame: pd.DataFrame) -> pd.DataFrame:
    required = ("thread_id", "activity_observed_replies", "temporal_active_minute_count", "temporal_first10_fraction", "temporal_mid10_fraction", "temporal_last10_fraction", "temporal_hist_1m_entropy", "temporal_hist_1m_gini", "topology_max_depth", "topology_leaf_fraction", "topology_root_children", "topology_width_to_depth", "topology_depth_entropy", "topology_outdegree_entropy", "topology_outdegree_gini", "topology_outdegree_hhi")
    if any(column not in frame for column in required): raise ValueError("missing strict-30 topology/time inputs")
    x = frame.copy()
    reply_scale = np.log1p(x.activity_observed_replies)
    return pd.DataFrame({
        "thread_id": x.thread_id.astype(str),
        "tt_root_children_per_reply": x.topology_root_children / np.maximum(x.activity_observed_replies, 1.0),
        "tt_depth_per_active_minute": x.topology_max_depth / np.maximum(x.temporal_active_minute_count, 1.0),
        "tt_width_per_active_minute": x.topology_width_to_depth / np.maximum(x.temporal_active_minute_count, 1.0),
        "tt_root_children_x_first10": x.topology_root_children * x.temporal_first10_fraction,
        "tt_root_children_x_last10": x.topology_root_children * x.temporal_last10_fraction,
        "tt_depth_x_first10": x.topology_max_depth * x.temporal_first10_fraction,
        "tt_depth_x_last10": x.topology_max_depth * x.temporal_last10_fraction,
        "tt_leaf_x_time_entropy": x.topology_leaf_fraction * x.temporal_hist_1m_entropy,
        "tt_leaf_x_time_gini": x.topology_leaf_fraction * x.temporal_hist_1m_gini,
        "tt_outdegree_hhi_x_last10": x.topology_outdegree_hhi * x.temporal_last10_fraction,
        "tt_outdegree_gini_x_time_gini": x.topology_outdegree_gini * x.temporal_hist_1m_gini,
        "tt_depth_entropy_x_time_entropy": x.topology_depth_entropy * x.temporal_hist_1m_entropy,
        "tt_outdegree_entropy_x_time_entropy": x.topology_outdegree_entropy * x.temporal_hist_1m_entropy,
        "tt_log_replies_x_depth": reply_scale * x.topology_max_depth,
        "tt_log_replies_x_root_children": reply_scale * x.topology_root_children,
    })


def write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_strict30_topology_time_feature_materialization")
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "purpose": "new topology plus topology-by-time block; no model fitting", "cutoff_seconds": 1800, "source_feature_table": str(SOURCE.resolve()), "research_safety": "All columns are deterministic transformations of existing strict-30 cutoff-safe topology and temporal features; outcomes are not read by topology_time_block."}
    write_record(output, record)
    try:
        print("Starting strict-30 topology-time feature materialization (foreground job).", flush=True); print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/4] Loading and validating protected strict-30 feature table...", flush=True)
        source = pd.read_csv(SOURCE, dtype={"thread_id": str, "event_id": str})
        if len(source) != EXPECTED_THREADS or source.thread_id.duplicated().any(): raise ValueError("invalid source feature table")
        print("[2/4] Constructing topology-by-time interaction block...", flush=True)
        block = topology_time_block(source)
        if block.thread_id.duplicated().any() or not np.isfinite(block.drop(columns="thread_id").to_numpy(float)).all(): raise ValueError("invalid topology-time block")
        print("[3/4] Joining new block without changing identities or outcomes...", flush=True)
        combined = source.merge(block, on="thread_id", how="inner", validate="one_to_one")
        if len(combined) != EXPECTED_THREADS or combined.thread_id.duplicated().any(): raise ValueError("feature join changed identities")
        print("[4/4] Writing new artifact and run record...", flush=True)
        block.to_csv(output / "topology_time_feature_block.csv", index=False); combined.to_csv(output / "strict30_protected_topology_time_snapshot_features.csv", index=False)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "metrics": {"threads": len(combined), "new_columns": len(block.columns)-1}, "output_files": ["topology_time_feature_block.csv", "strict30_protected_topology_time_snapshot_features.csv"]}); write_record(output, record)
        print(f"SUCCESS: strict-30 topology-time feature materialization saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}); write_record(output, record); print(f"FAILURE: topology-time feature materialization preserved at: {output.resolve()}", flush=True); raise


if __name__ == "__main__":
    main()
