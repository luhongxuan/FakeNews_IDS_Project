"""Create a traceable, leakage-audited strict-30 feature table.

This is deliberately a foreground, user-run preparation job: it reads the
protected raw node table and every strict-30 graph snapshot.  It is not a model
training or hyperparameter-search script.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import pandas as pd
import torch

from strict30_protected_feature_registry import (
    CUTOFF_SECONDS,
    FEATURE_GROUPS,
    STRICT30_ROOT,
    feature_group,
    read_protected_raw_frames,
    scalar_features,
)


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
DATASET = STRICT30_ROOT / "pheme_graphs_roberta_30min_replyv4_preventableimpact_semantic.pt"
OUT_ROOT = BASE / "experiments"
EXPECTED_THREADS = 2402


def _write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = OUT_ROOT / f"{stamp}_strict30_protected_feature_materialization"
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "protected strict-30 cutoff-safe tier feature materialization; no model fitting",
        "configuration": {"cutoff_seconds": CUTOFF_SECONDS, "expected_threads": EXPECTED_THREADS},
        "inputs": {
            "graph_snapshots": str(DATASET.resolve()),
            "protected_reply_rows": str((STRICT30_ROOT / "pheme_reply_level_v4.csv").resolve()),
            "protected_node_features": str((STRICT30_ROOT / "pheme_node_features_roberta_replyv4.csv").resolve()),
        },
        "split": "no split used; feature construction only",
        "feature_groups": list(FEATURE_GROUPS),
        "research_safety": (
            "Uses only protected strict-30 graph nodes and offset<=1800 raw rows. "
            "No outcome, coverage, follower, friend, status, or verification field is read as a feature."
        ),
    }
    _write_record(output, record)
    try:
        print("Starting protected strict-30 feature materialization (foreground job).", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/5] Loading protected strict-30 graph snapshots...", flush=True)
        graphs = torch.load(DATASET, weights_only=False)
        if len(graphs) != EXPECTED_THREADS or len({str(graph.thread_id) for graph in graphs}) != EXPECTED_THREADS:
            raise ValueError(f"Expected {EXPECTED_THREADS} unique strict-30 graphs")
        print("[2/5] Loading only approved raw reply/node fields...", flush=True)
        replies, nodes = read_protected_raw_frames()
        reply_by_thread = {str(key): value.copy() for key, value in replies.groupby(replies.thread_id.astype(str), sort=False)}
        node_by_thread = {str(key): value.copy() for key, value in nodes.groupby(nodes.thread_id.astype(str), sort=False)}
        print("[3/5] Validating every snapshot and materializing scalar feature groups...", flush=True)
        rows = []
        for index, graph in enumerate(graphs, 1):
            thread_id = str(graph.thread_id)
            if thread_id not in reply_by_thread or thread_id not in node_by_thread:
                raise ValueError(f"{thread_id}: missing protected raw records")
            row = scalar_features(graph, reply_by_thread[thread_id], node_by_thread[thread_id])
            row["preventable_impact"] = float(graph.preventable_impact.item())
            row["preventable_y"] = float(graph.preventable_y.item())
            rows.append(row)
            if index % 100 == 0 or index == len(graphs):
                print(f"  materialized {index}/{len(graphs)} snapshots", flush=True)
        frame = pd.DataFrame(rows)
        if frame.thread_id.duplicated().any() or len(frame) != EXPECTED_THREADS:
            raise ValueError("Materialized output lost or duplicated thread identities")
        model_columns = [name for name in frame.columns if feature_group(name) is not None]
        if not model_columns or not frame[model_columns].apply(pd.to_numeric, errors="coerce").notna().all().all():
            raise ValueError("Feature table contains missing/non-numeric model feature values")
        if any(name in model_columns for name in ("preventable_impact", "preventable_y")):
            raise AssertionError("Outcome leaked into model feature columns")
        print("[4/5] Writing feature table and schema inventory...", flush=True)
        frame.to_csv(output / "strict30_protected_snapshot_features.csv", index=False)
        schema = pd.DataFrame([
            {"column": name, "feature_group": feature_group(name) or "metadata_or_outcome"}
            for name in frame.columns
        ])
        schema.to_csv(output / "feature_schema.csv", index=False)
        print("[5/5] Finalizing run record...", flush=True)
        record.update({
            "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
            "metrics": {"threads": len(frame), "model_feature_count": len(model_columns)},
            "output_files": ["strict30_protected_snapshot_features.csv", "feature_schema.csv"],
        })
        _write_record(output, record)
        print(f"SUCCESS: protected strict-30 feature materialization saved to: {output.resolve()}", flush=True)
    except Exception as error:
        record.update({
            "status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
        })
        _write_record(output, record)
        print(f"FAILURE: protected strict-30 feature materialization preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
