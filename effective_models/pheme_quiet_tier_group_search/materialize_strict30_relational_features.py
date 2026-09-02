"""Materialize a new strict-30 feature artifact with source/reply relations."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import pandas as pd
import torch

from strict30_protected_feature_registry import STRICT30_ROOT, read_protected_raw_frames
from strict30_relational_feature_registry import relational_features


BASE = Path(__file__).resolve().parent
SOURCE_MATERIALIZATION = BASE / "experiments" / "20260902_030441_849111_strict30_protected_feature_materialization"
SOURCE_TABLE = SOURCE_MATERIALIZATION / "strict30_protected_snapshot_features.csv"
DATASET = STRICT30_ROOT / "pheme_graphs_roberta_30min_replyv4_preventableimpact_semantic.pt"
OUT_ROOT = BASE / "experiments"
EXPECTED_THREADS = 2402


def write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_strict30_relational_feature_materialization")
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "purpose": "new cutoff-safe source/reply relational feature block; no model fitting", "cutoff_seconds": 1800, "source_feature_table": str(SOURCE_TABLE.resolve()), "graph_snapshots": str(DATASET.resolve()), "research_safety": "Each relational feature is derived solely from nodes present in a validated strict-30 graph and approved raw reply/node columns at offset<=1800."}
    write_record(output, record)
    try:
        print("Starting strict-30 relational feature materialization (foreground job).", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/5] Loading and checking prior protected feature table...", flush=True)
        source = pd.read_csv(SOURCE_TABLE, dtype={"thread_id": str, "event_id": str})
        if len(source) != EXPECTED_THREADS or source.thread_id.duplicated().any(): raise ValueError("invalid source feature table")
        print("[2/5] Loading strict-30 snapshots and approved raw fields...", flush=True)
        graphs = torch.load(DATASET, weights_only=False)
        if len(graphs) != EXPECTED_THREADS or len({str(g.thread_id) for g in graphs}) != EXPECTED_THREADS: raise ValueError("invalid protected snapshots")
        replies, nodes = read_protected_raw_frames()
        replies_by = {str(key): value.copy() for key, value in replies.groupby(replies.thread_id.astype(str), sort=False)}
        nodes_by = {str(key): value.copy() for key, value in nodes.groupby(nodes.thread_id.astype(str), sort=False)}
        print("[3/5] Auditing snapshots and building source/reply relations...", flush=True)
        rows = []
        for index, graph in enumerate(graphs, 1):
            thread_id = str(graph.thread_id)
            if thread_id not in replies_by or thread_id not in nodes_by: raise ValueError(f"{thread_id}: missing approved raw rows")
            rows.append(relational_features(graph, replies_by[thread_id], nodes_by[thread_id]))
            if index % 100 == 0 or index == len(graphs): print(f"  materialized {index}/{len(graphs)} snapshots", flush=True)
        relational = pd.DataFrame(rows)
        if len(relational) != EXPECTED_THREADS or relational.thread_id.duplicated().any() or not relational.drop(columns="thread_id").apply(pd.to_numeric, errors="coerce").notna().all().all(): raise ValueError("invalid relational feature block")
        print("[4/5] Joining new block to protected feature table...", flush=True)
        combined = source.merge(relational, on="thread_id", how="inner", validate="one_to_one")
        if len(combined) != EXPECTED_THREADS or combined.thread_id.duplicated().any(): raise ValueError("join changed protected identities")
        combined.to_csv(output / "strict30_protected_relational_snapshot_features.csv", index=False)
        relational.to_csv(output / "relational_feature_block.csv", index=False)
        print("[5/5] Finalizing artifact record...", flush=True)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "metrics": {"threads": len(combined), "new_relational_columns": len(relational.columns) - 1, "combined_columns": len(combined.columns)}, "output_files": ["strict30_protected_relational_snapshot_features.csv", "relational_feature_block.csv"]})
        write_record(output, record)
        print(f"SUCCESS: strict-30 relational feature materialization saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
        write_record(output, record)
        print(f"FAILURE: relational feature materialization preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
