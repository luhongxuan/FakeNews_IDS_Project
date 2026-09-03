"""Bounded safety smoke for a new protected-v5 strict-60 derived dataset."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
V5 = ROOT / "data" / "protected_research_assets" / "pheme_v5_strict30"
FEATURES = V5 / "pheme_node_features_roberta_replyv4.csv"
REPLIES = V5 / "pheme_reply_level_v4.csv"
LEGACY = ROOT / "research_scratch" / "legacy_full" / "graphsage_intervention_6" / "prepare_60min_preventable_impact_dataset.py"
OUT_ROOT = Path(__file__).resolve().parent / "experiments"
FEATURE_ROWS = 20_000
MUTABLE = ("followers_log", "friends_log", "statuses_log", "verified")

def write(output: Path, value: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(value, indent=2), encoding="utf-8")

def load_builder():
    spec = importlib.util.spec_from_file_location("strict60_builder", LEGACY)
    if spec is None or spec.loader is None: raise ImportError(LEGACY)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.WINDOW_SEC = 3600.0
    return module

def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_v5_strict60_materialization_smoke")
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "mode": "smoke", "feature_source": str(FEATURES), "reply_source": str(REPLIES), "cutoff_seconds": 3600, "feature_rows": FEATURE_ROWS, "mutable_columns_zeroed": list(MUTABLE), "research_safety": "The smoke creates new in-memory graphs only. Snapshot node/edge inputs are <=3600 seconds; mutable profile columns are overwritten with zero before graph construction; future nodes define only the recomputed 60-minute target."}
    write(output, record)
    try:
        print("Starting protected-v5 strict-60 materialization smoke.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/5] Loading a bounded prefix of protected v5 node features...", flush=True)
        builder = load_builder()
        features = pd.read_csv(FEATURES, nrows=FEATURE_ROWS, dtype={"thread_id": str, "tweet_id": str}, low_memory=False)
        if features.duplicated(["thread_id", "tweet_id"]).any(): raise ValueError("Duplicate feature identity")
        for column in MUTABLE: features[column] = 0.0
        thread_ids = set(features.thread_id)
        print(f"[2/5] Loading reply trees for {len(thread_ids)} bounded smoke threads...", flush=True)
        reply = pd.read_csv(REPLIES, dtype={"thread_id": str, "tweet_id": str, "parent_id": str}, low_memory=False)
        reply = reply.loc[reply.thread_id.isin(thread_ids)].copy()
        if reply.offset_sec.isna().any() or (reply.offset_sec < 0).any(): raise ValueError("Invalid offsets")
        maps = builder.thread_maps(reply)
        print("[3/5] Constructing deterministic <=60-minute snapshots and targets...", flush=True)
        graphs = []
        for index, (thread_id, group) in enumerate(features.loc[features.is_rumour.eq(1)].groupby("thread_id", sort=False), 1):
            if str(thread_id) not in maps: continue
            graph = builder.snapshot_graph(group, *maps[str(thread_id)])
            if graph is not None: graphs.append(graph)
            if index % 25 == 0: print(f"  smoke threads: {index}", flush=True)
        print("[4/5] Auditing cutoff, edges, text layout, and recomputed target...", flush=True)
        builder.validate(graphs, maps)
        for graph in graphs:
            _, offsets = maps[str(graph.thread_id)]
            if any(offsets[str(node)] > 3600 for node in graph.node_ids): raise ValueError("Post-cutoff node")
            if graph.x.shape[1] != 781 or graph.x[:, 13:].shape[1] != 768: raise ValueError("Text layout mismatch")
            if not torch.allclose(graph.preventable_y, torch.log1p(graph.preventable_impact)): raise ValueError("Target mismatch")
            if not np.allclose(graph.x[:, 8:12].numpy(), 0.0): raise ValueError("Mutable profile values survived")
        print("[5/5] Writing smoke evidence...", flush=True)
        summary = pd.DataFrame({"thread_id": [str(g.thread_id) for g in graphs], "event_id": [str(g.event_id) for g in graphs], "observed_nodes_60m": [g.num_nodes for g in graphs], "preventable_impact_60m": [float(g.preventable_impact.item()) for g in graphs]})
        summary.to_csv(output / "smoke_graph_summary.csv", index=False)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "graphs": len(graphs), "events": sorted(summary.event_id.unique()), "output_files": ["smoke_graph_summary.csv"]}); write(output, record)
        print(f"SUCCESS: protected-v5 strict-60 materialization smoke saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}); write(output, record)
        print(f"FAILURE: protected-v5 strict-60 materialization smoke preserved at: {output.resolve()}", flush=True); raise

if __name__ == "__main__": main()
