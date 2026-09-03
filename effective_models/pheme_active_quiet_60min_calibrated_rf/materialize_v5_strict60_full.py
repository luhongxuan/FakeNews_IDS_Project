"""Foreground full protected-v5 strict-60 rumour snapshot materialization."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import torch

import materialize_v5_strict60_smoke as common

OUT_ROOT = Path(__file__).resolve().parent / "experiments"

def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_v5_strict60_materialization_full")
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / "pheme_graphs_roberta_v5_strict60_preventableimpact.pt"
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "mode": "full", "feature_source": str(common.FEATURES), "reply_source": str(common.REPLIES), "cutoff_seconds": 3600, "output_dataset": str(dataset), "mutable_columns_zeroed": list(common.MUTABLE), "target": "log1p(future nodes preventable by intervening every observed <=60-minute node)", "research_safety": "New derived artifact only; protected sources are read-only. Every snapshot node is <=3600 seconds, edges join observed nodes, mutable profile fields are zeroed before graph construction, and post-cutoff nodes define labels only."}
    common.write(output, record)
    try:
        print("Starting protected-v5 strict-60 full materialization (foreground job).", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/6] Loading protected v5 reply trees...", flush=True)
        reply = pd.read_csv(common.REPLIES, dtype={"thread_id": str, "tweet_id": str, "parent_id": str}, low_memory=False)
        if reply.duplicated(["thread_id", "tweet_id"]).any() or reply.offset_sec.isna().any() or (reply.offset_sec < 0).any(): raise ValueError("Invalid protected reply identities or offsets")
        builder = common.load_builder(); maps = builder.thread_maps(reply)
        print(f"  reply rows={len(reply)}, threads={len(maps)}", flush=True)
        print("[2/6] Loading protected v5 node features and zeroing excluded mutable fields...", flush=True)
        features = pd.read_csv(common.FEATURES, dtype={"thread_id": str, "tweet_id": str}, low_memory=False)
        if features.duplicated(["thread_id", "tweet_id"]).any(): raise ValueError("Duplicate protected node identity")
        for column in common.MUTABLE: features[column] = 0.0
        rumour = features.loc[features.is_rumour.eq(1)]
        print(f"  node rows={len(features)}, rumour threads={rumour.thread_id.nunique()}", flush=True)
        print("[3/6] Constructing deterministic strict-60 graphs and recomputed targets...", flush=True)
        graphs = []
        groups = list(rumour.groupby("thread_id", sort=False)); total = len(groups)
        for index, (thread_id, group) in enumerate(groups, 1):
            key = str(thread_id)
            if key not in maps: raise KeyError(f"Missing reply tree for {key}")
            graph = builder.snapshot_graph(group, *maps[key])
            if graph is None: raise ValueError(f"Empty source snapshot for {key}")
            graphs.append(graph)
            if index % 100 == 0 or index == total: print(f"  graphs {index}/{total}", flush=True)
        print("[4/6] Auditing cutoff, edges, text layout, targets, and excluded fields...", flush=True)
        builder.validate(graphs, maps)
        for index, graph in enumerate(graphs, 1):
            _, offsets = maps[str(graph.thread_id)]
            if any(offsets[str(node)] > 3600 for node in graph.node_ids): raise ValueError(f"Post-cutoff node in {graph.thread_id}")
            if graph.x.shape[1] != 781 or graph.x[:, 13:].shape[1] != 768: raise ValueError(f"Text layout mismatch in {graph.thread_id}")
            if not np.allclose(graph.x[:, 8:12].numpy(), 0.0): raise ValueError(f"Mutable profile survived in {graph.thread_id}")
            if index % 500 == 0 or index == len(graphs): print(f"  audit {index}/{len(graphs)}", flush=True)
        print("[5/6] Saving new derived strict-60 artifact...", flush=True)
        torch.save(graphs, dataset)
        summary = pd.DataFrame({"thread_id": [str(g.thread_id) for g in graphs], "event_id": [str(g.event_id) for g in graphs], "observed_nodes_60m": [g.num_nodes for g in graphs], "preventable_impact_60m": [float(g.preventable_impact.item()) for g in graphs]})
        summary.to_csv(output / "graph_summary.csv", index=False)
        print("[6/6] Finalizing complete run record...", flush=True)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "graphs": len(graphs), "events": sorted(summary.event_id.unique()), "nonzero_impact": int((summary.preventable_impact_60m > 0).sum()), "output_files": [dataset.name, "graph_summary.csv"]}); common.write(output, record)
        print(f"SUCCESS: protected-v5 strict-60 materialization saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}); common.write(output, record); print(f"FAILURE: protected-v5 strict-60 materialization preserved at: {output.resolve()}", flush=True); raise

if __name__ == "__main__": main()
