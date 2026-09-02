"""Bounded, leakage-safe feature-group ablation for strict-30 quiet tiers."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import torch

import run_strict30_two_expert_tier as base
from strict30_protected_feature_registry import feature_group


OUTER_EVENT = "charliehebdo"
INNER_EVENTS = 2
SOURCE_WEIGHT = 0.50
GROUP_CONFIGS = {
    "source_only": (),
    "tabular_activity": ("activity_level",),
    "tabular_temporal": ("temporal_dynamics",),
    "baseline_source_plus_activity_temporal": ("activity_level", "temporal_dynamics"),
    "source_plus_topology": ("topology",),
    "source_plus_text_surface": ("text_surface",),
    "source_plus_sentiment_account": ("sentiment", "account_age"),
    "source_plus_semantic_summary": ("semantic_summary",),
    "source_plus_activity_temporal_topology": ("activity_level", "temporal_dynamics", "topology"),
    "source_plus_activity_temporal_text": ("activity_level", "temporal_dynamics", "text_surface"),
    "source_plus_all_scalar": ("account_age", "activity_level", "semantic_summary", "sentiment", "temporal_dynamics", "text_surface", "topology"),
}


def write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def score_config(frame: pd.DataFrame, recency: dict[str, float], embeddings: dict[str, np.ndarray], columns: list[str], train_events: list[str], test_event: str, seed: int) -> dict:
    train, test, _ = base._quiet_partitions(frame, recency, train_events, test_event)
    if not columns:
        # Fit the source expert normally, then use source probabilities only.
        columns = [column for column in frame if feature_group(column) == "activity_level"]
        evaluated, source_prob, temporal_prob = base._fit_experts(train, test, columns, embeddings, seed)
        metrics, _ = base._evaluate_weight(evaluated, source_prob, temporal_prob, 1.0)
    else:
        evaluated, source_prob, temporal_prob = base._fit_experts(train, test, columns, embeddings, seed)
        metrics, _ = base._evaluate_weight(evaluated, source_prob, temporal_prob, SOURCE_WEIGHT)
    return metrics


def main() -> None:
    output = base.OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_strict30_feature_group_ablation_smoke")
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "mode": "smoke", "outer_event": OUTER_EVENT, "cutoff_seconds": 1800, "source_weight": SOURCE_WEIGHT, "configs": {key: list(value) for key, value in GROUP_CONFIGS.items()}, "split": "one outer event; two inner LOEO events; group choices are diagnostic only", "research_safety": "All tier labels are derived only inside each train partition; outer event outcomes do not choose a group."}
    write_record(output, record)
    try:
        print("Starting strict-30 feature-group ablation smoke.", flush=True); print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/5] Loading protected strict-30 inputs...", flush=True)
        materialization = json.loads(base.MATERIALIZATION_RECORD.read_text(encoding="utf-8"))
        if materialization.get("status") != "complete": raise ValueError("protected materialization incomplete")
        frame = pd.read_csv(base.FEATURE_TABLE, dtype={"thread_id": str, "event_id": str}); sizes = frame.groupby("event_id").size(); eligible = sorted(sizes[sizes >= 100].index.tolist())
        graphs = torch.load(base.DATASET, weights_only=False); recency = {}; embeddings = {}
        print("[2/5] Auditing snapshot embeddings...", flush=True)
        for index, graph in enumerate(graphs, 1):
            temporal = graph.temporal_features.detach().cpu().numpy().reshape(-1); source = torch.nonzero(graph.is_source_mask, as_tuple=False).flatten()
            if len(temporal) != 11 or source.numel() != 1: raise ValueError(f"{graph.thread_id}: invalid snapshot")
            root = int(source.item()); vectors = graph.x[:, 13:].detach().cpu().numpy().astype(np.float32); replies = np.delete(vectors, root, axis=0)
            recency[str(graph.thread_id)] = float(temporal[5]); embeddings[str(graph.thread_id)] = base._compose_embedding(vectors[root], replies)
            if index % 500 == 0 or index == len(graphs): print(f"  audited {index}/{len(graphs)} snapshots", flush=True)
        frame = frame.loc[frame.event_id.isin(eligible)].copy(); inner_events = [event for event in eligible if event != OUTER_EVENT][:INNER_EVENTS]
        rows = []
        print("[3/5] Running group ablations over two inner folds...", flush=True)
        for index, (name, groups) in enumerate(GROUP_CONFIGS.items(), 1):
            columns = [column for column in frame if feature_group(column) in groups]
            print(f"  config {index}/{len(GROUP_CONFIGS)}: {name} ({len(columns)} tabular columns)", flush=True)
            metrics = []
            for inner_index, event in enumerate(inner_events, 1):
                train_events = [value for value in eligible if value not in {OUTER_EVENT, event}]
                # Keep source and tabular forest randomness identical across
                # groups within a fold; otherwise group comparisons are noisy.
                metrics.append(score_config(frame, recency, embeddings, columns, train_events, event, base.SEED + inner_index))
            rows.append({"config": name, "feature_groups": "+".join(groups) if groups else "source_embedding_only", "tabular_feature_count": len(columns), "evaluation": "inner_mean", "high_top_fraction_overlap": float(np.mean([m["high_top_fraction_overlap"] for m in metrics])), "tier_macro_f1": float(np.mean([m["tier_macro_f1"] for m in metrics]))})
        print("[4/5] Evaluating all groups on the held-out smoke event...", flush=True)
        for index, (name, groups) in enumerate(GROUP_CONFIGS.items(), 1):
            columns = [column for column in frame if feature_group(column) in groups]
            metric = score_config(frame, recency, embeddings, columns, [event for event in eligible if event != OUTER_EVENT], OUTER_EVENT, base.SEED + 10000)
            rows.append({"config": name, "feature_groups": "+".join(groups) if groups else "source_embedding_only", "tabular_feature_count": len(columns), "evaluation": "outer_held_out", **metric})
        result = pd.DataFrame(rows)
        print("[5/5] Writing ablation table and run record...", flush=True)
        result.to_csv(output / "feature_group_ablation.csv", index=False)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "inner_events": inner_events, "output_files": ["feature_group_ablation.csv"]}); write_record(output, record)
        print(f"SUCCESS: strict-30 feature-group ablation smoke saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}); write_record(output, record); print(f"FAILURE: feature-group ablation smoke preserved at: {output.resolve()}", flush=True); raise


if __name__ == "__main__":
    main()
