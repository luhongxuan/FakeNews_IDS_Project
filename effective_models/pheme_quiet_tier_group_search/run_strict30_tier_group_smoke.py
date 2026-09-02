"""Strict-30 quiet Oracle-tier feature-group smoke.

This candidate reuses the already-audited broad scalar *schema* solely for a
smoke.  It rechecks every protected snapshot against protected offsets before
fitting.  Full nested 255-combination search is intentionally not run here.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
import torch


RUN_MODE = "smoke"
SMOKE_OUTER_EVENT = "charliehebdo"
SMOKE_MAX_TRAIN_PER_EVENT = 80
CUTOFF_SECONDS = 1800.0
QUIET_QUANTILE = 0.60
HIGH_FRACTION = 0.20
MIDDLE_CUMULATIVE_FRACTION = 0.50
SEED = 42

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
V5 = ROOT / "data" / "protected_research_assets" / "pheme_v5_strict30"
DATASET = V5 / "pheme_graphs_roberta_30min_replyv4_preventableimpact_semantic.pt"
OFFSETS = V5 / "pheme_reply_level_v4.csv"
SCALAR_SCHEMA = ROOT / "research_scratch" / "legacy_full" / "graphsage_intervention_14" / "experiments" / "20260901_214940_724199_quiet_broad_safe_feature_screen" / "broad_safe_snapshot_features.csv"
OUT_ROOT = BASE / "experiments"
META = {"event_id", "thread_id", "preventable_impact", "oracle_rank", "quiet", "oracle_top50"}


def group_of(name: str) -> str:
    if name.startswith("account_age_"): return "account_age"
    if name.startswith("vader_"): return "sentiment"
    if name.startswith(("source_text_", "reply_text_", "snapshot_text_")) or name == "observed_text_coverage": return "text_surface"
    if name.startswith("semantic_") or name in {"source_embedding_norm", "snapshot_embedding_centroid_norm", "source_reply_cosine_mean", "source_reply_cosine_std", "source_reply_cosine_min", "source_reply_cosine_max", "reply_embedding_centroid_distance"}: return "semantic_summary"
    if name.startswith(("root_", "internal_", "leaf_", "degree_", "depth_", "outdegree_", "width_")): return "topology"
    if name.startswith("bin5_") or name in {"log1p_observed_nodes", "reply_fraction", "active_minute_count", "peak_minute_count", "first10_fraction", "middle10_fraction", "last10_fraction"}: return "activity_level"
    return "temporal_dynamics"


def load_offsets() -> dict[tuple[str, str], float]:
    data = pd.read_csv(OFFSETS, usecols=["thread_id", "tweet_id", "offset_sec"], dtype={"thread_id": str, "tweet_id": str})
    if data.duplicated(["thread_id", "tweet_id"]).any() or data.offset_sec.isna().any() or (data.offset_sec < 0).any():
        raise ValueError("Protected offset table is invalid")
    return {(row.thread_id, row.tweet_id): float(row.offset_sec) for row in data.itertuples(index=False)}


def audit(graphs: list, offsets: dict[tuple[str, str], float]) -> dict[str, float]:
    if len(graphs) != 2402 or len({str(g.thread_id) for g in graphs}) != 2402: raise ValueError("Unexpected protected graph identities")
    recency: dict[str, float] = {}
    for index, graph in enumerate(graphs, 1):
        if graph.temporal_features.numel() != 11 or not torch.isfinite(graph.temporal_features).all(): raise ValueError(f"Invalid temporal input {graph.thread_id}")
        if graph.edge_index.numel() and (int(graph.edge_index.min()) < 0 or int(graph.edge_index.max()) >= graph.num_nodes): raise ValueError(f"Invalid edge {graph.thread_id}")
        if not torch.allclose(graph.preventable_y, torch.log1p(graph.preventable_impact)): raise ValueError(f"Target mismatch {graph.thread_id}")
        for node in graph.node_ids:
            value = offsets.get((str(graph.thread_id), str(node)))
            if value is None or value > CUTOFF_SECONDS + 1e-9: raise ValueError(f"Future/missing snapshot node {graph.thread_id}/{node}")
        recency[str(graph.thread_id)] = float(graph.temporal_features.detach().cpu().numpy().reshape(-1)[5])
        if index % 500 == 0 or index == len(graphs): print(f"  audited {index}/{len(graphs)} snapshots", flush=True)
    return recency


def tiers(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, pool in frame.groupby("event_id", sort=True):
        value = pool.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable").copy()
        high = max(1, math.ceil(len(value) * HIGH_FRACTION)); middle = max(high + 1, math.ceil(len(value) * MIDDLE_CUMULATIVE_FRACTION))
        value["tier"] = "low"; value.iloc[:high, value.columns.get_loc("tier")] = "high"; value.iloc[high:middle, value.columns.get_loc("tier")] = "middle"
        parts.append(value)
    return pd.concat(parts, ignore_index=True)


def semantic_matrix(graphs: list, ids: list[str]) -> np.ndarray:
    by_id = {str(g.thread_id): g for g in graphs}; rows = []
    for thread_id in ids:
        graph = by_id[thread_id]; source = torch.nonzero(graph.is_source_mask, as_tuple=False).flatten()
        if source.numel() != 1: raise ValueError(f"Missing source mask {thread_id}")
        emb = graph.x[:, 13:].detach().cpu().numpy().astype(np.float32)
        root = emb[int(source.item())]; replies = np.delete(emb, int(source.item()), axis=0); centroid = replies.mean(axis=0) if len(replies) else np.zeros_like(root)
        rows.append(np.concatenate([root, centroid]))
    return np.vstack(rows)


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_strict30_quiet_tier_all_groups_smoke")
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "mode": RUN_MODE, "dataset": str(DATASET), "scalar_schema": str(SCALAR_SCHEMA), "cutoff_seconds": CUTOFF_SECONDS, "split": "single held-out smoke fold only; not a formal LOEO result", "tiers": {"high": "top 20% quiet impact rank within event", "middle": "next 30%", "low": "remaining 50%"}, "research_safety": "Protected snapshot nodes are re-audited against protected <=1800-second offsets. Inputs are only pre-audited snapshot scalars and source/observed-reply embeddings with train-only PCA; no mutable profile/engagement field is read."}
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        print("Starting strict-30 all-group quiet tier smoke.", flush=True); print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/4] Loading protected graphs and auditing cutoff boundaries...", flush=True)
        graphs = torch.load(DATASET, weights_only=False); recency = audit(graphs, load_offsets())
        print("[2/4] Loading pre-audited scalar feature schema and reconciling protected identities...", flush=True)
        frame = pd.read_csv(SCALAR_SCHEMA, dtype={"event_id": str, "thread_id": str})
        features = [name for name in frame.columns if name not in META]
        if frame.thread_id.duplicated().any() or len(frame) != 2327 or not np.isfinite(frame[features].to_numpy(float)).all(): raise ValueError("Invalid archived scalar schema")
        impact = {str(g.thread_id): float(g.preventable_impact.item()) for g in graphs}
        if any(thread not in impact or not np.isclose(value, impact[thread]) for thread, value in zip(frame.thread_id, frame.preventable_impact)): raise ValueError("Scalar schema does not match protected target")
        frame["recency"] = frame.thread_id.map(recency); eligible = sorted(frame.event_id.unique()); outer = SMOKE_OUTER_EVENT
        raw_train, raw_test = frame[frame.event_id != outer].copy(), frame[frame.event_id == outer].copy(); threshold = float(np.quantile(raw_train.recency, QUIET_QUANTILE))
        train, test = raw_train[raw_train.recency >= threshold].copy(), raw_test[raw_test.recency >= threshold].copy()
        train = pd.concat([part.sort_values("thread_id").head(SMOKE_MAX_TRAIN_PER_EVENT) for _, part in train.groupby("event_id", sort=True)], ignore_index=True)
        print(f"  groups={sorted({group_of(name) for name in features})}; train quiet={len(train)}; test quiet={len(test)}", flush=True)
        print("[3/4] Fitting all predeclared scalar groups plus train-only source/reply semantic PCA...", flush=True)
        train = tiers(train); train_ids, test_ids = train.thread_id.tolist(), test.thread_id.tolist(); scaler = StandardScaler().fit(semantic_matrix(graphs, train_ids)); pca = PCA(n_components=16, random_state=SEED).fit(scaler.transform(semantic_matrix(graphs, train_ids)))
        x_train = np.hstack([train[features].to_numpy(np.float32), pca.transform(scaler.transform(semantic_matrix(graphs, train_ids)))]); x_test = np.hstack([test[features].to_numpy(np.float32), pca.transform(scaler.transform(semantic_matrix(graphs, test_ids)))])
        model = ExtraTreesClassifier(n_estimators=50, min_samples_leaf=3, max_features=.7, class_weight="balanced", n_jobs=1, random_state=SEED).fit(x_train, train.tier)
        probabilities = model.predict_proba(x_test); evaluated = tiers(test); predicted = model.classes_[probabilities.argmax(axis=1)]; evaluated["predicted_tier"] = predicted
        truth, selected = evaluated.tier.eq("high"), evaluated.predicted_tier.eq("high"); tp = int((truth & selected).sum()); precision = tp / int(selected.sum()) if selected.any() else 0.0; recall = tp / int(truth.sum()) if truth.any() else 0.0
        print("[4/4] Writing smoke evidence...", flush=True)
        evaluated.to_csv(output / "held_out_tier_predictions.csv", index=False)
        metrics = {"outer_event": outer, "quiet_threshold": threshold, "train_quiet_threads": len(train), "test_quiet_threads": len(evaluated), "test_high_threads": int(truth.sum()), "predicted_high_threads": int(selected.sum()), "high_precision": precision, "high_recall": recall}
        pd.DataFrame([metrics]).to_csv(output / "held_out_tier_metrics.csv", index=False)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "feature_count": len(features), "input_groups": sorted({group_of(name) for name in features}) + ["source_reply_embedding_pca16"], "metrics": metrics, "output_files": ["held_out_tier_predictions.csv", "held_out_tier_metrics.csv"]}); (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"SUCCESS: strict-30 all-group quiet tier smoke saved to: {output.resolve()}", flush=True)
    except Exception as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}"}); (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8"); print(f"FAILURE: output preserved at: {output.resolve()}", flush=True); raise


if __name__ == "__main__": main()
