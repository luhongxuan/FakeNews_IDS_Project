"""Shared, protected-split utilities for Twitter15/16 graph-only ranking."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "derived" / "20260830_024514_251_twitter15_16_30min_time_respecting_preventable_impact"
SPLIT = ROOT / "data" / "derived" / "20260830_024515_294_twitter15_16_source_level_split_v1"
TOP_K = 10
SEED = 42
EARLY_GRAPH_FEATURES = [
    "observed_node_count", "observed_edge_count", "log1p_observed_nodes",
    "log1p_count_0_10m", "log1p_count_10_20m", "log1p_count_20_30m",
    "log1p_count_recent_5m", "log1p_count_recent_10m",
    "log1p_seconds_since_last_activity", "log1p_median_interarrival_sec",
    "log1p_std_interarrival_sec", "late_activity_frac", "observed_leaf_fraction",
    "mean_observed_children", "max_observed_children",
]
PARAMETERS = {
    "rf_depth_6": {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 3, "max_features": 0.8, "n_jobs": 1, "random_state": SEED},
    "rf_depth_10": {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 3, "max_features": 0.8, "n_jobs": 1, "random_state": SEED},
    "rf_unrestricted": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 3, "max_features": 0.8, "n_jobs": 1, "random_state": SEED},
}


def load_protected_data() -> tuple[pd.DataFrame, dict]:
    """Load only the pre-existing records/split, rejecting unsafe input columns."""
    records = pd.read_csv(DATA / "thread_level_records.csv", dtype={"sample_id": "string", "thread_id": "string", "corpus": "string", "original_label": "string"})
    assignments = pd.read_csv(SPLIT / "assignments.csv", dtype={"sample_id": "string", "original_label": "string", "split": "string"})
    manifest = json.loads((SPLIT / "manifest.json").read_text(encoding="utf-8"))
    frame = records.merge(assignments[["sample_id", "split"]], on="sample_id", validate="one_to_one")
    if len(frame) != len(records) or set(frame.split) != {"train", "validation", "test"}:
        raise ValueError("Protected source split does not cover every record")
    if set(EARLY_GRAPH_FEATURES) - set(frame.columns):
        raise ValueError("Missing graph-only early feature")
    forbidden = {"preventable_impact", "preventable_y", "future_reachable_node_count", "raw_node_count", "raw_edges", "raw_forward_edges", "reachable_nodes", "unreachable_nodes", "original_label", "corpus", "sample_id", "thread_id", "split"}
    if set(EARLY_GRAPH_FEATURES) & forbidden:
        raise ValueError("Unsafe graph-only feature list")
    if not np.isfinite(frame[EARLY_GRAPH_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite early graph feature")
    return frame, manifest


def evaluate(model: RandomForestRegressor, frame: pd.DataFrame) -> tuple[dict[str, float | int], pd.DataFrame]:
    pool = frame.copy()
    pool["score"] = model.predict(pool[EARLY_GRAPH_FEATURES])
    selected = pool.sort_values(["score", "thread_id"], ascending=[False, True], kind="stable").head(TOP_K).copy()
    oracle = pool.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable").head(TOP_K)
    total = float(pool.preventable_impact.sum())
    captured = float(selected.preventable_impact.sum())
    oracle_captured = float(oracle.preventable_impact.sum())
    result = {
        "candidates": int(len(pool)), "selected_preventable_impact": captured,
        "total_preventable_impact": total, "crr_at_10": captured / total if total else 0.0,
        "oracle_selected_preventable_impact": oracle_captured,
        "oracle_crr_at_10": oracle_captured / total if total else 0.0,
        "impact_efficiency_vs_oracle": captured / oracle_captured if oracle_captured else 0.0,
        "exact_oracle_overlap": int(selected.thread_id.isin(set(oracle.thread_id)).sum()),
    }
    selected["selected_by_model"] = True
    selected["selected_by_oracle"] = selected.thread_id.isin(set(oracle.thread_id))
    return result, selected


def fit(frame: pd.DataFrame, corpus: str, params: dict[str, object], split: str = "train") -> RandomForestRegressor:
    train = frame[(frame.corpus == corpus) & (frame.split == split)]
    return RandomForestRegressor(**params).fit(train[EARLY_GRAPH_FEATURES], train.preventable_y)
