"""Protected split loader and cutoff-safe graph-only feature contract."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from effective_models.twitter_graph_only_rf.config import (
    DATA_ROOT, EARLY_GRAPH_FEATURES, ELIGIBLE_SPLITS, SPLIT_ROOT,
)


METADATA_OR_OUTCOME_COLUMNS = {
    "preventable_impact",
    "preventable_y",
    "future_reachable_node_count",
    "raw_node_count",
    "raw_edges",
    "raw_forward_edges",
    "reachable_nodes",
    "unreachable_nodes",
    "original_label",
    "corpus",
    "sample_id",
    "thread_id",
    "split",
}


def load_protected_data() -> tuple[pd.DataFrame, dict]:
    """Load the immutable records and predefined source-level split."""
    records = pd.read_csv(
        DATA_ROOT / "thread_level_records.csv",
        dtype={
            "sample_id": "string",
            "thread_id": "string",
            "corpus": "string",
            "original_label": "string",
        },
    )
    assignments = pd.read_csv(
        SPLIT_ROOT / "assignments.csv",
        dtype={"sample_id": "string", "original_label": "string", "split": "string"},
    )
    manifest = json.loads((SPLIT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    frame = records.merge(
        assignments[["sample_id", "split"]], on="sample_id", validate="one_to_one"
    )
    if len(frame) != len(records) or set(frame.split) != set(ELIGIBLE_SPLITS):
        raise ValueError("Protected source split does not cover every record")
    missing = set(EARLY_GRAPH_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing graph-only early features: {sorted(missing)}")
    if set(EARLY_GRAPH_FEATURES) & METADATA_OR_OUTCOME_COLUMNS:
        raise ValueError("Unsafe graph-only feature list")
    if not np.isfinite(frame[list(EARLY_GRAPH_FEATURES)].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite early graph feature")
    return frame, manifest
