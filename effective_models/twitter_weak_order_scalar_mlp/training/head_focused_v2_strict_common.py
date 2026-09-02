"""Strict train/validation loader and focused weak-order ablation helpers.

The loader deliberately reads only the split column for every record, then
skips every test row before parsing any feature or target column.  Therefore a
development run cannot inspect test outcomes accidentally.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA = BASE / "artifacts" / "20260901_133740_887252_twitter15_16_strict30_graph_head_features_v1"
RECORDS = DATA / "thread_level_records.csv"


def load_train_validation_only() -> tuple[pd.DataFrame, pd.DataFrame, list[str], int]:
    """Load train/validation records without parsing any test feature/target."""
    schema = json.loads((DATA / "schema.json").read_text(encoding="utf-8"))
    features = list(schema["feature_columns"])

    split_index = pd.read_csv(RECORDS, usecols=["split"], dtype={"split": "string"})
    allowed = {"train", "validation", "test"}
    observed = set(split_index["split"].dropna().unique())
    if not observed <= allowed or {"train", "validation", "test"} - observed:
        raise ValueError(f"Unexpected or missing split labels: {sorted(observed)}")
    test_lines = set((np.flatnonzero(split_index["split"].eq("test")) + 1).tolist())

    # CSV line 0 is the header; test record lines are skipped before their
    # target cells are parsed.  The preceding index read has no target columns.
    frame = pd.read_csv(
        RECORDS,
        skiprows=lambda line_number: line_number in test_lines,
        dtype={"corpus": "string", "thread_id": "string", "split": "string"},
    )
    if "test" in set(frame["split"].dropna().unique()):
        raise ValueError("Test rows were unexpectedly loaded.")
    train = frame[frame["split"] == "train"].copy()
    validation = frame[frame["split"] == "validation"].copy()
    if len(train) == 0 or len(validation) == 0:
        raise ValueError("Train or validation split is empty.")
    if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite train/validation feature value.")
    if not np.allclose(
        frame["preventable_y"].to_numpy(dtype=float),
        np.log1p(frame["preventable_impact"].to_numpy(dtype=float)),
    ):
        raise ValueError("Train/validation preventable_y integrity failure.")
    return train, validation, features, len(test_lines)
