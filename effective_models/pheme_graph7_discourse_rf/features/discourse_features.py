"""Load and validate this model's frozen discourse/context feature artifact."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from effective_models.pheme_graph7_discourse_rf import config


def feature_variants(schema: dict[str, object]) -> dict[str, list[str]]:
    """Reproduce the three frozen feature bundles used by the reference runner."""
    base = list(schema["base_feature_columns"])
    discourse = list(schema["discourse_feature_columns"])
    context = [column for column in base if column.startswith("event_context_")]
    role = [column for column in base if column not in context]
    variants = {
        "role_base": role,
        "plus_event_context": role + context,
        "plus_event_context_discourse": role + context + discourse,
    }
    unsafe = set(schema["metadata_columns"] + schema["target_columns"])
    for name, columns in variants.items():
        overlap = set(columns) & unsafe
        if overlap:
            raise ValueError(f"Unsafe columns in {name}: {sorted(overlap)}")
    return variants


def load_schema_locked_data() -> tuple[pd.DataFrame, dict[str, object]]:
    schema_path = config.DATASET_DIR / "schema.json"
    records_path = config.DATASET_DIR / "thread_level_records.csv"
    if not schema_path.is_file() or not records_path.is_file():
        raise FileNotFoundError(
            "Missing local Graph7 artifact. Restore the paths documented in ARTIFACTS.md."
        )

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(records_path, dtype=schema["csv_dtypes"])
    variants = feature_variants(schema)
    required = set(schema["metadata_columns"] + schema["target_columns"])
    required.update(column for columns in variants.values() for column in columns)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing schema-locked columns: {missing}")
    if frame["thread_id"].duplicated().any():
        raise ValueError("Duplicate thread_id in schema-locked artifact")
    if frame[list(schema["metadata_columns"])].isna().any().any():
        raise ValueError("Missing model identity metadata")
    numeric = sorted(required - set(schema["metadata_columns"]))
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite feature or target value")
    expected_y = np.log1p(frame["preventable_impact"].to_numpy(dtype=float))
    if not np.allclose(expected_y, frame["preventable_y"].to_numpy(dtype=float)):
        raise ValueError("preventable_y is not log1p(preventable_impact)")
    return frame, schema
