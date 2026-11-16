"""Load and validate this model's frozen schema-locked feature artifact."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from effective_models.pheme_graph7_author_aware_rf import config


def load_schema_locked_data() -> tuple[pd.DataFrame, dict[str, object]]:
    schema_path = config.DATASET_DIR / "schema.json"
    records_path = config.DATASET_DIR / "thread_level_records.csv"
    if not schema_path.is_file() or not records_path.is_file():
        raise FileNotFoundError(
            "Missing local Graph7 artifact. Restore the paths documented in ARTIFACTS.md."
        )

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(records_path, dtype=schema["csv_dtypes"])
    features = list(schema["feature_columns"])
    unsafe = set(features) & set(schema["metadata_columns"] + schema["target_columns"])
    if unsafe:
        raise ValueError(f"Unsafe schema columns used as features: {sorted(unsafe)}")
    if frame["thread_id"].duplicated().any():
        raise ValueError("Duplicate thread_id in schema-locked artifact")
    if frame[["thread_id", "event_id", "category"]].isna().any().any():
        raise ValueError("Missing model identity metadata")
    if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite author-aware feature value")
    if not np.isfinite(frame[["preventable_impact", "preventable_y"]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite target value")
    return frame, schema
