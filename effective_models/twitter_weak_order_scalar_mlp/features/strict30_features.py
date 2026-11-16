from __future__ import annotations
import json
import numpy as np
import pandas as pd
from effective_models.twitter_weak_order_scalar_mlp import config


def feature_columns() -> list[str]:
    if not config.SCHEMA.exists():
        raise FileNotFoundError(config.SCHEMA)
    schema = json.loads(config.SCHEMA.read_text(encoding="utf-8"))
    if int(schema["cutoff_seconds"]) != config.CUTOFF_SECONDS:
        raise ValueError("Unexpected cutoff")
    return list(schema["feature_columns"])


def load_non_test() -> tuple[pd.DataFrame, pd.DataFrame, list[str], int]:
    columns = feature_columns()
    split_index = pd.read_csv(config.RECORDS, usecols=["split"], dtype={"split": "string"})
    observed = set(split_index.split.dropna().unique())
    if observed != {"train", "validation", "test"}:
        raise ValueError(f"Unexpected split labels: {observed}")
    test_lines = set((np.flatnonzero(split_index.split.eq("test")) + 1).tolist())
    frame = pd.read_csv(config.RECORDS, skiprows=lambda line: line in test_lines,
                        dtype={"corpus": "string", "thread_id": "string", "split": "string"})
    if "test" in set(frame.split.dropna().unique()):
        raise ValueError("Protected test row entered development loader")
    _validate(frame, columns)
    return frame.loc[frame.split.eq("train")].copy(), frame.loc[frame.split.eq("validation")].copy(), columns, len(test_lines)


def load_test(columns: list[str]) -> pd.DataFrame:
    split_index = pd.read_csv(config.RECORDS, usecols=["split"], dtype={"split": "string"})
    non_test_lines = set((np.flatnonzero(~split_index.split.eq("test")) + 1).tolist())
    frame = pd.read_csv(config.RECORDS, skiprows=lambda line: line in non_test_lines,
                        dtype={"corpus": "string", "thread_id": "string", "split": "string"})
    if set(frame.split.dropna().unique()) != {"test"}:
        raise ValueError("Test loader returned non-test rows")
    _validate(frame, columns)
    return frame


def _validate(frame: pd.DataFrame, columns: list[str]) -> None:
    if frame.duplicated(["corpus", "thread_id"]).any():
        raise ValueError("Duplicate corpus/thread identity")
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite feature")
    if not np.allclose(frame.preventable_y, np.log1p(frame.preventable_impact)):
        raise ValueError("preventable_y target integrity failure")

