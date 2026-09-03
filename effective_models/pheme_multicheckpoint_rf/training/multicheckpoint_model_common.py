"""Shared fitting and evaluation for pooled multi-checkpoint RF models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import multicheckpoint_common as features


MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 3,
    "max_features": 0.8,
    "n_jobs": 1,
    "random_state": features.SEED,
}
BUDGETS = (10, 20, 50)
FAMILIES = ("cumulative", "window_delta")


def latest_complete_materialization(mode: str) -> Path:
    suffix = f"_multicheckpoint_materialization_{mode}"
    candidates = []
    for directory in features.HERE.parent.joinpath("experiments").glob(f"*{suffix}"):
        record_path = directory / "run_record.json"
        dataset_path = directory / "pheme_multicheckpoint_features.csv"
        schema_path = directory / "feature_schema.json"
        if not (record_path.is_file() and dataset_path.is_file() and schema_path.is_file()):
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") == "complete":
            candidates.append(directory)
    if not candidates:
        raise FileNotFoundError(f"no complete {mode} multi-checkpoint materialization")
    return sorted(candidates)[-1]


def load_materialization(directory: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(
        directory / "pheme_multicheckpoint_features.csv",
        dtype={"thread_id": str, "event_id": str},
    )
    schema = json.loads((directory / "feature_schema.json").read_text(encoding="utf-8"))
    expected_checkpoints = set(features.CHECKPOINTS)
    if set(frame.checkpoint_sec.astype(int)) != expected_checkpoints:
        raise ValueError("unexpected checkpoint coverage")
    if frame.duplicated(["thread_id", "checkpoint_sec"]).any():
        raise ValueError("duplicate thread/checkpoint identity")
    return frame, schema


def family_columns(schema: dict, family: str) -> list[str]:
    columns = list(schema[f"{family}_feature_columns"])
    forbidden = ("impact", "future", "target", "oracle", "label", "score", "rank")
    bad = [column for column in columns if any(token in column.lower() for token in forbidden)]
    if bad:
        raise ValueError(f"outcome-like feature columns: {bad}")
    return columns


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    params: dict | None = None,
) -> tuple[RandomForestRegressor, np.ndarray]:
    if set(train.event_id) & set(test.event_id):
        raise ValueError("event overlap between train and test")
    x_train = train[columns].to_numpy(dtype=np.float32)
    x_test = test[columns].to_numpy(dtype=np.float32)
    y_train = train.dynamic_preventable_y.to_numpy(dtype=np.float32)
    if not all(np.isfinite(value).all() for value in (x_train, x_test, y_train)):
        raise ValueError("non-finite model array")
    model = RandomForestRegressor(**(params or MODEL_PARAMS)).fit(x_train, y_train)
    prediction = model.predict(x_test)
    if prediction.shape != (len(test),) or not np.isfinite(prediction).all():
        raise ValueError("invalid prediction")
    return model, prediction


def metric_rows(scored: pd.DataFrame, family: str) -> list[dict]:
    rows = []
    for (event, checkpoint), group in scored.groupby(["event_id", "checkpoint_sec"], sort=True):
        oracle = group.sort_values(
            ["dynamic_preventable_impact", "thread_id"],
            ascending=[False, True], kind="stable",
        )
        ranked = group.sort_values(
            ["prediction", "thread_id"], ascending=[False, True], kind="stable"
        )
        total_future = float(group.future_growth.sum())
        total_impact = float(group.dynamic_preventable_impact.sum())
        for budget in BUDGETS:
            actual = min(budget, len(group))
            chosen = ranked.head(actual)
            optimal = oracle.head(actual)
            blocked = float(chosen.dynamic_preventable_impact.sum())
            oracle_blocked = float(optimal.dynamic_preventable_impact.sum())
            oracle_ids = set(oracle.head(min(50, len(oracle))).thread_id)
            rows.append({
                "family": family, "event_id": event, "checkpoint_sec": int(checkpoint),
                "budget": budget, "selected": actual,
                "blocked_impact": blocked,
                "crr": blocked / total_future if total_future else 0.0,
                "preventable_recall": blocked / total_impact if total_impact else 0.0,
                "oracle_efficiency": blocked / oracle_blocked if oracle_blocked else 0.0,
                "oracle_top50_hits": len(set(chosen.thread_id) & oracle_ids),
            })
    return rows
