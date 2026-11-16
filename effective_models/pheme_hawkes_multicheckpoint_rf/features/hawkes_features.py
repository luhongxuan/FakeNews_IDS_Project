from __future__ import annotations

import json
import numpy as np
import pandas as pd

from effective_models.pheme_hawkes_multicheckpoint_rf import config

OUTCOMES = ("dynamic_preventable_impact", "dynamic_preventable_y", "future_growth")


def load_inputs() -> tuple[pd.DataFrame, list[str]]:
    for path in (config.FEATURE_TABLE, config.FEATURE_SCHEMA, config.REPLY_TABLE):
        if not path.exists():
            raise FileNotFoundError(f"Required local artifact is missing: {path}")
    frame = pd.read_csv(config.FEATURE_TABLE, dtype={"thread_id": str})
    schema = json.loads(config.FEATURE_SCHEMA.read_text(encoding="utf-8"))
    columns = list(schema["cumulative_feature_columns"])
    required = {"thread_id", "event_id", "checkpoint_sec", *OUTCOMES, *columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Frozen feature table is missing columns: {missing}")
    if frame.duplicated(["thread_id", "checkpoint_sec"]).any():
        raise ValueError("Duplicate thread/checkpoint identity")
    if set(columns) & set(OUTCOMES):
        raise ValueError("Outcome entered cumulative feature allowlist")
    if not np.allclose(np.log1p(frame.dynamic_preventable_impact), frame.dynamic_preventable_y):
        raise ValueError("Target is not log1p(dynamic_preventable_impact)")
    return frame, columns


def add_fixed_decay_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    replies = pd.read_csv(
        config.REPLY_TABLE,
        usecols=["thread_id", "is_source", "offset_sec"],
        dtype={"thread_id": str},
    )
    replies = replies.loc[
        replies.thread_id.isin(set(frame.thread_id)) & replies.is_source.ne(1)
    ]
    offsets = {
        key: group.offset_sec.to_numpy(dtype=float)
        for key, group in replies.groupby("thread_id", sort=False)
    }
    rows = []
    for item in frame[["thread_id", "checkpoint_sec"]].itertuples(index=False):
        checkpoint = float(item.checkpoint_sec)
        observed = offsets.get(str(item.thread_id), np.empty(0, dtype=float))
        observed = observed[observed <= checkpoint]
        ages = checkpoint - observed
        intensity = {
            decay: float(np.exp(-ages / decay).sum()) if ages.size else 0.0
            for decay in config.DECAYS_SECONDS
        }
        rows.append({
            "thread_id": str(item.thread_id),
            "checkpoint_sec": int(checkpoint),
            **{f"hawkes_exp_decay_{int(d)}s": float(np.log1p(v)) for d, v in intensity.items()},
            "hawkes_short_to_long_ratio": intensity[300.0] / (intensity[1200.0] + 1e-8),
            "hawkes_recent_share": (
                intensity[300.0] - intensity[1200.0] * np.exp(-900.0 / 1200.0)
            ) / (intensity[300.0] + intensity[1200.0] + 1e-8),
        })
    additions = pd.DataFrame(rows)
    names = [name for name in additions if name.startswith("hawkes_")]
    result = frame.merge(additions, on=["thread_id", "checkpoint_sec"], validate="one_to_one")
    if not np.isfinite(result[names].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite fixed-decay feature")
    return result, names

