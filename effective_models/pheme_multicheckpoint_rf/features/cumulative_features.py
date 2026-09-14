"""Canonical cumulative features and frozen-artifact validation.

The live feature builder intentionally shares the exact feature definitions used
to materialize the frozen training table.  It accepts an already cutoff-filtered
reply frame; callers remain responsible for enforcing the observation boundary.
"""
from __future__ import annotations

from collections import defaultdict
import json
import re

import numpy as np
import pandas as pd

from effective_models.pheme_multicheckpoint_rf import config


_TOKEN = re.compile(r"\b\w+\b")
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)


def _safe_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {f"{prefix}_{name}": 0.0 for name in ("mean", "std", "median", "max")}
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_max": float(values.max()),
    }


def _text_features(texts: pd.Series, prefix: str) -> dict[str, float]:
    clean = texts.fillna("").astype(str).tolist()
    chars = np.asarray([len(text) for text in clean], dtype=float)
    tokenized = [_TOKEN.findall(text.lower()) for text in clean]
    tokens = [token for row in tokenized for token in row]
    alpha = max(sum(sum(char.isalpha() for char in text) for text in clean), 1)
    return {
        f"{prefix}_text_rows": float(len(clean)),
        f"{prefix}_text_chars": float(chars.sum()),
        f"{prefix}_text_chars_mean": float(chars.mean()) if len(chars) else 0.0,
        f"{prefix}_text_tokens": float(len(tokens)),
        f"{prefix}_text_unique_ratio": float(len(set(tokens)) / max(len(tokens), 1)),
        f"{prefix}_url_count": float(sum(len(_URL.findall(text)) for text in clean)),
        f"{prefix}_question_count": float(sum(text.count("?") for text in clean)),
        f"{prefix}_exclamation_count": float(sum(text.count("!") for text in clean)),
        f"{prefix}_mention_count": float(sum(text.count("@") for text in clean)),
        f"{prefix}_hashtag_count": float(sum(text.count("#") for text in clean)),
        f"{prefix}_uppercase_ratio": float(
            sum(sum(char.isupper() for char in text) for text in clean) / alpha
        ),
    }


def _window_summary(rows: pd.DataFrame, source_id: str, prefix: str) -> dict[str, float]:
    offsets = rows.offset_sec.to_numpy(float)
    depths = rows.depth.fillna(0.0).to_numpy(float)
    gaps = np.diff(np.sort(offsets))
    parents = rows.parent_id.fillna(source_id).astype(str)
    count = len(rows)
    return {
        f"{prefix}_count": float(count),
        f"{prefix}_reply_to_reply_count": float(parents.ne(source_id).sum()),
        f"{prefix}_reply_to_reply_fraction": float(parents.ne(source_id).mean()) if count else 0.0,
        f"{prefix}_unique_parent_count": float(parents.nunique()) if count else 0.0,
        f"{prefix}_parent_concentration": float(parents.value_counts(normalize=True).max()) if count else 0.0,
        **_safe_stats(depths, f"{prefix}_depth"),
        **_safe_stats(gaps, f"{prefix}_gap"),
        **_safe_stats(rows.reply_latency_sec.to_numpy(float), f"{prefix}_latency"),
        **_text_features(rows.text, prefix),
    }


def build_cumulative_features(
    observed_replies: pd.DataFrame,
    source_text: pd.Series,
    source_id: str,
    checkpoint_sec: float,
) -> dict[str, float]:
    """Return the model's 57 prefixed features for one observable snapshot.

    ``observed_replies`` must contain only replies whose ``offset_sec`` is at
    or before ``checkpoint_sec``.  Rejecting future rows here provides a second
    guard in addition to the caller's snapshot filtering.
    """
    required = {"tweet_id", "parent_id", "depth", "offset_sec", "reply_latency_sec", "text"}
    missing = sorted(required - set(observed_replies.columns))
    if missing:
        raise ValueError(f"Missing live cumulative input columns: {missing}")
    offsets = pd.to_numeric(observed_replies["offset_sec"], errors="coerce")
    if offsets.isna().any() or offsets.gt(checkpoint_sec).any() or offsets.lt(0).any():
        raise ValueError("Live cumulative replies violate the observation cutoff")

    offsets_array = offsets.to_numpy(float)
    node_ids = set(observed_replies.tweet_id.astype(str)) | {str(source_id)}
    child_count: defaultdict[str, int] = defaultdict(int)
    for row in observed_replies.itertuples(index=False):
        parent = None if pd.isna(row.parent_id) else str(row.parent_id)
        if parent in node_ids:
            child_count[parent] += 1
    degrees = np.asarray([child_count[node_id] for node_id in node_ids], dtype=float)
    bins = np.asarray(
        [
            ((offsets > start) & (offsets <= start + 600)).sum()
            for start in range(0, 3600, 600)
        ],
        dtype=float,
    )
    raw = {
        "elapsed_minutes": checkpoint_sec / 60.0,
        "cum_observed_reply_count": float(len(observed_replies)),
        "cum_log1p_observed_reply_count": float(np.log1p(len(observed_replies))),
        "cum_reply_rate_per_minute": float(len(observed_replies) / (checkpoint_sec / 60.0)),
        "cum_first_reply_offset": float(offsets_array.min()) if len(offsets_array) else 0.0,
        "cum_last_reply_offset": float(offsets_array.max()) if len(offsets_array) else 0.0,
        "cum_time_since_last_activity": (
            float(checkpoint_sec - offsets_array.max()) if len(offsets_array) else float(checkpoint_sec)
        ),
        "cum_active_span": (
            float(offsets_array.max() - offsets_array.min()) if len(offsets_array) else 0.0
        ),
        "cum_active_minute_count": float(
            np.unique(np.floor(offsets_array / 60.0).astype(int)).size if len(offsets_array) else 0
        ),
        "cum_root_children": float(child_count[str(source_id)]),
        "cum_leaf_fraction": float((degrees == 0).mean()),
        "cum_internal_fraction": float((degrees > 0).mean()),
        **{f"cum_bin_{index + 1}_count": float(value) for index, value in enumerate(bins)},
        **_safe_stats(np.diff(np.sort(offsets_array)), "cum_gap"),
        **_window_summary(observed_replies, str(source_id), "cum"),
        **_text_features(source_text, "source"),
    }
    result = {f"cumulative__{name}": value for name, value in raw.items()}
    if len(result) != 57 or not np.isfinite(np.fromiter(result.values(), dtype=float)).all():
        raise ValueError("Invalid live cumulative feature vector")
    return result


def load_frozen_materialization() -> tuple[pd.DataFrame, list[str]]:
    if not config.DATASET_PATH.is_file() or not config.SCHEMA_PATH.is_file():
        raise FileNotFoundError("Missing frozen multi-checkpoint reference inputs; see ARTIFACTS.md")
    frame = pd.read_csv(config.DATASET_PATH, dtype={"thread_id": str, "event_id": str})
    schema = json.loads(config.SCHEMA_PATH.read_text(encoding="utf-8"))
    columns = list(schema["cumulative_feature_columns"])
    forbidden = ("impact", "future", "target", "oracle", "label", "score", "rank")
    unsafe = [column for column in columns if any(token in column.lower() for token in forbidden)]
    if unsafe:
        raise ValueError(f"Outcome-like cumulative feature columns: {unsafe}")
    required = {
        "thread_id", "event_id", "checkpoint_sec", "dynamic_preventable_impact",
        "dynamic_preventable_y", "future_growth", *columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing frozen columns: {missing}")
    if frame.duplicated(["thread_id", "checkpoint_sec"]).any():
        raise ValueError("Duplicate thread/checkpoint identity")
    if set(frame["checkpoint_sec"].astype(int)) != set(config.CHECKPOINTS):
        raise ValueError("Unexpected checkpoint coverage")
    if not frame.groupby("thread_id")["checkpoint_sec"].apply(
        lambda values: tuple(sorted(values.astype(int))) == config.CHECKPOINTS
    ).all():
        raise ValueError("Incomplete checkpoint sequence")
    numeric = columns + ["dynamic_preventable_impact", "dynamic_preventable_y", "future_growth"]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite frozen feature or outcome")
    expected = np.log1p(frame["dynamic_preventable_impact"].to_numpy(dtype=float))
    if not np.allclose(expected, frame["dynamic_preventable_y"].to_numpy(dtype=float)):
        raise ValueError("dynamic_preventable_y target mismatch")
    if (frame["dynamic_preventable_impact"] > frame["future_growth"]).any():
        raise ValueError("Preventable impact exceeds future growth")
    return frame, columns
