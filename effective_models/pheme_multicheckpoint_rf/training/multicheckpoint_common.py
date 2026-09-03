"""Leakage-safe 10-minute cumulative and window/delta features for PHEME."""
from __future__ import annotations

from collections import defaultdict
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPLIES = (
    ROOT / "data" / "protected_research_assets" / "pheme_v5_strict30"
    / "pheme_reply_level_v4.csv"
)
CANDIDATE_GRAPHS = (
    ROOT / "effective_models" / "pheme_v5_corrected_target" / "experiments"
    / "20260903_143643_168807_corrected_v5_nested_full"
    / "pheme_v5_strict30_corrected_all_roots_semantic.pt"
)
CHECKPOINTS = tuple(range(600, 3601, 600))
WINDOW_SECONDS = 600.0
ELIGIBLE_MIN_THREADS = 100
SEED = 42
_TOKEN = re.compile(r"\b\w+\b")
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)


def load_replies() -> pd.DataFrame:
    frame = pd.read_csv(
        REPLIES,
        usecols=[
            "thread_id", "tweet_id", "parent_id", "is_source", "depth",
            "offset_sec", "reply_latency_sec", "text", "event_id",
        ],
        dtype={"thread_id": str, "tweet_id": str, "parent_id": str, "event_id": str},
        low_memory=False,
    )
    frame["offset_sec"] = pd.to_numeric(frame.offset_sec, errors="coerce")
    frame["depth"] = pd.to_numeric(frame.depth, errors="coerce")
    frame["reply_latency_sec"] = pd.to_numeric(frame.reply_latency_sec, errors="coerce")
    if frame[["thread_id", "tweet_id", "event_id", "offset_sec"]].isna().any().any():
        raise ValueError("reply table contains missing required values")
    if (frame.offset_sec < 0).any() or frame.duplicated(["thread_id", "tweet_id"]).any():
        raise ValueError("reply table has negative offsets or duplicate node identities")
    return frame


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
    result = {
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
    return result


def _cumulative_features(
    observed_replies: pd.DataFrame, source_text: pd.Series, source_id: str, checkpoint: float
) -> dict[str, float]:
    offsets = observed_replies.offset_sec.to_numpy(float)
    node_ids = set(observed_replies.tweet_id.astype(str)) | {source_id}
    child_count = defaultdict(int)
    for row in observed_replies.itertuples(index=False):
        parent = None if pd.isna(row.parent_id) else str(row.parent_id)
        if parent in node_ids:
            child_count[parent] += 1
    degrees = np.asarray([child_count[node_id] for node_id in node_ids], dtype=float)
    bins = np.asarray(
        [
            ((observed_replies.offset_sec > start) & (observed_replies.offset_sec <= start + 600)).sum()
            for start in range(0, 3600, 600)
        ], dtype=float,
    )
    result = {
        "elapsed_minutes": checkpoint / 60.0,
        "cum_observed_reply_count": float(len(observed_replies)),
        "cum_log1p_observed_reply_count": float(np.log1p(len(observed_replies))),
        "cum_reply_rate_per_minute": float(len(observed_replies) / (checkpoint / 60.0)),
        "cum_first_reply_offset": float(offsets.min()) if len(offsets) else 0.0,
        "cum_last_reply_offset": float(offsets.max()) if len(offsets) else 0.0,
        "cum_time_since_last_activity": float(checkpoint - offsets.max()) if len(offsets) else checkpoint,
        "cum_active_span": float(offsets.max() - offsets.min()) if len(offsets) else 0.0,
        "cum_active_minute_count": float(
            np.unique(np.floor(offsets / 60.0).astype(int)).size if len(offsets) else 0
        ),
        "cum_root_children": float(child_count[source_id]),
        "cum_leaf_fraction": float((degrees == 0).mean()),
        "cum_internal_fraction": float((degrees > 0).mean()),
        **{f"cum_bin_{index + 1}_count": float(value) for index, value in enumerate(bins)},
        **_safe_stats(np.diff(np.sort(offsets)), "cum_gap"),
        **_window_summary(observed_replies, source_id, "cum"),
        **_text_features(source_text, "source"),
    }
    return result


def _window_delta_features(
    replies: pd.DataFrame, source_text: pd.Series, source_id: str, checkpoint: float
) -> dict[str, float]:
    current = replies.loc[
        replies.offset_sec.gt(checkpoint - WINDOW_SECONDS) & replies.offset_sec.le(checkpoint)
    ]
    previous = replies.loc[
        replies.offset_sec.gt(checkpoint - 2 * WINDOW_SECONDS)
        & replies.offset_sec.le(checkpoint - WINDOW_SECONDS)
    ]
    current_features = _window_summary(current, source_id, "win_current")
    previous_features = _window_summary(previous, source_id, "win_previous")
    result = {
        "elapsed_minutes": checkpoint / 60.0,
        **_text_features(source_text, "source"),
        **current_features,
        **previous_features,
    }
    comparable = (
        "count", "reply_to_reply_count", "reply_to_reply_fraction", "unique_parent_count",
        "parent_concentration", "depth_mean", "depth_max", "gap_median", "latency_mean",
        "text_chars", "text_tokens", "text_unique_ratio", "url_count", "question_count",
        "exclamation_count", "mention_count", "hashtag_count", "uppercase_ratio",
    )
    for name in comparable:
        current_value = current_features[f"win_current_{name}"]
        previous_value = previous_features[f"win_previous_{name}"]
        result[f"delta_{name}"] = current_value - previous_value
    result["delta_count_ratio"] = (len(current) + 1.0) / (len(previous) + 1.0)
    result["delta_gap_compression"] = (
        previous_features["win_previous_gap_median"] + 1.0
    ) / (current_features["win_current_gap_median"] + 1.0)
    return result


def dynamic_preventable_impact(group: pd.DataFrame, checkpoint: float) -> int:
    parents = {
        str(row.tweet_id): None if pd.isna(row.parent_id) else str(row.parent_id)
        for row in group.itertuples(index=False)
    }
    offsets = {str(row.tweet_id): float(row.offset_sec) for row in group.itertuples(index=False)}
    children: dict[str, list[str]] = defaultdict(list)
    roots = []
    for node_id, parent_id in parents.items():
        if parent_id is None or parent_id not in parents:
            roots.append(node_id)
        else:
            children[parent_id].append(node_id)
    if not roots:
        raise ValueError("thread has no independently traversable root")
    observed = {node_id for node_id, offset in offsets.items() if offset <= checkpoint}
    visited: set[str] = set()
    blocked = 0

    def visit(node_id: str, inherited: bool) -> None:
        nonlocal blocked
        if node_id in visited:
            raise ValueError(f"cycle or duplicate traversal at {node_id}")
        visited.add(node_id)
        if offsets[node_id] <= checkpoint:
            child_block = node_id in observed
        else:
            if inherited:
                blocked += 1
            child_block = inherited
        for child_id in children.get(node_id, []):
            visit(child_id, child_block)

    for root in roots:
        visit(root, False)
    if len(visited) != len(parents):
        raise ValueError("not all nodes traversed")
    return blocked


def build_rows(raw: pd.DataFrame, max_threads_per_event: int | None = None) -> pd.DataFrame:
    groups = list(raw.groupby("thread_id", sort=True))
    if max_threads_per_event is not None:
        kept = []
        counts: dict[str, int] = defaultdict(int)
        for thread_id, group in groups:
            event = str(group.event_id.iloc[0])
            if counts[event] < max_threads_per_event:
                kept.append((thread_id, group))
                counts[event] += 1
        groups = kept
    rows = []
    for index, (thread_id, group) in enumerate(groups, 1):
        events = group.event_id.astype(str).unique()
        source = group.loc[group.is_source.eq(1)]
        if len(events) != 1 or len(source) != 1:
            raise ValueError(f"{thread_id}: invalid event/source identity")
        source_id = str(source.iloc[0].tweet_id)
        replies = group.loc[group.is_source.ne(1)].sort_values(
            ["offset_sec", "tweet_id"], kind="stable"
        )
        for checkpoint in CHECKPOINTS:
            cumulative = _cumulative_features(
                replies.loc[replies.offset_sec.le(checkpoint)], source.text, source_id, checkpoint
            )
            window = _window_delta_features(replies, source.text, source_id, checkpoint)
            impact = dynamic_preventable_impact(group, checkpoint)
            rows.append({
                "thread_id": str(thread_id), "event_id": events[0],
                "checkpoint_sec": checkpoint,
                "dynamic_preventable_impact": impact,
                "dynamic_preventable_y": math.log1p(impact),
                "future_growth": int(replies.offset_sec.gt(checkpoint).sum()),
                **{f"cumulative__{name}": value for name, value in cumulative.items()},
                **{f"window_delta__{name}": value for name, value in window.items()},
            })
        if index % 300 == 0 or index == len(groups):
            print(f"  materialized threads {index}/{len(groups)}", flush=True)
    result = pd.DataFrame(rows)
    if result.duplicated(["thread_id", "checkpoint_sec"]).any():
        raise ValueError("duplicate thread/checkpoint rows")
    return result


def feature_columns(frame: pd.DataFrame, family: str) -> list[str]:
    prefix = family + "__"
    columns = [column for column in frame.columns if column.startswith(prefix)]
    if not columns:
        raise ValueError(f"no {family} features")
    numeric = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"non-finite {family} features")
    return columns
