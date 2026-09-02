"""Leakage-audited feature registry for strict-30 quiet-tier experiments.

This module deliberately materializes only features that are observable inside
the protected strict-30 snapshot.  It never exposes engagement/profile fields
such as followers, friends, statuses, verification, or coverage outcomes.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd


# Resolve from this source file, never from the caller's current directory.
# The foreground scripts are intentionally runnable from their own folder.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STRICT30_ROOT = REPOSITORY_ROOT / "data" / "protected_research_assets" / "pheme_v5_strict30"
REPLY_PATH = STRICT30_ROOT / "pheme_reply_level_v4.csv"
NODE_PATH = STRICT30_ROOT / "pheme_node_features_roberta_replyv4.csv"
CUTOFF_SECONDS = 30 * 60

SAFE_REPLY_COLUMNS = [
    "thread_id", "tweet_id", "parent_id", "is_source", "is_text_available",
    "depth", "offset_sec", "reply_latency_sec", "text", "event_id",
]
SAFE_NODE_COLUMNS = [
    "thread_id", "tweet_id", "event_id", "vader_compound", "vader_pos",
    "vader_neg", "vader_neu", "account_age_days_log",
]

FEATURE_GROUPS = (
    "account_age", "activity_level", "semantic_summary", "sentiment",
    "temporal_dynamics", "text_surface", "topology",
)


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def _moments(values: Iterable[float], prefix: str) -> dict[str, float]:
    array = _finite(values)
    if array.size == 0:
        return {f"{prefix}_{name}": 0.0 for name in ("mean", "std", "min", "max", "median")}
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_std": float(array.std()),
        f"{prefix}_min": float(array.min()),
        f"{prefix}_max": float(array.max()),
        f"{prefix}_median": float(np.median(array)),
    }


def _entropy(values: Iterable[float]) -> float:
    array = _finite(values)
    total = float(array.sum())
    if array.size == 0 or total <= 0:
        return 0.0
    probs = array[array > 0] / total
    return float(-(probs * np.log(probs)).sum())


def _gini(values: Iterable[float]) -> float:
    array = np.sort(_finite(values))
    if array.size == 0 or array.sum() <= 0:
        return 0.0
    return float((2.0 * np.dot(np.arange(1, array.size + 1), array) /
                  (array.size * array.sum())) - (array.size + 1.0) / array.size)


_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_TAG = re.compile(r"#[\w_]+")
_MENTION = re.compile(r"@[\w_]+")
_TOKEN = re.compile(r"\b\w+\b")


def _text_stats(texts: Iterable[object], prefix: str) -> dict[str, float]:
    clean = [str(text or "") for text in texts]
    lengths = np.asarray([len(text) for text in clean], dtype=float)
    tokens = [_TOKEN.findall(text.lower()) for text in clean]
    token_counts = np.asarray([len(item) for item in tokens], dtype=float)
    all_tokens = [token for item in tokens for token in item]
    counts = Counter(all_tokens)
    total_tokens = max(len(all_tokens), 1)
    upper_chars = sum(sum(char.isupper() for char in text) for text in clean)
    alpha_chars = max(sum(sum(char.isalpha() for char in text) for text in clean), 1)
    return {
        **_moments(lengths, f"{prefix}_chars"),
        **_moments(token_counts, f"{prefix}_tokens"),
        f"{prefix}_unique_token_ratio": float(len(counts) / total_tokens),
        f"{prefix}_token_entropy": _entropy(counts.values()),
        f"{prefix}_hapax_ratio": float(sum(value == 1 for value in counts.values()) / max(len(counts), 1)),
        f"{prefix}_url_count": float(sum(len(_URL.findall(text)) for text in clean)),
        f"{prefix}_hashtag_count": float(sum(len(_TAG.findall(text)) for text in clean)),
        f"{prefix}_mention_count": float(sum(len(_MENTION.findall(text)) for text in clean)),
        f"{prefix}_question_count": float(sum(text.count("?") for text in clean)),
        f"{prefix}_exclamation_count": float(sum(text.count("!") for text in clean)),
        f"{prefix}_quote_count": float(sum(text.count('"') + text.count("'") for text in clean)),
        f"{prefix}_digit_ratio": float(sum(sum(char.isdigit() for char in text) for text in clean) / max(lengths.sum(), 1.0)),
        f"{prefix}_uppercase_ratio": float(upper_chars / alpha_chars),
        f"{prefix}_retweet_count": float(sum(text.lower().startswith("rt ") for text in clean)),
    }


def read_protected_raw_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read only explicitly approved, cutoff-safe raw columns.

    The caller is responsible for writing a run record.  This function refuses
    malformed identifiers before feature construction begins.
    """
    replies = pd.read_csv(REPLY_PATH, usecols=SAFE_REPLY_COLUMNS, low_memory=False)
    nodes = pd.read_csv(NODE_PATH, usecols=SAFE_NODE_COLUMNS, low_memory=False)
    for name, frame in (("replies", replies), ("nodes", nodes)):
        if frame[["thread_id", "tweet_id"]].isna().any().any():
            raise ValueError(f"protected {name} contains missing identifiers")
        if frame.duplicated(["thread_id", "tweet_id"]).any():
            raise ValueError(f"protected {name} contains duplicate thread/tweet identifiers")
    # The protected reply table is an event-level provenance table and may
    # retain post-cutoff rows.  They are not errors by themselves, but they
    # must never reach feature construction.  Keep a deterministic strict-30
    # view here; validate_snapshot_graph subsequently proves every graph node
    # is present in this view.
    reply_offsets = pd.to_numeric(replies["offset_sec"], errors="coerce")
    if reply_offsets.isna().any() or (reply_offsets < 0).any():
        raise ValueError("protected reply table has invalid offsets")
    replies = replies.loc[reply_offsets <= CUTOFF_SECONDS].copy()
    return replies, nodes


def validate_snapshot_graph(graph: Any, replies: pd.DataFrame, nodes: pd.DataFrame) -> None:
    """Check graph nodes/edges against protected cutoff-safe raw records."""
    thread_id = str(graph.thread_id)
    node_ids = [str(item) for item in graph.node_ids]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError(f"{thread_id}: duplicate graph node identifiers")
    graph_replies = replies.loc[(replies.thread_id.astype(str) == thread_id) &
                                (replies.tweet_id.astype(str).isin(node_ids))]
    graph_nodes = nodes.loc[(nodes.thread_id.astype(str) == thread_id) &
                            (nodes.tweet_id.astype(str).isin(node_ids))]
    if len(graph_replies) != len(node_ids) or len(graph_nodes) != len(node_ids):
        raise ValueError(f"{thread_id}: graph nodes do not exactly match protected raw rows")
    if (pd.to_numeric(graph_replies.offset_sec, errors="coerce") > CUTOFF_SECONDS).any():
        raise ValueError(f"{thread_id}: graph includes post-cutoff node")
    edge_index = np.asarray(graph.edge_index.cpu() if hasattr(graph.edge_index, "cpu") else graph.edge_index)
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= len(node_ids)):
        raise ValueError(f"{thread_id}: graph edge index is out of bounds")


def scalar_features(graph: Any, replies: pd.DataFrame, nodes: pd.DataFrame) -> dict[str, float]:
    """Build observable feature groups for one already-validated snapshot graph."""
    validate_snapshot_graph(graph, replies, nodes)
    thread_id = str(graph.thread_id)
    order = {node_id: index for index, node_id in enumerate(str(item) for item in graph.node_ids)}
    raw = replies.loc[(replies.thread_id.astype(str) == thread_id) &
                      (replies.tweet_id.astype(str).isin(order))].copy()
    raw["_order"] = raw.tweet_id.astype(str).map(order)
    raw.sort_values("_order", inplace=True)
    raw_nodes = nodes.loc[(nodes.thread_id.astype(str) == thread_id) &
                          (nodes.tweet_id.astype(str).isin(order))].copy()
    raw_nodes["_order"] = raw_nodes.tweet_id.astype(str).map(order)
    raw_nodes.sort_values("_order", inplace=True)
    if len(raw) != len(order) or len(raw_nodes) != len(order):
        raise AssertionError(f"{thread_id}: unexpected aligned row count")

    source = raw.loc[raw.is_source.astype(bool)]
    if len(source) != 1:
        raise ValueError(f"{thread_id}: expected exactly one source node")
    replies_only = raw.loc[~raw.is_source.astype(bool)]
    offsets = pd.to_numeric(raw.offset_sec, errors="coerce").fillna(0.0).to_numpy(float)
    reply_offsets = pd.to_numeric(replies_only.offset_sec, errors="coerce").dropna().to_numpy(float)
    depths = pd.to_numeric(raw.depth, errors="coerce").fillna(0.0).to_numpy(float)
    bins_1m = np.bincount(np.clip((offsets // 60).astype(int), 0, 30), minlength=31)
    bins_5m = np.bincount(np.clip((offsets // 300).astype(int), 0, 6), minlength=7)
    edge_index = np.asarray(graph.edge_index.cpu() if hasattr(graph.edge_index, "cpu") else graph.edge_index)
    outdegree = np.bincount(edge_index[0], minlength=len(raw)) if edge_index.size else np.zeros(len(raw))
    indegree = np.bincount(edge_index[1], minlength=len(raw)) if edge_index.size else np.zeros(len(raw))
    features: dict[str, float] = {
        "thread_id": thread_id,
        "event_id": str(raw.event_id.iloc[0]),
        "activity_observed_nodes": float(len(raw)),
        "activity_observed_replies": float(len(replies_only)),
        "activity_text_available_fraction": float(raw.is_text_available.astype(float).mean()),
        "temporal_active_minute_count": float((bins_1m > 0).sum()),
        "temporal_peak_minute_count": float(bins_1m.max()),
        "temporal_peak_minute_index": float(bins_1m.argmax()),
        "temporal_hist_1m_entropy": _entropy(bins_1m),
        "temporal_hist_1m_gini": _gini(bins_1m),
        "temporal_hist_5m_entropy": _entropy(bins_5m),
        "temporal_hist_5m_gini": _gini(bins_5m),
        "temporal_first10_fraction": float((offsets <= 600).mean()),
        "temporal_mid10_fraction": float(((offsets > 600) & (offsets <= 1200)).mean()),
        "temporal_last10_fraction": float((offsets > 1200).mean()),
        "topology_max_depth": float(depths.max()),
        "topology_leaf_fraction": float((outdegree == 0).mean()),
        "topology_internal_fraction": float((outdegree > 0).mean()),
        "topology_root_children": float(outdegree[0]) if len(outdegree) else 0.0,
        "topology_width_to_depth": float(len(raw) / max(depths.max(), 1.0)),
        "topology_depth_entropy": _entropy(np.bincount(depths.astype(int))),
        "topology_outdegree_entropy": _entropy(outdegree),
        "topology_outdegree_gini": _gini(outdegree),
        "topology_outdegree_hhi": float(np.square(outdegree / max(outdegree.sum(), 1.0)).sum()),
    }
    features.update(_moments(offsets, "temporal_offset"))
    features.update(_moments(np.diff(np.sort(reply_offsets)), "temporal_reply_interarrival"))
    features.update(_moments(pd.to_numeric(replies_only.reply_latency_sec, errors="coerce"), "temporal_reply_latency"))
    features.update(_moments(depths, "topology_depth"))
    features.update(_moments(outdegree, "topology_outdegree"))
    features.update(_moments(indegree, "topology_indegree"))
    features.update(_text_stats(source.text.tolist(), "source_text"))
    features.update(_text_stats(replies_only.text.tolist(), "reply_text"))
    features.update(_text_stats(raw.text.tolist(), "snapshot_text"))
    for column in ("vader_compound", "vader_pos", "vader_neg", "vader_neu"):
        features.update(_moments(pd.to_numeric(raw_nodes[column], errors="coerce"), f"sentiment_{column[6:]}"))
    features.update(_moments(pd.to_numeric(raw_nodes.account_age_days_log, errors="coerce"), "account_age_log_days"))
    semantic = np.asarray(graph.semantic_features.cpu() if hasattr(graph.semantic_features, "cpu") else graph.semantic_features, dtype=float)
    features.update(_moments(semantic.ravel(), "semantic_graph"))
    return features


def feature_group(column: str) -> str | None:
    if column.startswith("account_age_"):
        return "account_age"
    if column.startswith("activity_"):
        return "activity_level"
    if column.startswith("semantic_"):
        return "semantic_summary"
    if column.startswith("sentiment_"):
        return "sentiment"
    if column.startswith("temporal_"):
        return "temporal_dynamics"
    if column.startswith(("source_text_", "reply_text_", "snapshot_text_")):
        return "text_surface"
    if column.startswith("topology_"):
        return "topology"
    return None
