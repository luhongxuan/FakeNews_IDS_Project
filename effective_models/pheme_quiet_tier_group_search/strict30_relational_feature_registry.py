"""Cutoff-safe source/reply relational features for strict-30 quiet studies."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strict30_protected_feature_registry import _moments, validate_snapshot_graph


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def relational_features(graph: object, replies: pd.DataFrame, nodes: pd.DataFrame) -> dict[str, float]:
    """Return only relations among nodes already present in one strict snapshot."""
    validate_snapshot_graph(graph, replies, nodes)
    thread_id = str(graph.thread_id)
    order = {str(value): index for index, value in enumerate(graph.node_ids)}
    raw = replies.loc[(replies.thread_id.astype(str) == thread_id) & replies.tweet_id.astype(str).isin(order)].copy()
    raw["_order"] = raw.tweet_id.astype(str).map(order); raw.sort_values("_order", inplace=True)
    raw_nodes = nodes.loc[(nodes.thread_id.astype(str) == thread_id) & nodes.tweet_id.astype(str).isin(order)].copy()
    raw_nodes["_order"] = raw_nodes.tweet_id.astype(str).map(order); raw_nodes.sort_values("_order", inplace=True)
    source_index = np.flatnonzero(raw.is_source.astype(bool).to_numpy())
    if len(source_index) != 1 or graph.x.ndim != 2 or graph.x.shape[1] <= 13:
        raise ValueError(f"{thread_id}: invalid source/reply relation inputs")
    source_index = int(source_index[0]); reply_index = np.flatnonzero(~raw.is_source.astype(bool).to_numpy())
    result: dict[str, float] = {"thread_id": thread_id}
    if not len(reply_index):
        return {**result, **{name: 0.0 for name in _RELATIONAL_NAMES}}
    x = graph.x.detach().cpu().numpy() if hasattr(graph.x, "detach") else np.asarray(graph.x)
    vectors = x[:, 13:].astype(np.float32)
    source_vector, reply_vectors = vectors[source_index], vectors[reply_index]
    similarities = np.asarray([_cosine(source_vector, vector) for vector in reply_vectors], dtype=float)
    ranked_replies = raw.iloc[reply_index].assign(_offset=pd.to_numeric(raw.iloc[reply_index].offset_sec, errors="coerce").fillna(0.0)).sort_values(["_offset", "tweet_id"], kind="stable")
    first_index = int(ranked_replies.iloc[0]._order); last_index = int(ranked_replies.iloc[-1]._order)
    source_node = raw_nodes.iloc[source_index]; reply_nodes = raw_nodes.iloc[reply_index]
    source_age = float(pd.to_numeric(pd.Series([source_node.account_age_days_log]), errors="coerce").fillna(0.0).iloc[0])
    reply_age = pd.to_numeric(reply_nodes.account_age_days_log, errors="coerce").fillna(0.0).to_numpy(float)
    source_sentiment = float(pd.to_numeric(pd.Series([source_node.vader_compound]), errors="coerce").fillna(0.0).iloc[0])
    reply_sentiment = pd.to_numeric(reply_nodes.vader_compound, errors="coerce").fillna(0.0).to_numpy(float)
    source_text = str(raw.iloc[source_index].text or "")
    reply_text = raw.iloc[reply_index].text.fillna("").astype(str)
    offsets = ranked_replies._offset.to_numpy(float)
    result.update({
        **_moments(similarities, "rel_source_reply_cosine"),
        "rel_source_reply_centroid_cosine": _cosine(source_vector, reply_vectors.mean(axis=0)),
        "rel_source_first_reply_cosine": _cosine(source_vector, vectors[first_index]),
        "rel_source_last_reply_cosine": _cosine(source_vector, vectors[last_index]),
        "rel_first_reply_offset_sec": float(offsets[0]),
        "rel_last_reply_offset_sec": float(offsets[-1]),
        "rel_reply_active_span_sec": float(offsets[-1] - offsets[0]),
        "rel_source_reply_sentiment_delta_mean": float(reply_sentiment.mean() - source_sentiment),
        "rel_source_reply_sentiment_delta_abs_mean": float(np.abs(reply_sentiment - source_sentiment).mean()),
        "rel_source_reply_sentiment_delta_first": float(reply_nodes.loc[reply_nodes._order.eq(first_index), "vader_compound"].iloc[0] - source_sentiment),
        "rel_source_reply_age_delta_mean": float(reply_age.mean() - source_age),
        "rel_source_reply_age_delta_abs_mean": float(np.abs(reply_age - source_age).mean()),
        "rel_reply_younger_than_source_fraction": float((reply_age < source_age).mean()),
        "rel_source_reply_char_delta_mean": float(len(source_text) - reply_text.str.len().mean()),
        "rel_source_reply_question_delta": float(source_text.count("?") - reply_text.str.count(r"\?").mean()),
        "rel_source_reply_url_delta": float((source_text.lower().count("http")) - reply_text.str.lower().str.count("http").mean()),
    })
    return result


_RELATIONAL_NAMES = (
    "rel_source_reply_cosine_mean", "rel_source_reply_cosine_std", "rel_source_reply_cosine_min", "rel_source_reply_cosine_max", "rel_source_reply_cosine_median",
    "rel_source_reply_centroid_cosine", "rel_source_first_reply_cosine", "rel_source_last_reply_cosine", "rel_first_reply_offset_sec", "rel_last_reply_offset_sec", "rel_reply_active_span_sec",
    "rel_source_reply_sentiment_delta_mean", "rel_source_reply_sentiment_delta_abs_mean", "rel_source_reply_sentiment_delta_first", "rel_source_reply_age_delta_mean", "rel_source_reply_age_delta_abs_mean", "rel_reply_younger_than_source_fraction",
    "rel_source_reply_char_delta_mean", "rel_source_reply_question_delta", "rel_source_reply_url_delta",
)
