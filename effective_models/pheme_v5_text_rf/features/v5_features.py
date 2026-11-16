"""Cutoff-safe structural and text features for the canonical v5 model."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch

from effective_models.pheme_v5_text_rf import config

RAW_COLUMNS = ["thread_id", "tweet_id", "event_id", "depth", "followers_log"]
TEMPORAL_FEATURE_NAMES = [
    "log1p_count_0_10m", "log1p_count_10_20m", "log1p_count_20_30m",
    "log1p_count_recent_5m", "log1p_count_recent_10m",
    "log1p_seconds_since_last_activity", "log1p_median_interarrival_sec",
    "log1p_std_interarrival_sec", "observed_leaf_fraction",
    "mean_observed_children", "max_observed_children",
]
AUX_FEATURE_NAMES = [
    "log1p_observed_nodes", "late_activity_frac", "max_followers_log",
    "mean_depth", "max_depth",
]
BASELINE_FEATURE_NAMES = TEMPORAL_FEATURE_NAMES + AUX_FEATURE_NAMES


def load_graphs() -> list[object]:
    if not config.DATASET_PATH.is_file():
        raise FileNotFoundError(f"Missing protected graph artifact: {config.DATASET_PATH}")
    graphs = torch.load(config.DATASET_PATH, weights_only=False)
    identities = [(str(graph.event_id), str(graph.thread_id)) for graph in graphs]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate (event_id, thread_id) graph identity")
    for graph in graphs:
        if int(graph.temporal_features.numel()) != len(TEMPORAL_FEATURE_NAMES):
            raise ValueError(f"Unexpected temporal feature length for {graph.thread_id}")
        if len(graph.node_ids) != int(graph.num_nodes):
            raise ValueError(f"node_ids/num_nodes mismatch for {graph.thread_id}")
        source_count = int(torch.count_nonzero(graph.is_source_mask).item())
        if source_count != 1:
            raise ValueError(f"Expected one source node for {graph.thread_id}")
        impact = float(graph.preventable_impact.item())
        target = float(graph.preventable_y.item())
        if not np.isclose(np.log1p(impact), target, rtol=1e-5, atol=1e-6):
            raise ValueError(f"Target mismatch for {graph.thread_id}")
    return graphs


def observable_node_keys(graphs: Iterable[object]) -> set[tuple[str, str]]:
    return {
        (str(graph.thread_id), str(tweet_id))
        for graph in graphs
        for tweet_id in graph.node_ids
    }


def load_raw_node_features(
    required_keys: set[tuple[str, str]],
) -> dict[tuple[str, str], tuple[str, float, float]]:
    """Scan protected raw rows and retain only nodes observable in selected snapshots."""
    if not config.RAW_NODE_FEATURES_PATH.is_file():
        raise FileNotFoundError(f"Missing protected raw-node artifact: {config.RAW_NODE_FEATURES_PATH}")
    required_index = pd.MultiIndex.from_tuples(required_keys, names=["thread_id", "tweet_id"])
    found: dict[tuple[str, str], tuple[str, float, float]] = {}
    for chunk in pd.read_csv(
        config.RAW_NODE_FEATURES_PATH,
        usecols=RAW_COLUMNS,
        dtype={"thread_id": str, "tweet_id": str, "event_id": str},
        chunksize=250_000,
        low_memory=False,
    ):
        index = pd.MultiIndex.from_frame(chunk[["thread_id", "tweet_id"]])
        selected = chunk.loc[index.isin(required_index)]
        if selected[["depth", "followers_log"]].isna().any().any():
            raise ValueError("Raw node features contain missing depth/followers values")
        for row in selected.itertuples(index=False):
            key = (row.thread_id, row.tweet_id)
            if key in found:
                raise ValueError(f"Duplicate raw feature row for {key}")
            found[key] = (row.event_id, float(row.depth), float(row.followers_log))
    missing = required_keys - set(found)
    if missing:
        raise KeyError(f"Missing {len(missing)} observable raw-node rows; example={next(iter(missing))}")
    return found


def structural_matrix(
    graphs: list[object],
    raw_nodes: dict[tuple[str, str], tuple[str, float, float]],
) -> np.ndarray:
    rows = []
    for graph in graphs:
        node_rows = []
        for tweet_id in graph.node_ids:
            key = (str(graph.thread_id), str(tweet_id))
            event_id, depth, followers_log = raw_nodes[key]
            if event_id != graph.event_id:
                raise ValueError(f"Event mismatch for {key}: {event_id} != {graph.event_id}")
            node_rows.append((depth, followers_log))
        values = np.asarray(node_rows, dtype=np.float32)
        temporal = graph.temporal_features.detach().cpu().numpy().reshape(-1)
        aux = np.asarray(
            [
                np.log1p(graph.num_nodes),
                float(graph.late_activity_frac.item()),
                float(values[:, 1].max()),
                float(values[:, 0].mean()),
                float(values[:, 0].max()),
            ],
            dtype=np.float32,
        )
        vector = np.concatenate([temporal, aux]).astype(np.float32)
        if not np.isfinite(vector).all():
            raise ValueError(f"Non-finite structural feature for {graph.thread_id}")
        rows.append(vector)
    return np.vstack(rows)


def text_matrix(graphs: list[object]) -> np.ndarray:
    rows = []
    for graph in graphs:
        source_indices = torch.nonzero(graph.is_source_mask, as_tuple=False).flatten()
        embedding = graph.x[:, config.EMBEDDING_START:].detach().cpu().numpy().astype(np.float32)
        if source_indices.numel() != 1 or embedding.shape[1] != 768:
            raise ValueError(f"Invalid source/text embedding shape for {graph.thread_id}")
        if not np.isfinite(embedding).all():
            raise ValueError(f"Non-finite text embedding for {graph.thread_id}")
        source_index = int(source_indices.item())
        source = embedding[source_index]
        replies = np.delete(embedding, source_index, axis=0)
        centroid = replies.mean(axis=0) if len(replies) else np.zeros_like(source)
        rows.append(np.concatenate([source, centroid]))
    return np.vstack(rows)
