"""Fold-safe PHEME v5 text-RF helpers with one immutable account-age ablation.

This module deliberately adds only the mean of the precomputed account age on
nodes already present in the frozen 30-minute snapshot.  Account creation time
is immutable; mutable profile and engagement values are not added here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import torch

from effective_models.pheme_active_quiet_calibrated_rf import config
from effective_models.pheme_active_quiet_calibrated_rf.features.text_feature_common import text_matrix


ACCOUNT_AGE_COLUMN = "account_age_days_log"
ACCOUNT_AGE_FEATURE = "observed_account_age_days_log_mean"
TEMPORAL_FEATURE_NAMES = [
    "log1p_count_0_10m", "log1p_count_10_20m", "log1p_count_20_30m",
    "log1p_count_recent_5m", "log1p_count_recent_10m",
    "log1p_seconds_since_last_activity", "log1p_median_interarrival_sec",
    "log1p_std_interarrival_sec", "observed_leaf_fraction",
    "mean_observed_children", "max_observed_children",
]
SAFE_BASE_FEATURE_NAMES = TEMPORAL_FEATURE_NAMES + [
    "log1p_observed_nodes", "late_activity_frac", "mean_depth", "max_depth",
]
FEATURE_SETS = ("text_augmented_no_profile", "text_augmented_no_profile_plus_account_age")
ELIGIBLE_MIN_THREADS = config.ELIGIBLE_MIN_THREADS


def load_graphs() -> tuple[list, list[str], list[str]]:
    """Load the existing protected rumour-only v5 artifact without modifying it."""
    graphs = torch.load(config.DATASET_PATH, weights_only=False)
    events = sorted({str(graph.event_id) for graph in graphs})
    eligible = [event for event in events if sum(str(graph.event_id) == event for graph in graphs) >= ELIGIBLE_MIN_THREADS]
    if len(graphs) != 2402 or len(events) != 9:
        raise ValueError(f"Unexpected protected v5 artifact shape: graphs={len(graphs)}, events={len(events)}")
    return graphs, events, eligible


def load_safe_nodes(required_keys: set[tuple[str, str]]) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Read depth and fixed account age only; no mutable profile fields are read."""
    required_index = pd.MultiIndex.from_tuples(required_keys, names=["thread_id", "tweet_id"])
    depth_nodes, age_nodes = {}, {}
    for frame in pd.read_csv(
        config.RAW_NODES_PATH,
        usecols=["thread_id", "tweet_id", "event_id", "depth", ACCOUNT_AGE_COLUMN],
        dtype={"thread_id": str, "tweet_id": str, "event_id": str},
        chunksize=250_000,
        low_memory=False,
    ):
        index = pd.MultiIndex.from_frame(frame[["thread_id", "tweet_id"]])
        selected = frame.loc[index.isin(required_index)]
        if selected[["depth", ACCOUNT_AGE_COLUMN]].isna().any().any() or not np.isfinite(selected[["depth", ACCOUNT_AGE_COLUMN]].to_numpy(dtype=float)).all():
            raise ValueError("Safe-node source has missing or non-finite depth/account-age values")
        for row in selected.itertuples(index=False):
            key = (row.thread_id, row.tweet_id)
            if key in depth_nodes:
                raise ValueError(f"Duplicate safe-node row: {key}")
            depth_nodes[key] = float(row.depth)
            age_nodes[key] = float(getattr(row, ACCOUNT_AGE_COLUMN))
    if required_keys - set(depth_nodes):
        raise KeyError(f"Missing {len(required_keys - set(depth_nodes))} required safe-node rows")
    return depth_nodes, age_nodes


def load_node_offsets(required_keys: set[tuple[str, str]]) -> dict[tuple[str, str], float]:
    """Load offsets solely to verify that every model node is observable at 30 min."""
    frame = pd.read_csv(
        config.REPLIES_PATH,
        usecols=["thread_id", "tweet_id", "offset_sec"],
        dtype={"thread_id": str, "tweet_id": str},
        low_memory=False,
    )
    frame = frame.loc[pd.MultiIndex.from_frame(frame[["thread_id", "tweet_id"]]).isin(
        pd.MultiIndex.from_tuples(required_keys, names=["thread_id", "tweet_id"])
    )]
    if frame.duplicated(["thread_id", "tweet_id"]).any():
        raise ValueError("Reply offsets are not unique by (thread_id, tweet_id)")
    if frame.offset_sec.isna().any() or not np.isfinite(frame.offset_sec.to_numpy(dtype=float)).all():
        raise ValueError("Reply offsets contain missing or non-finite values")
    offsets = {(row.thread_id, row.tweet_id): float(row.offset_sec) for row in frame.itertuples(index=False)}
    if required_keys - set(offsets):
        raise KeyError(f"Missing {len(required_keys - set(offsets))} required observation offsets")
    return offsets


def safe_base_features(graph, depth_nodes: dict[tuple[str, str], float]) -> np.ndarray:
    """The original v5 temporal/depth summaries with the follower feature removed."""
    depths = []
    for tweet_id in graph.node_ids:
        key = (str(graph.thread_id), str(tweet_id))
        try:
            depths.append(depth_nodes[key])
        except KeyError as error:
            raise KeyError(f"Missing depth for observable node {key}") from error
    temporal = graph.temporal_features.detach().cpu().numpy().reshape(-1)
    if temporal.size != len(TEMPORAL_FEATURE_NAMES):
        raise ValueError(f"Unexpected temporal feature length for {graph.thread_id}")
    depth_values = np.asarray(depths, dtype=np.float32)
    aux = np.asarray([
        np.log1p(graph.num_nodes), float(graph.late_activity_frac.item()),
        float(depth_values.mean()), float(depth_values.max()),
    ], dtype=np.float32)
    return np.concatenate([temporal, aux]).astype(np.float32)


def observed_account_age_mean(graph, account_age_nodes: dict[tuple[str, str], float]) -> float:
    values = []
    for tweet_id in graph.node_ids:
        key = (str(graph.thread_id), str(tweet_id))
        try:
            values.append(account_age_nodes[key])
        except KeyError as error:
            raise KeyError(f"Missing account age for observable node {key}") from error
    value = float(np.mean(values))
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {ACCOUNT_AGE_FEATURE} for {graph.thread_id}")
    return value


def validate_fold(train: list, test: list, depth_nodes, account_age_nodes, node_offsets) -> None:
    train_events = {str(graph.event_id) for graph in train}
    test_events = {str(graph.event_id) for graph in test}
    if train_events & test_events:
        raise ValueError(f"Event overlap between train and test: {train_events & test_events}")
    for graph in (*train, *test):
        base = safe_base_features(graph, depth_nodes)
        age = observed_account_age_mean(graph, account_age_nodes)
        if not np.isfinite(base).all() or not np.isfinite(age):
            raise ValueError(f"Non-finite feature for {graph.thread_id}")
        if graph.edge_index.numel() and (int(graph.edge_index.min()) < 0 or int(graph.edge_index.max()) >= graph.num_nodes):
            raise ValueError(f"Snapshot edge index references a non-observable node for {graph.thread_id}")
        for tweet_id in graph.node_ids:
            key = (str(graph.thread_id), str(tweet_id))
            try:
                offset = node_offsets[key]
            except KeyError as error:
                raise KeyError(f"Missing observation offset for snapshot node {key}") from error
            if offset > config.CUTOFF_SECONDS + 1e-9:
                raise ValueError(f"Future node used in snapshot {key}: offset={offset}")


def matrices(train: list, test: list, depth_nodes, account_age_nodes, feature_set: str) -> tuple[np.ndarray, np.ndarray]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    base_train = np.vstack([safe_base_features(graph, depth_nodes) for graph in train])
    base_test = np.vstack([safe_base_features(graph, depth_nodes) for graph in test])
    text_train, text_test = text_matrix(train), text_matrix(test)
    text_scaler = StandardScaler().fit(text_train)
    pca = PCA(n_components=config.PCA_COMPONENTS, random_state=config.SEED).fit(text_scaler.transform(text_train))
    x_train = np.hstack([base_train, pca.transform(text_scaler.transform(text_train))])
    x_test = np.hstack([base_test, pca.transform(text_scaler.transform(text_test))])
    if feature_set == "text_augmented_no_profile_plus_account_age":
        age_train = np.asarray([observed_account_age_mean(graph, account_age_nodes) for graph in train], dtype=np.float32).reshape(-1, 1)
        age_test = np.asarray([observed_account_age_mean(graph, account_age_nodes) for graph in test], dtype=np.float32).reshape(-1, 1)
        x_train = np.hstack([x_train, age_train])
        x_test = np.hstack([x_test, age_test])
    return x_train.astype(np.float32), x_test.astype(np.float32)


def fit_predict(train: list, test: list, depth_nodes, account_age_nodes, node_offsets, feature_set: str, model_params: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fit only on train events and return test predictions plus RF importances."""
    validate_fold(train, test, depth_nodes, account_age_nodes, node_offsets)
    x_train, x_test = matrices(train, test, depth_nodes, account_age_nodes, feature_set)
    y_train = np.asarray([float(graph.preventable_y.item()) for graph in train], dtype=np.float32)
    scaler = StandardScaler().fit(x_train)
    params = dict(config.FULL_PARAMETERS if model_params is None else model_params)
    model = RandomForestRegressor(**params).fit(scaler.transform(x_train), y_train)
    prediction = model.predict(scaler.transform(x_test))
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite prediction")
    return prediction, model.feature_importances_


def feature_names(feature_set: str) -> list[str]:
    names = list(SAFE_BASE_FEATURE_NAMES) + [f"text_pca_{index}" for index in range(config.PCA_COMPONENTS)]
    return names + ([ACCOUNT_AGE_FEATURE] if feature_set == "text_augmented_no_profile_plus_account_age" else [])


def safety_statement() -> str:
    return (
        "Every graph.node_id is rechecked against raw offset_sec <= 1800 and every edge index is checked to reference snapshot nodes. "
        "The only ablated field is the mean account_age_days_log for graph.node_ids already in each frozen 30-minute snapshot. "
        "The baseline uses temporal/depth summaries and text PCA only; retweet/favorite counts, followers/friends/statuses, verified, and profile fields are neither read nor used. "
        "PCA, StandardScaler, and RandomForest fit only on each fold's train events."
    )
