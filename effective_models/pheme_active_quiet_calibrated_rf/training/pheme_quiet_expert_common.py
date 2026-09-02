"""Fold-safe recency-gated two-expert PHEME utility ranker.

The gate is unsupervised: the train-fold 60th percentile of the existing
``log1p_seconds_since_last_activity`` feature.  Both experts predict the same
unchanged utility target, so their predictions remain on the same log-impact
scale and can be ranked together.
"""
from __future__ import annotations

import numpy as np

from pheme_account_age_v5_common import (
    ACCOUNT_AGE_FEATURE,
    CUTOFF_SECONDS,
    FEATURE_SETS,
    MODEL_PARAMS,
    TEMPORAL_FEATURE_NAMES,
    feature_names,
    fit_predict,
)


MODEL_FEATURE_SET = "text_augmented_no_profile_plus_account_age"
GATE_FEATURE = "log1p_seconds_since_last_activity"
QUIET_QUANTILE = 0.60
GATE_INDEX = TEMPORAL_FEATURE_NAMES.index(GATE_FEATURE)


def recency_values(graphs: list) -> np.ndarray:
    values = np.asarray([
        float(graph.temporal_features.detach().cpu().numpy().reshape(-1)[GATE_INDEX])
        for graph in graphs
    ], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite recency gate feature")
    return values


def gated_predict(train: list, test: list, depth_nodes, account_age_nodes, node_offsets, model_params: dict | None = None) -> tuple[np.ndarray, dict, list[dict]]:
    """Fit active/quiet experts from train events only and score one test event."""
    params = MODEL_PARAMS if model_params is None else model_params
    train_recency = recency_values(train)
    test_recency = recency_values(test)
    threshold = float(np.quantile(train_recency, QUIET_QUANTILE))
    train_quiet = train_recency >= threshold
    test_quiet = test_recency >= threshold
    if train_quiet.sum() == 0 or (~train_quiet).sum() == 0:
        raise ValueError("Recency gate created an empty train expert")
    predictions = np.empty(len(test), dtype=np.float32)
    importance_rows: list[dict] = []
    for expert, train_mask, test_mask in (
        ("quiet", train_quiet, test_quiet),
        ("active", ~train_quiet, ~test_quiet),
    ):
        expert_train = [graph for graph, include in zip(train, train_mask) if include]
        expert_test = [graph for graph, include in zip(test, test_mask) if include]
        if not expert_test:
            continue
        if len(expert_train) <= 65:
            raise ValueError(f"{expert} expert has too few train threads for 64-component text PCA: {len(expert_train)}")
        prediction, importance = fit_predict(expert_train, expert_test, depth_nodes, account_age_nodes, node_offsets, MODEL_FEATURE_SET, params)
        predictions[test_mask] = prediction
        importance_rows.extend({"expert": expert, "feature": name, "importance": float(value)} for name, value in zip(feature_names(MODEL_FEATURE_SET), importance))
    if not np.isfinite(predictions).all():
        raise ValueError("Non-finite two-expert prediction")
    gate = {
        "feature": GATE_FEATURE,
        "train_quantile": QUIET_QUANTILE,
        "threshold": threshold,
        "train_quiet_threads": int(train_quiet.sum()),
        "train_active_threads": int((~train_quiet).sum()),
        "test_quiet_threads": int(test_quiet.sum()),
        "test_active_threads": int((~test_quiet).sum()),
    }
    return predictions, gate, importance_rows


def safety_statement() -> str:
    return (
        f"The unsupervised gate is the train-fold {QUIET_QUANTILE:.0%} quantile of {GATE_FEATURE}; held-out event outcomes do not set it. "
        "Both experts use only <=30-minute snapshot nodes, fixed account age, safe temporal/depth features, and snapshot text PCA. "
        "Retweet/favorite counts and all mutable profile fields are excluded. "
        "Each expert fits PCA, StandardScaler, and RandomForest only on its own train-event subset."
    )
