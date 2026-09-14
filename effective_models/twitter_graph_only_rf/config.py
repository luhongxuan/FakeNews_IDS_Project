"""Frozen configuration for the Twitter15/16 graph-only RF."""
from __future__ import annotations

from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = MODEL_ROOT / "artifacts"
DATA_ROOT = (
    ARTIFACT_ROOT
    / "20260830_024514_251_twitter15_16_30min_time_respecting_preventable_impact"
)
SPLIT_ROOT = (
    ARTIFACT_ROOT
    / "20260830_024515_294_twitter15_16_source_level_split_v1"
)
EXPERIMENT_ROOT = MODEL_ROOT / "experiments"

CUTOFF_SECONDS = 1800
TOP_K = 10
SEED = 42
ELIGIBLE_SPLITS = ("train", "validation", "test")

EARLY_GRAPH_FEATURES = (
    "observed_node_count",
    "observed_edge_count",
    "log1p_observed_nodes",
    "log1p_count_0_10m",
    "log1p_count_10_20m",
    "log1p_count_20_30m",
    "log1p_count_recent_5m",
    "log1p_count_recent_10m",
    "log1p_seconds_since_last_activity",
    "log1p_median_interarrival_sec",
    "log1p_std_interarrival_sec",
    "late_activity_frac",
    "observed_leaf_fraction",
    "mean_observed_children",
    "max_observed_children",
)

FULL_PARAMETERS = {
    "rf_depth_6": {
        "n_estimators": 300,
        "max_depth": 6,
        "min_samples_leaf": 3,
        "max_features": 0.8,
        "n_jobs": 1,
        "random_state": SEED,
    },
    "rf_depth_10": {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_leaf": 3,
        "max_features": 0.8,
        "n_jobs": 1,
        "random_state": SEED,
    },
    "rf_unrestricted": {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 3,
        "max_features": 0.8,
        "n_jobs": 1,
        "random_state": SEED,
    },
}

SMOKE_PARAMETERS = {
    name: {**params, "n_estimators": 5, "n_jobs": 1}
    for name, params in FULL_PARAMETERS.items()
}
