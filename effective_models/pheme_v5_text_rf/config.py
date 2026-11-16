"""Canonical configuration for the PHEME v5 fixed-30 text RF."""
from __future__ import annotations

from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent
ASSET_DIR = MODEL_ROOT / "artifacts" / "pheme_v5_strict30"
DATASET_PATH = ASSET_DIR / "pheme_graphs_roberta_30min_replyv4_preventableimpact_semantic.pt"
RAW_NODE_FEATURES_PATH = ASSET_DIR / "pheme_node_features_roberta_replyv4.csv"
EXPERIMENTS_DIR = MODEL_ROOT / "experiments"

SEED = 42
CUTOFF_SECONDS = 1800
TOP_K = 10
ELIGIBLE_EVENT_MIN_CANDIDATES = 100
PCA_COMPONENTS = 64
EMBEDDING_START = 13
FEATURE_SETS = ("baseline", "text_augmented")
FULL_PARAMETERS: dict[str, object] = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 3,
    "max_features": 0.8,
    "n_jobs": 1,
    "random_state": SEED,
}
SMOKE_PARAMETERS: dict[str, object] = {**FULL_PARAMETERS, "n_estimators": 5}
