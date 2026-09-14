"""Canonical configuration for the calibrated Active/Quiet fixed-30 RF."""
from __future__ import annotations

from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent
ASSET_DIR = MODEL_ROOT / "artifacts" / "pheme_v5_strict30"
DATASET_PATH = ASSET_DIR / "pheme_graphs_roberta_30min_replyv4_preventableimpact_semantic.pt"
RAW_NODES_PATH = ASSET_DIR / "pheme_node_features_roberta_replyv4.csv"
REPLIES_PATH = ASSET_DIR / "pheme_reply_level_v4.csv"
EXPERIMENTS_DIR = MODEL_ROOT / "experiments"

SEED = 42
CUTOFF_SECONDS = 1800.0
TOP_K = 10
ELIGIBLE_MIN_THREADS = 100
PCA_COMPONENTS = 64
EMBEDDING_START = 13
QUIET_QUANTILE = 0.60
CALIBRATION_ALPHA = 1.0
BUDGETS = (1, 3, 5, 10, 20, 50, 100)
FULL_PARAMETERS: dict[str, object] = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 3,
    "max_features": 0.8,
    "n_jobs": 1,
    "random_state": SEED,
}
SMOKE_PARAMETERS: dict[str, object] = {**FULL_PARAMETERS, "n_estimators": 5}
