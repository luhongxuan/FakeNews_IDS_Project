"""Canonical configuration for the balanced cumulative multi-checkpoint RF."""
from __future__ import annotations

from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent
REFERENCE_BUNDLE = MODEL_ROOT / "reference_result" / "20260903_160025_balanced_cumulative_sequential_policy"
DATASET_PATH = REFERENCE_BUNDLE / "inputs" / "pheme_multicheckpoint_features.csv"
SCHEMA_PATH = REFERENCE_BUNDLE / "inputs" / "feature_schema.json"
EXPERIMENTS_DIR = MODEL_ROOT / "experiments"

SEED = 42
CHECKPOINTS = (600, 1200, 1800, 2400, 3000, 3600)
BALANCED_QUOTAS = (9, 9, 8, 8, 8, 8)
TOTAL_BUDGET = 50
ELIGIBLE_MIN_THREADS = 100
FULL_PARAMETERS: dict[str, object] = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 3,
    "max_features": 0.8,
    "n_jobs": 1,
    "random_state": SEED,
}
SMOKE_PARAMETERS: dict[str, object] = {**FULL_PARAMETERS, "n_estimators": 5}

