"""Canonical configuration for the Graph7 author-aware fixed-30 RF."""
from __future__ import annotations

from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODEL_ROOT.parents[1]
DATASET_DIR = MODEL_ROOT / "artifacts" / "20260830_133000_pheme_author_aware_30min_schema_v2"
EXPERIMENTS_DIR = MODEL_ROOT / "experiments"

SEED = 42
CUTOFF_MINUTES = 30
TOP_K = 10
ELIGIBLE_EVENT_MIN_CANDIDATES = 100
FULL_PARAMETERS: dict[str, object] = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 3,
    "max_features": 0.8,
    "n_jobs": 1,
    "random_state": SEED,
}
SMOKE_PARAMETERS: dict[str, object] = {**FULL_PARAMETERS, "n_estimators": 5}

