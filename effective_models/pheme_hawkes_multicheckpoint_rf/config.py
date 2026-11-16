from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
EXPERIMENTS = ROOT / "experiments"
FEATURE_TABLE = ARTIFACTS / "pheme_multicheckpoint_features.csv"
FEATURE_SCHEMA = ARTIFACTS / "feature_schema.json"
REPLY_TABLE = ARTIFACTS / "pheme_reply_level.csv"

CHECKPOINTS_SECONDS = (600, 1200, 1800, 2400, 3000, 3600)
BALANCED_QUOTAS = (9, 9, 8, 8, 8, 8)
SEED = 42
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 3,
    "max_features": 0.8,
    "n_jobs": 1,
    "random_state": SEED,
}
DECAYS_SECONDS = (300.0, 600.0, 1200.0)

