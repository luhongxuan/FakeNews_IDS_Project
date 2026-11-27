"""Fixed candidate methods and model fitting for a single platform run."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from triage_pipeline.core import FEATURES

SEEDS = (20261110, 20261111, 20261112)
PARAMS = {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20}
METHODS = ("M", "U", "R1", "R2", "random")


def fit_models(rows: list[dict], progress=None) -> list[RandomForestRegressor]:
    if not rows:
        raise ValueError("no training candidates")
    matrix = np.asarray([[row["features"][name] for name in FEATURES] for row in rows], dtype=float)
    labels = np.log1p([row["target"] for row in rows])
    fitted = []
    for completed, seed in enumerate(SEEDS, 1):
        model = RandomForestRegressor(**PARAMS, random_state=seed, n_jobs=1)
        model.fit(matrix, labels)
        fitted.append(model)
        if progress:
            progress(completed, len(SEEDS))
    return fitted


def scores(rows: list[dict], models: list[RandomForestRegressor] | None) -> list[dict]:
    matrix = np.asarray([[row["features"][name] for name in FEATURES] for row in rows], dtype=float)
    model_scores = np.mean([model.predict(matrix) for model in models], axis=0) if models and rows else [None] * len(rows)
    return [{"M": None if score is None else float(score),
             "U": row["rules"]["U"], "R1": row["rules"]["R1"], "R2": row["rules"]["R2"],
             "random": 0.0} for row, score in zip(rows, model_scores)]
