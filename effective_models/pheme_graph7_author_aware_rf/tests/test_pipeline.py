from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from effective_models.pheme_graph7_author_aware_rf import config
from effective_models.pheme_graph7_author_aware_rf.training.common import evaluate_fold


class PipelineTest(unittest.TestCase):
    def test_formal_configuration_matches_frozen_reference(self) -> None:
        self.assertEqual(config.TOP_K, 10)
        self.assertEqual(config.FULL_PARAMETERS["n_estimators"], 300)
        self.assertEqual(config.FULL_PARAMETERS["max_depth"], 8)
        self.assertEqual(config.FULL_PARAMETERS["min_samples_leaf"], 3)
        self.assertEqual(config.FULL_PARAMETERS["max_features"], 0.8)
        self.assertEqual(config.FULL_PARAMETERS["random_state"], 42)

    def test_smoke_changes_only_tree_count(self) -> None:
        full = dict(config.FULL_PARAMETERS)
        smoke = dict(config.SMOKE_PARAMETERS)
        self.assertEqual(smoke.pop("n_estimators"), 5)
        full.pop("n_estimators")
        self.assertEqual(smoke, full)

    def test_fit_and_metric_pipeline_is_finite(self) -> None:
        rng = np.random.default_rng(42)
        features = ["observed_actions", "x"]
        train = pd.DataFrame(
            {
                "observed_actions": rng.integers(1, 8, 40),
                "x": rng.normal(size=40),
                "preventable_y": rng.normal(size=40),
            }
        )
        test = pd.DataFrame(
            {
                "thread_id": [f"t{i}" for i in range(15)],
                "category": ["rumours"] * 15,
                "observed_actions": rng.integers(1, 8, 15),
                "x": rng.normal(size=15),
                "preventable_impact": rng.integers(0, 20, 15),
            }
        )
        model = RandomForestRegressor(**config.SMOKE_PARAMETERS).fit(
            train[features], train["preventable_y"]
        )
        metrics = evaluate_fold(model, test, features, np.random.default_rng(42))
        for value in metrics.values():
            self.assertTrue(np.isfinite(float(value)))


if __name__ == "__main__":
    unittest.main()
