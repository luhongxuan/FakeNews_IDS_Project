from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from effective_models.pheme_graph7_discourse_rf import config
from effective_models.pheme_graph7_discourse_rf.features.discourse_features import feature_variants
from effective_models.pheme_graph7_discourse_rf.training.common import choose_variant, evaluate_split


class PipelineTest(unittest.TestCase):
    def test_formal_configuration_matches_frozen_reference(self) -> None:
        self.assertEqual(config.TOP_K, 10)
        self.assertEqual(config.ELIGIBLE_EVENT_MIN_CANDIDATES, 100)
        self.assertEqual(config.FULL_PARAMETERS["n_estimators"], 300)
        self.assertEqual(config.FULL_PARAMETERS["max_depth"], 8)
        self.assertEqual(config.FULL_PARAMETERS["min_samples_leaf"], 3)
        self.assertEqual(config.FULL_PARAMETERS["max_features"], 0.8)
        self.assertEqual(config.FULL_PARAMETERS["random_state"], 42)

    def test_feature_bundles_are_ordered_and_safe(self) -> None:
        schema = {
            "metadata_columns": ["thread_id", "event_id", "category"],
            "target_columns": ["preventable_impact", "preventable_y"],
            "base_feature_columns": ["role_a", "event_context_a"],
            "discourse_feature_columns": ["discourse_a"],
        }
        variants = feature_variants(schema)
        self.assertEqual(variants["role_base"], ["role_a"])
        self.assertEqual(variants["plus_event_context"], ["role_a", "event_context_a"])
        self.assertEqual(
            variants["plus_event_context_discourse"],
            ["role_a", "event_context_a", "discourse_a"],
        )

    def test_evaluate_split_rejects_event_overlap(self) -> None:
        frame = pd.DataFrame(
            {
                "event_id": ["same"],
                "category": ["rumours"],
                "x": [1.0],
                "preventable_y": [0.0],
                "preventable_impact": [0.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "Event leakage"):
            evaluate_split(frame, frame, ["x"], config.SMOKE_PARAMETERS)

    def test_fit_metric_and_nested_selection_are_finite(self) -> None:
        rng = np.random.default_rng(42)
        rows = []
        for event in ("a", "b", "c"):
            for index in range(105):
                impact = int(rng.integers(0, 20))
                rows.append(
                    {
                        "event_id": event,
                        "category": "rumours",
                        "role": float(rng.normal()),
                        "event_context": float(rng.normal()),
                        "discourse": float(rng.normal()),
                        "preventable_impact": impact,
                        "preventable_y": float(np.log1p(impact)),
                    }
                )
        frame = pd.DataFrame(rows)
        variants = {
            "role_base": ["role"],
            "plus_event_context": ["role", "event_context"],
            "plus_event_context_discourse": ["role", "event_context", "discourse"],
        }
        chosen, means = choose_variant(
            frame, variants, "a", config.SMOKE_PARAMETERS, ["b", "c"]
        )
        self.assertIn(chosen, variants)
        self.assertTrue(all(np.isfinite(value) for value in means.values()))
        result = evaluate_split(
            frame.loc[frame["event_id"].ne("a")],
            frame.loc[frame["event_id"].eq("a")],
            variants[chosen],
            config.SMOKE_PARAMETERS,
        )
        self.assertTrue(np.isfinite(float(result["crr_at_10"])))


if __name__ == "__main__":
    unittest.main()
