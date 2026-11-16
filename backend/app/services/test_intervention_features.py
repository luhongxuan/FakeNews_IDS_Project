from __future__ import annotations

import math
import unittest

from app.services import intervention_features, intervention_model


def _nodes() -> list[dict]:
    return [
        {
            "id": "source", "parent_id": None, "is_source": True,
            "offset_sec": 0.0, "depth": 0, "text": "Breaking source report",
        },
        {
            "id": "early", "parent_id": "source", "is_source": False,
            "offset_sec": 120.0, "depth": 1, "text": "First reply!",
        },
        {
            "id": "future", "parent_id": "early", "is_source": False,
            "offset_sec": 1900.0, "depth": 2, "text": "After the cutoff",
        },
    ]


class InterventionFeatureIntegrationTest(unittest.TestCase):
    def test_live_vector_matches_frozen_model_schema_and_scores(self) -> None:
        features = intervention_features.build_cumulative_features(_nodes(), 1800.0)
        self.assertEqual(len(features), 57)
        self.assertEqual(set(features), set(intervention_model.feature_columns()))
        result = intervention_model.predict_impact(features)
        self.assertTrue(math.isfinite(result["predicted_impact"]))
        self.assertEqual(len(result["top_features"]), 8)

    def test_post_cutoff_reply_is_not_observed(self) -> None:
        features = intervention_features.build_cumulative_features(_nodes(), 1800.0)
        self.assertEqual(features["cumulative__cum_observed_reply_count"], 1.0)
        self.assertEqual(features["cumulative__cum_depth_max"], 1.0)

    def test_missing_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no source node"):
            intervention_features.build_cumulative_features(_nodes()[1:], 1800.0)


if __name__ == "__main__":
    unittest.main()
