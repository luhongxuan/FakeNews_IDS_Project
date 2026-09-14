from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from effective_models.pheme_multicheckpoint_rf import config
from effective_models.pheme_multicheckpoint_rf.evaluation.sequential_policy import sequential_select
from effective_models.pheme_multicheckpoint_rf.features.cumulative_features import (
    build_cumulative_features, load_frozen_materialization,
)
from effective_models.pheme_multicheckpoint_rf.training.common import fit_predict


class PipelineTest(unittest.TestCase):
    def test_formal_configuration_matches_reference(self) -> None:
        self.assertEqual(config.CHECKPOINTS, (600, 1200, 1800, 2400, 3000, 3600))
        self.assertEqual(config.BALANCED_QUOTAS, (9, 9, 8, 8, 8, 8))
        self.assertEqual(sum(config.BALANCED_QUOTAS), 50)
        self.assertEqual(config.FULL_PARAMETERS["n_estimators"], 300)
        self.assertEqual(config.FULL_PARAMETERS["max_depth"], 8)
        self.assertEqual(config.FULL_PARAMETERS["random_state"], 42)

    def test_event_overlap_is_rejected(self) -> None:
        frame = pd.DataFrame({"event_id": ["same"], "x": [1.0], "dynamic_preventable_y": [0.0]})
        with self.assertRaisesRegex(ValueError, "Event overlap"):
            fit_predict(frame, frame, ["x"], config.SMOKE_PARAMETERS)

    def test_sequential_policy_uses_50_unique_threads(self) -> None:
        rows = []
        for checkpoint in config.CHECKPOINTS:
            for index in range(70):
                rows.append(
                    {
                        "thread_id": f"t{index}", "event_id": "e", "checkpoint_sec": checkpoint,
                        "prediction": float(70 - index), "dynamic_preventable_impact": index,
                    }
                )
        chosen = sequential_select(pd.DataFrame(rows))
        self.assertEqual(len(chosen), 50)
        self.assertEqual(chosen["thread_id"].nunique(), 50)
        self.assertEqual(tuple(chosen.groupby("action_checkpoint_sec").size()), config.BALANCED_QUOTAS)

    def test_fit_predict_is_finite(self) -> None:
        rng = np.random.default_rng(42)
        train = pd.DataFrame(
            {"event_id": ["train"] * 100, "x": rng.normal(size=100), "dynamic_preventable_y": rng.normal(size=100)}
        )
        test = pd.DataFrame(
            {"event_id": ["test"] * 20, "x": rng.normal(size=20), "dynamic_preventable_y": rng.normal(size=20)}
        )
        _, prediction = fit_predict(train, test, ["x"], config.SMOKE_PARAMETERS)
        self.assertEqual(prediction.shape, (20,))
        self.assertTrue(np.isfinite(prediction).all())

    def test_live_cumulative_features_are_cutoff_safe_and_schema_aligned(self) -> None:
        replies = pd.DataFrame(
            [
                {
                    "tweet_id": "r1", "parent_id": "source", "depth": 1.0,
                    "offset_sec": 120.0, "reply_latency_sec": 120.0, "text": "First reply!",
                },
                {
                    "tweet_id": "r2", "parent_id": "r1", "depth": 2.0,
                    "offset_sec": 900.0, "reply_latency_sec": 780.0, "text": "Why? https://example.test",
                },
            ]
        )
        features = build_cumulative_features(
            replies, pd.Series(["SOURCE #News"]), "source", 1800.0
        )
        _, columns = load_frozen_materialization()
        self.assertEqual(set(features), set(columns))
        self.assertEqual(features["cumulative__cum_observed_reply_count"], 2.0)
        self.assertEqual(features["cumulative__cum_root_children"], 1.0)
        self.assertTrue(np.isfinite(np.asarray(list(features.values()), dtype=float)).all())

        with self.assertRaisesRegex(ValueError, "observation cutoff"):
            build_cumulative_features(
                replies.assign(offset_sec=[120.0, 1900.0]),
                pd.Series(["SOURCE #News"]), "source", 1800.0,
            )


if __name__ == "__main__":
    unittest.main()
