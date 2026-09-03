"""Tests for fixed-budget quiet challenger replay."""
from __future__ import annotations

import unittest

import pandas as pd

import quiet_hybrid_challenger_common as common


class QuietHybridChallengerTest(unittest.TestCase):
    def test_replay_respects_reserves_budget_and_identity(self) -> None:
        hybrid = pd.DataFrame([
            {"thread_id": f"t{i}", "event_id": "e", "hybrid_calibrated_score": 100-i,
             "dynamic_preventable_impact": i % 7, "future_growth": 20}
            for i in range(60)
        ])
        dual_rows = []
        for checkpoint in common.CHECKPOINTS:
            for i in range(10):
                dual_rows.append({
                    "thread_id": f"t{i}", "event_id": "e", "checkpoint_sec": checkpoint,
                    "acceleration_prediction": 1-i/100, "predicted_impact": 10-i/10,
                    "dynamic_preventable_impact": 10-i, "future_growth": 20,
                })
        config = common.ChallengerConfig(
            "gate", .8, 1.0, .0, .0, reserve_20m=2, reserve_30m=3,
        )
        selected = common.replay(pd.DataFrame(dual_rows), hybrid, config)
        self.assertEqual(len(selected), 50)
        self.assertFalse(selected.thread_id.duplicated().any())
        challenger = selected.loc[selected.selection_source.eq("quiet_challenger")]
        self.assertEqual(challenger.action_checkpoint_sec.value_counts().to_dict(), {1200: 2, 1800: 3})


if __name__ == "__main__":
    unittest.main()
