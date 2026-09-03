"""Tests for leakage-safe timing-only intervention replay."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import quiet_timing_only_common as common


class QuietTimingOnlyTest(unittest.TestCase):
    def test_timing_changes_actions_not_identities(self) -> None:
        utility = pd.DataFrame([
            {"thread_id": f"t{i}", "event_id": "e", "utility_prediction": 100-i,
             "predicted_impact": 10-i/10}
            for i in range(60)
        ])
        outcomes = pd.DataFrame([
            {"thread_id": f"t{i}", "event_id": "e", "checkpoint_sec": checkpoint,
             "dynamic_preventable_impact": 5 if checkpoint == 1200 else 4}
            for i in range(60) for checkpoint in (1200, 1800)
        ])
        committed = common.commit_at_20(utility, outcomes)
        quiet = pd.DataFrame([
            {"thread_id": f"t{i}", "event_id": "e", "acceleration_prediction": 1.0,
             "predicted_positive_acceleration": 1.0, "next10_positive_acceleration": 1.0}
            for i in range(5)
        ])
        selected = common.timing_replay(committed, quiet, .5, 1.0, "test")
        self.assertEqual(set(selected.thread_id), set(committed.thread_id))
        self.assertEqual(int(selected.early_intervention.sum()), 5)
        metrics = common.timing_metrics(selected)
        self.assertEqual(metrics["timing_gain"], 5.0)
        self.assertEqual(metrics["interventions"], 50)

    def test_missing_quiet_score_does_not_trigger(self) -> None:
        committed = pd.DataFrame({
            "thread_id": [f"t{i}" for i in range(50)], "event_id": "e",
            "utility_prediction": np.arange(50), "predicted_impact": 2.0,
            "impact_20m": 2.0, "impact_30m": 1.0, "wait_loss_20_to_30": 1.0,
        })
        empty = pd.DataFrame(columns=[
            "thread_id", "event_id", "acceleration_prediction",
            "predicted_positive_acceleration", "next10_positive_acceleration",
        ])
        selected = common.timing_replay(committed, empty, .1, 1.0, "test")
        self.assertFalse(selected.early_intervention.any())


if __name__ == "__main__":
    unittest.main()
