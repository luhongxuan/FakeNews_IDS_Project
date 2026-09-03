"""Tests for next-window target construction and hazard-gated timing."""
from __future__ import annotations

import unittest

import pandas as pd

import adaptive_policy_common as adaptive
import wait_loss_hazard_common as hazard


class WaitLossHazardTest(unittest.TestCase):
    def test_grid_is_fixed(self) -> None:
        self.assertEqual(len(hazard.candidate_configs()), 72)

    def test_hazard_gate_defers_until_urgent(self) -> None:
        rows = []
        for checkpoint in adaptive.CHECKPOINTS:
            rows.append({
                "thread_id": "t", "event_id": "e", "checkpoint_sec": checkpoint,
                "prediction": 3.0,
                "hazard_prediction": 0.0 if checkpoint < 1800 else 2.0,
                "dynamic_preventable_impact": 5, "future_growth": 10,
            })
        chosen = hazard.hazard_select(
            pd.DataFrame(rows), hazard.HazardConfig(20, 3.0, 1.0)
        )
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen.iloc[0].action_checkpoint_sec, 1800)

    def test_prediction_metrics_are_finite(self) -> None:
        scored = pd.DataFrame({
            "event_id": ["e"] * 4, "checkpoint_sec": [600] * 4,
            "next10_wait_loss": [0.0, 0.0, 1.0, 4.0],
            "hazard_prediction": [0.0, 0.1, 0.7, 1.5],
        })
        metrics = hazard.prediction_metrics(scored).iloc[0]
        self.assertGreater(metrics.roc_auc_positive, 0.5)
        self.assertGreater(metrics.pr_auc_positive, 0.5)
        self.assertGreater(metrics.top20_wait_loss_capture, 0.0)


if __name__ == "__main__":
    unittest.main()
