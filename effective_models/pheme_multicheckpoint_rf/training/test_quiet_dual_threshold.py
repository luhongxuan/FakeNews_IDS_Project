"""Unit tests for two-score threshold-only qualification."""
from __future__ import annotations

import unittest

import pandas as pd

import quiet_dual_threshold_common as common


class QuietDualThresholdTest(unittest.TestCase):
    def test_both_thresholds_are_required_and_identity_is_unique(self) -> None:
        rows = []
        for checkpoint in common.CHECKPOINTS:
            rows.extend([
                {"thread_id": "both", "event_id": "e", "checkpoint_sec": checkpoint,
                 "acceleration_prediction": .9, "predicted_impact": 5.0,
                 "dynamic_preventable_impact": 5, "future_growth": 8},
                {"thread_id": "acc_only", "event_id": "e", "checkpoint_sec": checkpoint,
                 "acceleration_prediction": .9, "predicted_impact": .1,
                 "dynamic_preventable_impact": 1, "future_growth": 8},
                {"thread_id": "impact_only", "event_id": "e", "checkpoint_sec": checkpoint,
                 "acceleration_prediction": .1, "predicted_impact": 5.0,
                 "dynamic_preventable_impact": 1, "future_growth": 8},
            ])
        selected = common.apply_config(
            pd.DataFrame(rows), common.DualThresholdConfig(.5, 1.0),
            {1200: .5, 1800: .5},
        )
        self.assertEqual(selected.thread_id.tolist(), ["both"])
        self.assertFalse(selected.thread_id.duplicated().any())

    def test_policy_selection_uses_minimum_interventions(self) -> None:
        macro = pd.DataFrame([
            {"config_id": "many", "acceleration_quantile": 0.0,
             "min_predicted_impact": 0.0, "acceleration_threshold_20m": 0.0,
             "acceleration_threshold_30m": 0.0, "validation_events": 2,
             "mean_interventions": 20, "max_interventions": 20,
             "mean_blocked_impact": 10, "mean_quiet_impact_capture": .70,
             "worst_event_quiet_impact_capture": .40},
            {"config_id": "few", "acceleration_quantile": .5,
             "min_predicted_impact": 1.0, "acceleration_threshold_20m": .2,
             "acceleration_threshold_30m": .1, "validation_events": 2,
             "mean_interventions": 10, "max_interventions": 11,
             "mean_blocked_impact": 9, "mean_quiet_impact_capture": .40,
             "worst_event_quiet_impact_capture": .30},
        ])
        selected = common.select_capture_policies(macro)
        target40 = selected.loc[
            selected.policy_name.eq("inner_impact_capture_40")
        ].iloc[0]
        self.assertEqual(target40.config_id, "few")


if __name__ == "__main__":
    unittest.main()
