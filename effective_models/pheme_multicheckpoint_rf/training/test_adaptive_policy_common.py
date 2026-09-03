"""Unit tests for adaptive intervention selection and inner-only policy choice."""
from __future__ import annotations

import unittest

import pandas as pd

import adaptive_policy_common as common


def event_frame(event_id: str, predictions: tuple[float, float]) -> pd.DataFrame:
    rows = []
    for checkpoint in common.CHECKPOINTS:
        for index, prediction in enumerate(predictions):
            rows.append({
                "thread_id": f"{event_id}_t{index}",
                "event_id": event_id,
                "checkpoint_sec": checkpoint,
                "prediction": prediction,
                "dynamic_preventable_impact": max(6 - checkpoint // 600 - index, 0),
                "future_growth": 10,
            })
    return pd.DataFrame(rows)


class AdaptivePolicyTest(unittest.TestCase):
    def test_threshold_does_not_force_budget_fill(self) -> None:
        frame = event_frame("event", (2.0, 0.0))
        chosen = common.adaptive_select(
            frame,
            common.AdaptiveConfig(max_budget=5, min_predicted_impact=3.0),
        )
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen.iloc[0].thread_id, "event_t0")
        self.assertEqual(chosen.iloc[0].action_checkpoint_sec, 600)

    def test_thread_is_selected_only_once(self) -> None:
        frame = event_frame("event", (3.0, 2.0))
        chosen = common.adaptive_select(
            frame,
            common.AdaptiveConfig(max_budget=5, min_predicted_impact=0.0),
        )
        self.assertEqual(len(chosen), 2)
        self.assertFalse(chosen.thread_id.duplicated().any())

    def test_target_policy_prefers_fewer_interventions(self) -> None:
        summary = pd.DataFrame([
            {
                "config_id": "many", "max_budget": 50, "min_predicted_impact": 0.0,
                "validation_events": 3, "mean_interventions": 40.0,
                "max_interventions": 50, "mean_positive_interventions": 30.0,
                "mean_blocked_impact": 100.0, "mean_horizon_reduction": 0.31,
                "worst_event_horizon_reduction": 0.20,
            },
            {
                "config_id": "few", "max_budget": 30, "min_predicted_impact": 5.0,
                "validation_events": 3, "mean_interventions": 20.0,
                "max_interventions": 25, "mean_positive_interventions": 18.0,
                "mean_blocked_impact": 90.0, "mean_horizon_reduction": 0.30,
                "worst_event_horizon_reduction": 0.18,
            },
        ])
        selected = common.select_policies(summary)
        target30 = selected.loc[selected.policy_name.eq("target_30")].iloc[0]
        self.assertEqual(target30.config_id, "few")
        self.assertFalse(bool(target30.used_fallback))

    def test_pareto_frontier_removes_dominated_policy(self) -> None:
        macro = pd.DataFrame([
            {"policy_name": "efficient", "mean_interventions": 20.0,
             "mean_horizon_reduction": 0.25},
            {"policy_name": "dominated", "mean_interventions": 30.0,
             "mean_horizon_reduction": 0.24},
            {"policy_name": "higher_return", "mean_interventions": 40.0,
             "mean_horizon_reduction": 0.35},
        ])
        frontier = common.pareto_frontier(macro)
        self.assertEqual(set(frontier.policy_name), {"efficient", "higher_return"})


if __name__ == "__main__":
    unittest.main()
