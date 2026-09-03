"""Unit tests for opportunity-reserve release and threshold behavior."""
from __future__ import annotations

import unittest

import pandas as pd

import adaptive_policy_common as adaptive
import opportunity_reserve_policy_common as reserve


def event_frame() -> pd.DataFrame:
    rows = []
    for checkpoint in adaptive.CHECKPOINTS:
        for index in range(20):
            rows.append({
                "thread_id": f"t{index:02d}",
                "event_id": "event",
                "checkpoint_sec": checkpoint,
                "prediction": 3.0,
                "dynamic_preventable_impact": 5,
                "future_growth": 10,
            })
    return pd.DataFrame(rows)


class OpportunityReservePolicyTest(unittest.TestCase):
    def test_candidate_grid_is_unique_and_fixed(self) -> None:
        configs = reserve.candidate_configs()
        self.assertEqual(len(configs), 208)
        self.assertEqual(len({config.config_id for config in configs}), 208)

    def test_strong_reserve_does_not_spend_full_budget_at_first_checkpoint(self) -> None:
        config = reserve.ReserveConfig(
            max_budget=20,
            min_predicted_impact=0.0,
            release_profile="strong_reserve",
            early_premium_profile="flat",
        )
        selected = reserve.reserve_select(event_frame(), config)
        first = selected.loc[selected.action_checkpoint_sec.eq(600)]
        self.assertEqual(len(first), 1)
        self.assertEqual(len(selected), 20)
        self.assertFalse(selected.thread_id.duplicated().any())

    def test_early_premium_delays_marginal_thread(self) -> None:
        frame = event_frame()
        frame["prediction"] = 2.0  # expm1(2) is below the strong minute-10 threshold 12.
        config = reserve.ReserveConfig(
            max_budget=20,
            min_predicted_impact=3.0,
            release_profile="unrestricted",
            early_premium_profile="strong",
        )
        selected = reserve.reserve_select(frame, config)
        self.assertFalse(selected.action_checkpoint_sec.eq(600).any())
        self.assertTrue(selected.action_checkpoint_sec.ge(1200).all())


if __name__ == "__main__":
    unittest.main()
