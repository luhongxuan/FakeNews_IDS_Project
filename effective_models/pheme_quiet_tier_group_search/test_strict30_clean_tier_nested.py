"""Focused tests for deterministic, outcome-free tier selection."""

from __future__ import annotations

import unittest

import pandas as pd

from run_strict30_clean_tier_nested import _ranked_high_overlap


class DeterministicTopKTests(unittest.TestCase):
    def test_equal_probabilities_break_by_thread_id_not_input_order(self) -> None:
        frame = pd.DataFrame({
            "event_id": ["event"] * 5,
            "thread_id": ["z", "a", "b", "c", "d"],
            "prob_high": [0.9, 0.5, 0.5, 0.5, 0.1],
            "tier": ["low", "high", "low", "low", "low"],
        })
        # k=ceil(5*0.2)=1: the unique highest score is selected.
        self.assertEqual(_ranked_high_overlap(frame), 0.0)
        frame.loc[0, "prob_high"] = 0.5
        # With every top candidate tied, identifier "a" is selected, not the
        # first input row "z" and not a target-informed item.
        self.assertEqual(_ranked_high_overlap(frame), 1.0)


if __name__ == "__main__":
    unittest.main()
