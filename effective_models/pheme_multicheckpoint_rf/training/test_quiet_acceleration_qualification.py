"""Tests for split-safe quiet acceleration qualification."""
from __future__ import annotations

import unittest

import pandas as pd

import quiet_acceleration_qualification_common as common


class QuietAccelerationQualificationTest(unittest.TestCase):
    def test_threshold_meets_macro_and_worst_recall(self) -> None:
        frame = pd.DataFrame([
            {"event_id": event, "checkpoint_sec": 1200, "prediction": score,
             "next10_positive_acceleration": actual}
            for event, score, actual in [
                ("a", .9, 1), ("a", .8, 0), ("a", .2, 1),
                ("b", .7, 1), ("b", .6, 0), ("b", .1, 1),
            ]
        ])
        frontier = common.threshold_frontier(frame, 1200)
        chosen = common.choose_threshold(frontier, 0.50)
        self.assertGreaterEqual(chosen.macro_positive_recall, 0.50)
        self.assertGreaterEqual(chosen.worst_event_positive_recall, 0.25)
        self.assertEqual(chosen.selected_rows, 3)

        sparse = common.choose_threshold(frontier, 0.25, worst_event_fraction=None)
        self.assertEqual(sparse.selected_rows, 1)
        self.assertEqual(sparse.macro_positive_recall, 0.25)

    def test_replay_never_selects_thread_twice(self) -> None:
        rows = []
        for checkpoint in common.CHECKPOINTS:
            rows.extend([
                {"thread_id": "same", "event_id": "e", "checkpoint_sec": checkpoint,
                 "prediction": .9, "next10_positive_acceleration": 1,
                 "dynamic_preventable_impact": 5, "future_growth": 10},
                {"thread_id": f"only{checkpoint}", "event_id": "e",
                 "checkpoint_sec": checkpoint, "prediction": .8,
                 "next10_positive_acceleration": 0,
                 "dynamic_preventable_impact": 1, "future_growth": 10},
            ])
        selected, checkpoints = common.apply_thresholds(
            pd.DataFrame(rows), {1200: .5, 1800: .5}, "test",
        )
        self.assertFalse(selected.thread_id.duplicated().any())
        self.assertEqual(len(selected), 3)
        self.assertEqual(checkpoints.new_interventions.tolist(), [2, 1])


if __name__ == "__main__":
    unittest.main()
