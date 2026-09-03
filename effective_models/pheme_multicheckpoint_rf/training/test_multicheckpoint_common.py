"""Boundary and future-perturbation tests for multi-checkpoint features."""
from __future__ import annotations

import unittest

import pandas as pd

import multicheckpoint_common as common


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"thread_id": "t", "tweet_id": "s", "parent_id": None, "is_source": 1,
             "depth": 0, "offset_sec": 0.0, "reply_latency_sec": 0.0, "text": "source", "event_id": "e"},
            {"thread_id": "t", "tweet_id": "a", "parent_id": "s", "is_source": 0,
             "depth": 1, "offset_sec": 600.0, "reply_latency_sec": 600.0, "text": "early?", "event_id": "e"},
            {"thread_id": "t", "tweet_id": "b", "parent_id": "a", "is_source": 0,
             "depth": 2, "offset_sec": 900.0, "reply_latency_sec": 300.0, "text": "middle!", "event_id": "e"},
            {"thread_id": "t", "tweet_id": "c", "parent_id": "b", "is_source": 0,
             "depth": 3, "offset_sec": 1500.0, "reply_latency_sec": 600.0, "text": "future", "event_id": "e"},
        ]
    )


class MultiCheckpointFeatureTest(unittest.TestCase):
    def test_window_boundaries(self) -> None:
        raw = frame()
        replies = raw.loc[raw.is_source.eq(0)]
        features = common._window_delta_features(replies, pd.Series(["source"]), "s", 1200.0)
        self.assertEqual(features["win_current_count"], 1.0)
        self.assertEqual(features["win_previous_count"], 1.0)

    def test_future_perturbation_does_not_change_features(self) -> None:
        raw = frame()
        altered = raw.copy()
        altered.loc[altered.offset_sec.gt(1200), ["text", "depth", "parent_id"]] = [
            "completely changed future", 99, "s"
        ]
        source = raw.loc[raw.is_source.eq(1)]
        observed = raw.loc[raw.is_source.eq(0) & raw.offset_sec.le(1200)]
        altered_observed = altered.loc[altered.is_source.eq(0) & altered.offset_sec.le(1200)]
        cumulative_a = common._cumulative_features(observed, source.text, "s", 1200.0)
        cumulative_b = common._cumulative_features(altered_observed, source.text, "s", 1200.0)
        window_a = common._window_delta_features(raw.loc[raw.is_source.eq(0)], source.text, "s", 1200.0)
        window_b = common._window_delta_features(altered.loc[altered.is_source.eq(0)], source.text, "s", 1200.0)
        self.assertEqual(cumulative_a, cumulative_b)
        self.assertEqual(window_a, window_b)

    def test_dynamic_target_is_bounded_by_future_growth(self) -> None:
        raw = frame()
        impact = common.dynamic_preventable_impact(raw, 1200.0)
        future = int((raw.is_source.eq(0) & raw.offset_sec.gt(1200)).sum())
        self.assertEqual(impact, 1)
        self.assertLessEqual(impact, future)


if __name__ == "__main__":
    unittest.main()
