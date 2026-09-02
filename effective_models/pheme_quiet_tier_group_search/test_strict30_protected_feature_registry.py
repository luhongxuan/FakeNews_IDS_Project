"""Small no-dataset tests for strict-30 feature safety and schema."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from strict30_protected_feature_registry import CUTOFF_SECONDS, feature_group, scalar_features


class _Graph:
    thread_id = "thread-1"
    node_ids = ["source", "reply"]
    edge_index = np.asarray([[0], [1]])
    semantic_features = np.asarray([[0.1, 0.2], [0.3, 0.4]])


def _frames(reply_offset: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    replies = pd.DataFrame({
        "thread_id": ["thread-1", "thread-1"], "tweet_id": ["source", "reply"],
        "parent_id": ["", "source"], "is_source": [True, False],
        "is_text_available": [True, True], "depth": [0, 1],
        "offset_sec": [0, reply_offset], "reply_latency_sec": [0, reply_offset],
        "text": ["Source #topic", "A reply!"], "event_id": ["event", "event"],
    })
    nodes = pd.DataFrame({
        "thread_id": ["thread-1", "thread-1"], "tweet_id": ["source", "reply"],
        "event_id": ["event", "event"], "vader_compound": [0.0, 0.2],
        "vader_pos": [0.1, 0.2], "vader_neg": [0.0, 0.1],
        "vader_neu": [0.9, 0.7], "account_age_days_log": [2.0, 3.0],
    })
    return replies, nodes


class Strict30RegistryTests(unittest.TestCase):
    def test_uses_only_snapshot_aligned_rows(self) -> None:
        replies, nodes = _frames()
        features = scalar_features(_Graph(), replies, nodes)
        self.assertEqual(features["activity_observed_nodes"], 2.0)
        self.assertEqual(features["activity_observed_replies"], 1.0)
        self.assertEqual(features["event_id"], "event")
        self.assertEqual(feature_group("source_text_chars_mean"), "text_surface")
        self.assertEqual(feature_group("followers_log"), None)

    def test_rejects_post_cutoff_graph_node(self) -> None:
        replies, nodes = _frames(CUTOFF_SECONDS + 1)
        with self.assertRaisesRegex(ValueError, "post-cutoff"):
            scalar_features(_Graph(), replies, nodes)


if __name__ == "__main__":
    unittest.main()
