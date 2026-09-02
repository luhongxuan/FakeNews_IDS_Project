from __future__ import annotations

import unittest

import numpy as np

from strict30_relational_feature_registry import relational_features
from test_strict30_protected_feature_registry import _Graph, _frames


class _EmbeddedGraph(_Graph):
    x = np.asarray([[0.0] * 13 + [1.0, 0.0], [0.0] * 13 + [0.0, 1.0]])


class RelationalFeatureTests(unittest.TestCase):
    def test_one_reply_uses_only_observed_source_and_reply(self) -> None:
        replies, nodes = _frames()
        result = relational_features(_EmbeddedGraph(), replies, nodes)
        self.assertEqual(result["rel_first_reply_offset_sec"], 120.0)
        self.assertAlmostEqual(result["rel_source_reply_cosine_mean"], 0.0)
        self.assertAlmostEqual(result["rel_source_reply_age_delta_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
