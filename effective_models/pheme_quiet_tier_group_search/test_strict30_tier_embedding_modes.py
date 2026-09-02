from __future__ import annotations

import unittest

import numpy as np

import run_strict30_two_expert_tier as tier


class EmbeddingModeTests(unittest.TestCase):
    def test_default_and_interaction_shapes_are_deterministic(self) -> None:
        source = np.asarray([1.0, 2.0], dtype=np.float32)
        replies = np.asarray([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        tier.EMBEDDING_MODE = "source_reply_mean"
        self.assertTrue(np.array_equal(tier._compose_embedding(source, replies), np.asarray([1.0, 2.0, 4.0, 5.0], dtype=np.float32)))
        tier.EMBEDDING_MODE = "source_reply_interactions"
        self.assertEqual(tier._compose_embedding(source, replies).shape, (10,))
        tier.EMBEDDING_MODE = "source_reply_mean"


if __name__ == "__main__":
    unittest.main()
