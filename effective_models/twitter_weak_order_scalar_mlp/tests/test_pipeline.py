import unittest
import numpy as np
from effective_models.twitter_weak_order_scalar_mlp import config
from effective_models.twitter_weak_order_scalar_mlp.model.scalar_mlp import ScalarNet, head_pairs
from effective_models.twitter_weak_order_scalar_mlp.features.strict30_features import feature_columns


class PipelineTest(unittest.TestCase):
    def test_frozen_configuration(self):
        self.assertEqual(config.ORDER_LAMBDA, 0.05); self.assertEqual(config.SOFT_TOPK_LAMBDA, 0.0)
        self.assertEqual(config.CUTOFF_SECONDS, 1800); self.assertEqual(len(feature_columns()), 31)

    def test_model_shape(self):
        import torch
        self.assertEqual(tuple(ScalarNet(31)(torch.zeros((4, 31))).shape), (4,))

    def test_pairs_follow_target_order(self):
        target = np.arange(60, dtype=float); high, low = head_pairs(target, 42)
        self.assertTrue(len(high) > 0); self.assertTrue((target[high] > target[low]).all())


if __name__ == "__main__": unittest.main()

