from __future__ import annotations

import unittest

import numpy as np

from quiet_head_blend_score import bounded_head_blend


class BoundedHeadBlendTests(unittest.TestCase):
    def test_zero_weight_is_exact_baseline(self) -> None:
        baseline = np.asarray([3.0, 1.0, 2.0], dtype=np.float32)
        result = bounded_head_blend(baseline, np.asarray([1.0, 3.0, 2.0], dtype=np.float32), ["a", "b", "c"], 0.0, 0.1)
        self.assertTrue(np.array_equal(result, baseline))

    def test_shift_is_bounded(self) -> None:
        result = bounded_head_blend(np.asarray([3.0, 2.0, 1.0]), np.asarray([1.0, 2.0, 3.0]), ["a", "b", "c"], 1.0, 0.1)
        self.assertLessEqual(float(np.abs(result - np.asarray([1.0, .5, 0.0])).max()), 0.1 + 1e-6)


if __name__ == "__main__":
    unittest.main()
