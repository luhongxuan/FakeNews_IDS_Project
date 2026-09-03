"""Unit tests for train-only quiet gates and acceleration metrics."""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

import quiet_acceleration_common as common


class QuietAccelerationTest(unittest.TestCase):
    def test_metrics_find_ordered_positive_acceleration(self) -> None:
        frame = pd.DataFrame({
            "thread_id": ["a", "b", "c", "d"], "event_id": ["e"] * 4,
            "checkpoint_sec": [600] * 4,
            "next10_positive_acceleration": [0.0, 0.0, 2.0, 5.0],
            "predicted_positive_acceleration": [0.0, 0.1, 1.0, 4.0],
        })
        row = common.metric_rows(frame).iloc[0]
        self.assertGreater(row.roc_auc, 0.5)
        self.assertGreater(row.pr_auc, 0.5)
        self.assertTrue(np.isfinite(row.spearman))


if __name__ == "__main__":
    unittest.main()
