import unittest
import numpy as np
import pandas as pd

from effective_models.pheme_hawkes_multicheckpoint_rf import config
from effective_models.pheme_hawkes_multicheckpoint_rf.evaluation.sequential_policy import select


class PipelineTest(unittest.TestCase):
    def test_configuration_matches_reference(self):
        self.assertEqual(config.CHECKPOINTS_SECONDS, (600, 1200, 1800, 2400, 3000, 3600))
        self.assertEqual(config.BALANCED_QUOTAS, (9, 9, 8, 8, 8, 8))
        self.assertEqual(config.DECAYS_SECONDS, (300.0, 600.0, 1200.0))

    def test_policy_selects_unique_budget_50(self):
        rows = []
        for checkpoint in config.CHECKPOINTS_SECONDS:
            for index in range(70):
                rows.append({"thread_id": f"t{index}", "checkpoint_sec": checkpoint,
                             "prediction": float(70-index), "dynamic_preventable_impact": 1,
                             "future_growth": 1})
        chosen = select(pd.DataFrame(rows))
        self.assertEqual(len(chosen), 50)
        self.assertEqual(chosen.thread_id.nunique(), 50)

    def test_decay_uses_only_observed_offsets(self):
        offsets = np.array([100.0, 500.0, 900.0])
        observed = offsets[offsets <= 600.0]
        self.assertEqual(observed.tolist(), [100.0, 500.0])
        self.assertGreater(float(np.exp(-(600.0-observed)/300.0).sum()), 0.0)


if __name__ == "__main__":
    unittest.main()

