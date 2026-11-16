from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from effective_models.pheme_active_quiet_calibrated_rf import config
from effective_models.pheme_active_quiet_calibrated_rf.evaluation.metrics import curve_rows
from effective_models.pheme_active_quiet_calibrated_rf.features.expert_score_calibration import fit_calibrators
from effective_models.pheme_active_quiet_calibrated_rf.features.quiet_active_gate_features import recency_values


class Graph:
    def __init__(self, recency: float) -> None:
        values = torch.zeros(11)
        values[5] = recency
        self.temporal_features = values


class PipelineTest(unittest.TestCase):
    def test_formal_configuration_matches_reference(self) -> None:
        self.assertEqual(config.CUTOFF_SECONDS, 1800.0)
        self.assertEqual(config.QUIET_QUANTILE, 0.60)
        self.assertEqual(config.CALIBRATION_ALPHA, 1.0)
        self.assertEqual(config.FULL_PARAMETERS["n_estimators"], 300)
        self.assertEqual(config.FULL_PARAMETERS["max_depth"], 8)
        self.assertEqual(config.FULL_PARAMETERS["random_state"], 42)

    def test_recency_gate_reads_documented_feature(self) -> None:
        np.testing.assert_allclose(recency_values([Graph(1.5), Graph(3.0)]), [1.5, 3.0])

    def test_calibrator_never_reverses_expert_order(self) -> None:
        rows = []
        for expert in ("active", "quiet"):
            for index in range(30):
                rows.append(
                    {
                        "validation_event": f"e{index % 3}",
                        "thread_id": f"{expert}-{index}",
                        "expert": expert,
                        "raw_score": float(index),
                        "target": float(29 - index),
                        "event_weight": 1.0,
                    }
                )
        models, details = fit_calibrators(pd.DataFrame(rows))
        self.assertEqual(models["active"], {"coefficient": 1.0, "intercept": 0.0})
        self.assertEqual(models["quiet"], {"coefficient": 1.0, "intercept": 0.0})
        self.assertTrue(all(row["calibration_method"].startswith("identity") for row in details))

    def test_curve_uses_impact_denominator_and_stable_tie_break(self) -> None:
        pool = pd.DataFrame(
            {
                "thread_id": ["b", "a", "c"],
                "event_id": ["event"] * 3,
                "score": [1.0, 1.0, 0.0],
                "preventable_impact": [1.0, 9.0, 10.0],
            }
        )
        rows = curve_rows(pool, "score", "model")
        budget1 = next(row for row in rows if row["budget"] == 1)
        self.assertEqual(budget1["model_blocked_impact"], 9.0)
        self.assertAlmostEqual(float(budget1["model_reduction"]), 9 / 20)


if __name__ == "__main__":
    unittest.main()
