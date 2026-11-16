"""Tests for the canonical Twitter graph-only RF pipeline."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from effective_models.twitter_graph_only_rf.config import (  # noqa: E402
    EARLY_GRAPH_FEATURES, FULL_PARAMETERS, SMOKE_PARAMETERS,
)
from effective_models.twitter_graph_only_rf.training.common import evaluate, fit_rows  # noqa: E402


def synthetic_frame(rows: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(rows, len(EARLY_GRAPH_FEATURES))), columns=EARLY_GRAPH_FEATURES)
    frame["preventable_impact"] = np.arange(rows, dtype=float)
    frame["preventable_y"] = np.log1p(frame.preventable_impact)
    frame["thread_id"] = [f"t{index:03d}" for index in range(rows)]
    return frame


class PipelineTest(unittest.TestCase):
    def test_smoke_and_full_share_the_same_configuration_names(self) -> None:
        self.assertEqual(set(SMOKE_PARAMETERS), set(FULL_PARAMETERS))
        for name in FULL_PARAMETERS:
            self.assertEqual(SMOKE_PARAMETERS[name]["max_depth"], FULL_PARAMETERS[name]["max_depth"])
            self.assertEqual(SMOKE_PARAMETERS[name]["min_samples_leaf"], FULL_PARAMETERS[name]["min_samples_leaf"])

    def test_fit_and_evaluate_return_finite_metrics(self) -> None:
        frame = synthetic_frame()
        model = fit_rows(frame.iloc[:20], SMOKE_PARAMETERS["rf_depth_6"])
        result, selected = evaluate(model, frame.iloc[20:])
        self.assertTrue(np.isfinite(result["crr_at_10"]))
        self.assertEqual(len(selected), 10)
        self.assertEqual(list(selected.score), sorted(selected.score, reverse=True))

    def test_outcome_and_identity_are_not_features(self) -> None:
        forbidden = {"preventable_impact", "preventable_y", "thread_id", "original_label"}
        self.assertFalse(forbidden & set(EARLY_GRAPH_FEATURES))


if __name__ == "__main__":
    unittest.main()
