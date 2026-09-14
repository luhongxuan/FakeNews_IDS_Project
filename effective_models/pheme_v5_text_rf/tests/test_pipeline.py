from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from effective_models.pheme_v5_text_rf import config
from effective_models.pheme_v5_text_rf.evaluation.metrics import selection_metrics
from effective_models.pheme_v5_text_rf.training.common import validate_event_split


class Scalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class Graph:
    def __init__(self, thread_id: str, event: str, impact: int, future: int, size: int) -> None:
        self.thread_id = thread_id
        self.event_id = event
        self.preventable_impact = Scalar(impact)
        self.y = torch.tensor(np.log1p(future), dtype=torch.float32)
        self.num_nodes = size


class PipelineTest(unittest.TestCase):
    def test_formal_configuration_matches_reference(self) -> None:
        self.assertEqual(config.CUTOFF_SECONDS, 1800)
        self.assertEqual(config.TOP_K, 10)
        self.assertEqual(config.PCA_COMPONENTS, 64)
        self.assertEqual(config.FULL_PARAMETERS["n_estimators"], 300)
        self.assertEqual(config.FULL_PARAMETERS["max_depth"], 8)
        self.assertEqual(config.FULL_PARAMETERS["min_samples_leaf"], 3)
        self.assertEqual(config.FULL_PARAMETERS["max_features"], 0.8)
        self.assertEqual(config.FULL_PARAMETERS["random_state"], 42)

    def test_smoke_changes_only_tree_count(self) -> None:
        full = dict(config.FULL_PARAMETERS)
        smoke = dict(config.SMOKE_PARAMETERS)
        self.assertEqual(smoke.pop("n_estimators"), 5)
        full.pop("n_estimators")
        self.assertEqual(smoke, full)

    def test_event_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Event overlap"):
            validate_event_split([Graph("a", "same", 1, 1, 1)], [Graph("b", "same", 1, 1, 1)])

    def test_metric_denominators_are_distinct_and_finite(self) -> None:
        graphs = [
            Graph("a", "e", 10, 20, 2),
            Graph("b", "e", 5, 5, 5),
            Graph("c", "e", 0, 10, 8),
        ]
        result = selection_metrics(graphs, np.asarray([3.0, 2.0, 1.0]))
        self.assertEqual(result["blocked_model"], 15)
        self.assertAlmostEqual(float(result["crr_model"]), 15 / 35)
        self.assertAlmostEqual(float(result["preventable_recall_model"]), 1.0)
        self.assertTrue(all(np.isfinite(float(value)) for value in result.values()))


if __name__ == "__main__":
    unittest.main()
