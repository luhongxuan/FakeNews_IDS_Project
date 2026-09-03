"""Unit tests for corrected target overlay and nested score evidence."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import pandas as pd
import torch

import corrected_quiet_tier_hybrid_common as common


class CorrectedQuietTierHybridTest(unittest.TestCase):
    def test_overlay_is_derived_and_identity_aligned(self) -> None:
        original_expected = common.EXPECTED_CHANGED_TARGETS
        common.EXPECTED_CHANGED_TARGETS = 1
        try:
            frame = pd.DataFrame({
                "thread_id": ["a", "b"], "preventable_impact": [1.0, 2.0]
            })
            graphs = [
                SimpleNamespace(thread_id="a", preventable_impact=torch.tensor([3.0])),
                SimpleNamespace(thread_id="b", preventable_impact=torch.tensor([2.0])),
            ]
            derived, changed = common.overlay_corrected_targets(frame, graphs)
            self.assertEqual(changed, 1)
            self.assertEqual(derived.preventable_impact.tolist(), [3.0, 2.0])
            self.assertEqual(frame.preventable_impact.tolist(), [1.0, 2.0])
        finally:
            common.EXPECTED_CHANGED_TARGETS = original_expected

    def test_nested_evidence_rejects_outer_event(self) -> None:
        rows = [{
            "validation_event": "outer", "thread_id": "t", "expert": "quiet",
            "target": 1.0, "raw_score": .5, "event_weight": 1.0,
        }]
        calibrators = {
            "quiet": {"coefficient": 1.0, "intercept": 0.0},
            "active": {"coefficient": 1.0, "intercept": 0.0},
        }
        with self.assertRaises(ValueError):
            common.nested_score_evidence(
                "outer", pd.DataFrame(rows), calibrators,
                pd.DataFrame(rows), calibrators, 0.5,
            )


if __name__ == "__main__":
    unittest.main()
