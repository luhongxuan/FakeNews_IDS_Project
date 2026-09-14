from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "postgresql://admin:secret@localhost:15432/misinfo_db")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services import intervention_policy as policy
from app.services import manual_intervention_mvp as mvp


def decide(credibility: str, confidence: float | None, impact: float = 10.0, already: int = 0):
    candidate = policy.Candidate("post", "cluster", impact, credibility, confidence)
    selection = policy.select_for_checkpoint([candidate], 10, already, 50)
    result = mvp.recommendation_from_selection(
        candidate, selection, {"summary": "evidence", "from_cache": False},
    )
    return result


class ManualInterventionMvpTest(unittest.TestCase):
    def test_confirmed_true_is_none(self):
        result = decide("likely_true", 0.9)
        self.assertEqual("none", result["action"])
        self.assertEqual("0%", result["action_strength"])
        self.assertIsNone(result["priority"])

    def test_confirmed_false_is_hard(self):
        result = decide("likely_false", 0.9)
        self.assertEqual("hard", result["action"])
        self.assertEqual(1.0, result["risk_weight"])

    def test_unverified_is_soft_not_hard(self):
        result = decide("unverified", 0.0)
        self.assertEqual("soft", result["action"])
        self.assertEqual(0.5, result["risk_weight"])

    def test_low_confidence_false_is_soft(self):
        result = decide("likely_false", 0.2)
        self.assertEqual("soft", result["action"])
        self.assertEqual("unverified", result["effective_credibility"])

    def test_agent_failure_is_deferred_without_priority(self):
        result = decide("agent_failure", None)
        self.assertEqual("deferred_agent_failure", result["action"])
        self.assertEqual("0%", result["action_strength"])
        self.assertIsNone(result["priority"])

    def test_inflight_is_deferred_without_priority(self):
        result = decide("deferred_inflight", None)
        self.assertEqual("deferred_inflight", result["action"])
        self.assertIsNone(result["priority"])

    def test_exhausted_released_cap_is_deferred(self):
        result = decide("unverified", 0.0, already=9)
        self.assertEqual("deferred_budget", result["action"])

    def test_every_action_is_recommendation_only(self):
        self.assertTrue(decide("likely_false", 0.9)["recommendation_only"])


if __name__ == "__main__":
    unittest.main()
