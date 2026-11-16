"""Offline orchestration smoke tests for the manual live-policy endpoint.

These tests deliberately replace Bluesky, RF inference, and verification with
stubs.  They verify that the real router wiring persists a recommendation and
never require network, an LLM, or a running database.
"""

from __future__ import annotations

import asyncio
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.models.schema import InterventionDecision, RadarThread
from app.routers import radar
from app.routers.radar import _verify_cache_key


class _FakeQuery:
    def __init__(self, first_value=None, all_value=None):
        self._first_value = first_value
        self._all_value = all_value or []

    def filter_by(self, **_kwargs):
        return self

    def filter(self, *_args):
        return self

    def order_by(self, *_args):
        return self

    def first(self):
        return self._first_value

    def all(self):
        return list(self._all_value)


class _FakeSession:
    def __init__(self, thread):
        self.thread = thread
        self.added = []
        self.commit_count = 0

    def query(self, model):
        if model is RadarThread:
            return _FakeQuery(first_value=self.thread)
        if model is InterventionDecision:
            return _FakeQuery(first_value=None, all_value=[])
        raise AssertionError(f"Unexpected query model: {model}")

    def add(self, value):
        self.added.append(value)

    def commit(self):
        self.commit_count += 1

    def refresh(self, value):
        value.id = value.id or uuid.uuid4()
        value.decided_at = value.decided_at or datetime.now(timezone.utc)


def _radar_thread():
    created_at = datetime.now(timezone.utc) - timedelta(minutes=31)
    return SimpleNamespace(
        id=uuid.uuid4(),
        representative_text="A public claim used by the offline smoke test.",
        representative_uri="at://did:plc:test/app.bsky.feed.post/one",
        representative_created_at=created_at,
        members=[],
    )


class ManualPolicyEndpointSmokeTest(unittest.TestCase):
    def _run_with_evidence(self, evidence):
        thread = _radar_thread()
        db = _FakeSession(thread)
        # Fixed per review (Live Radar Event Discovery V2, phase 5): manual_policy_decision
        # now looks up verification_batch["results"] by _verify_cache_key(thread_id) (the
        # SAME hashed key every other verification pathway in radar.py/intervention_agent.py
        # uses -- is_confirmed_true, get_verification_status_for, the /verify endpoint),
        # not the raw thread_id string this mock previously used. This thread has no
        # members, so it is never ambiguous (see event_clustering_v2.is_cluster_claim_
        # ambiguous on an empty list) -- the cluster-level key is what's actually looked up.
        cache_key = _verify_cache_key(str(thread.id))
        verification = {
            "results": {cache_key: evidence},
            "checked_now": [cache_key],
        }
        rf_result = {
            "predicted_impact": 12.0,
            "predicted_log1p_impact": 2.56,
            "top_features": [{"name": "reply_velocity", "value": 1.5}],
        }
        run_batch_verification_mock = AsyncMock(return_value=verification)
        with (
            patch.object(radar.bluesky_source, "get_post_thread", new=AsyncMock(return_value={"thread": {}})),
            patch.object(radar.bluesky_source, "build_cascade_graph", return_value={"nodes": [{"id": "root"}], "edges": []}),
            patch.object(radar.intervention_agent, "score_thread_at_checkpoint", return_value=rf_result),
            patch.object(radar.intervention_agent, "run_batch_verification", new=run_batch_verification_mock),
        ):
            response = asyncio.run(
                radar.manual_policy_decision(
                    str(thread.id), thread.representative_uri, False, db,
                )
            )
        # Regression check for the actual bug fix: run_batch_verification must be called
        # with the SAME hashed cluster_thread_id every other verification pathway uses,
        # never the raw thread_id.
        called_input = run_batch_verification_mock.call_args.args[1]
        self.assertEqual(cache_key, called_input[0]["cluster_thread_id"])
        return response, db

    def test_confirmed_false_becomes_persisted_hard_recommendation(self):
        response, db = self._run_with_evidence({
            "credibility": "likely_false",
            "confidence": 0.95,
            "summary": "Reliable evidence directly refutes the claim.",
        })

        self.assertEqual(response["action"], "hard")
        self.assertEqual(response["action_strength"], "75-100%")
        self.assertTrue(response["recommendation_only"])
        self.assertEqual(response["predicted_reach"], 12.0)
        self.assertEqual(db.commit_count, 1)
        self.assertEqual(len(db.added), 1)
        self.assertEqual(db.added[0].action, "hard")
        self.assertTrue(db.added[0].report_jsonb["recommendation_only"])

    def test_confirmed_true_is_persisted_as_no_intervention(self):
        response, db = self._run_with_evidence({
            "credibility": "likely_true",
            "confidence": 0.94,
            "summary": "Reliable evidence supports the claim.",
        })

        self.assertEqual(response["action"], "none")
        self.assertEqual(response["action_strength"], "0%")
        self.assertEqual(db.added[0].action, "none")


if __name__ == "__main__":
    unittest.main()
