"""Smoke tests for the deterministic priority + selection layer
(intervention_policy.py) and the batch-verification retry/failure-marking
orchestration (intervention_agent.run_batch_verification). No real LLM,
network, or DB call -- run_batch_verification is exercised with an injected
stub verify_fn and db=None, per its own docstring, so this stays a short,
self-contained check (AGENTS.md Section 10/Assistant Execution policy),
not a long-running integration test.

v2: adds the 9 tests a read-only review found missing after v1 (agent_failure
weighting, unconfirmed low-confidence caching, cluster dedup, concurrent-
session isolation, TTL expiry, tiered recommendations) -- see
intervention_policy.py's and intervention_agent.py's module docstrings for
what each fix addresses.
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "postgresql://admin:secret@localhost:15432/misinfo_db")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import intervention_policy as policy
from app.services import intervention_agent
from app.services.verification_inflight_lock import advisory_lock_key


class InFlightLockKeyTest(unittest.TestCase):
    def test_key_is_deterministic_signed_bigint(self) -> None:
        first = advisory_lock_key("cluster-123")
        self.assertEqual(first, advisory_lock_key("cluster-123"))
        self.assertGreaterEqual(first, -(2**63))
        self.assertLess(first, 2**63)

    def test_different_cluster_or_namespace_changes_key(self) -> None:
        key = advisory_lock_key("cluster-123")
        self.assertNotEqual(key, advisory_lock_key("cluster-456"))
        self.assertNotEqual(key, advisory_lock_key("cluster-123", namespace="different-purpose"))


class ConfirmedTrueExclusionTest(unittest.TestCase):
    def test_confirmed_true_excluded_from_selection(self) -> None:
        candidates = [
            policy.Candidate("p1", "c1", predicted_impact=10.0, credibility="likely_true", confidence=0.9),
            policy.Candidate("p2", "c2", predicted_impact=1.0, credibility="unverified", confidence=0.0),
        ]
        result = policy.select_for_checkpoint(candidates, checkpoint_minutes=10, already_intervened_count=0, total_budget=50)
        self.assertEqual(["p2"], [c.post_uri for c in result.selected])
        self.assertEqual(["p1"], result.excluded_confirmed_true)

    def test_low_confidence_true_is_not_excluded(self) -> None:
        candidates = [policy.Candidate("p1", "c1", predicted_impact=5.0, credibility="likely_true", confidence=0.4)]
        result = policy.select_for_checkpoint(candidates, checkpoint_minutes=10, already_intervened_count=0, total_budget=50)
        self.assertEqual(["p1"], [c.post_uri for c in result.selected])

    def test_deferred_inflight_never_enters_ranking(self) -> None:
        candidates = [
            policy.Candidate("locked", "c1", predicted_impact=1000.0, credibility="deferred_inflight", confidence=None),
            policy.Candidate("ready", "c2", predicted_impact=1.0, credibility="unverified", confidence=0.0),
        ]
        result = policy.select_for_checkpoint(
            candidates, checkpoint_minutes=10, already_intervened_count=0, total_budget=50,
        )
        self.assertEqual(["ready"], [candidate.post_uri for candidate in result.selected])
        self.assertEqual(["locked"], result.deferred_inflight)
        self.assertNotIn("locked", result.priority_by_uri)


class AgentFailureExclusionTest(unittest.TestCase):
    def test_agent_failure_never_selected_this_round(self) -> None:
        candidates = [
            policy.Candidate("p1", "c1", predicted_impact=100.0, credibility="agent_failure", confidence=None),
            policy.Candidate("p2", "c2", predicted_impact=1.0, credibility="unverified", confidence=0.0),
        ]
        result = policy.select_for_checkpoint(candidates, checkpoint_minutes=10, already_intervened_count=0, total_budget=50)
        self.assertEqual(["p2"], [c.post_uri for c in result.selected])
        self.assertEqual(["p1"], result.deferred_agent_failure)
        self.assertNotIn("p1", result.priority_by_uri, "agent_failure must never receive a priority score")

    def test_agent_failure_outranking_predicted_impact_still_excluded(self) -> None:
        # Even a huge predicted_impact must not let an agent_failure candidate in.
        candidates = [policy.Candidate("p1", "c1", predicted_impact=1000.0, credibility="agent_failure", confidence=None)]
        result = policy.select_for_checkpoint(candidates, checkpoint_minutes=10, already_intervened_count=0, total_budget=50)
        self.assertEqual([], result.selected)
        self.assertEqual(["p1"], result.deferred_agent_failure)


class ConfidenceGatedResolutionTest(unittest.TestCase):
    def test_low_confidence_likely_false_not_resolved(self) -> None:
        record = policy.record_check(None, checkpoint_minutes=10, credibility="likely_false", confidence=0.2, is_high_risk=True)
        self.assertFalse(record.resolved)
        self.assertEqual("unverified", record.last_status)
        self.assertTrue(policy.needs_check(record, checkpoint_minutes=20))

    def test_low_confidence_likely_true_not_resolved(self) -> None:
        record = policy.record_check(None, checkpoint_minutes=10, credibility="likely_true", confidence=0.3, is_high_risk=True)
        self.assertFalse(record.resolved)
        self.assertEqual("unverified", record.last_status)
        self.assertTrue(policy.needs_check(record, checkpoint_minutes=20))

    def test_high_confidence_likely_false_resolved(self) -> None:
        record = policy.record_check(None, checkpoint_minutes=10, credibility="likely_false", confidence=0.9, is_high_risk=True)
        self.assertTrue(record.resolved)
        self.assertEqual("likely_false", record.last_status)

    def test_low_confidence_false_still_eligible_but_weighted_as_unverified_not_disputed(self) -> None:
        candidates = [policy.Candidate("p1", "c1", predicted_impact=10.0, credibility="likely_false", confidence=0.2)]
        result = policy.select_for_checkpoint(candidates, checkpoint_minutes=10, already_intervened_count=0, total_budget=50)
        self.assertEqual(10.0 * policy.RISK_WEIGHTS["unverified"], result.priority_by_uri["p1"])

    def test_low_confidence_true_does_not_outrank_unverified(self) -> None:
        # The bug this guards against: disputed (0.75) > unverified (0.5) would have
        # let a weak "probably true" verdict outrank genuinely-unknown content.
        weak_true_priority = policy.priority_for(10.0, "likely_true", 0.3)
        unverified_priority = policy.priority_for(10.0, "unverified", 0.0)
        self.assertEqual(weak_true_priority, unverified_priority)


class ResolvedTTLTest(unittest.TestCase):
    def test_resolved_record_not_rechecked_within_ttl(self) -> None:
        record = policy.record_check(None, checkpoint_minutes=10, credibility="likely_false", confidence=0.9, is_high_risk=True)
        self.assertFalse(policy.needs_check(record, checkpoint_minutes=10 + policy.RESOLVED_TTL_MINUTES - 10))

    def test_resolved_record_rechecked_after_ttl_expires(self) -> None:
        record = policy.record_check(None, checkpoint_minutes=10, credibility="likely_false", confidence=0.9, is_high_risk=True)
        self.assertTrue(policy.needs_check(record, checkpoint_minutes=10 + policy.RESOLVED_TTL_MINUTES + 1))


class ReleasedCapTest(unittest.TestCase):
    def test_cumulative_caps_match_balanced_profile(self) -> None:
        expected = [9, 18, 26, 34, 42, 50]
        actual = [policy.released_cap(m, total_budget=50) for m in policy.CHECKPOINTS_MINUTES]
        self.assertEqual(expected, actual)

    def test_never_exceeds_cap_even_with_many_eligible(self) -> None:
        candidates = [
            policy.Candidate(f"p{i}", f"c{i}", predicted_impact=float(100 - i), credibility="unverified", confidence=0.0)
            for i in range(30)
        ]
        result = policy.select_for_checkpoint(candidates, checkpoint_minutes=10, already_intervened_count=0, total_budget=50)
        self.assertLessEqual(len(result.selected), 9)

    def test_unused_capacity_carries_forward_no_forced_fill(self) -> None:
        first_batch = [
            policy.Candidate("p1", "c1", predicted_impact=5.0, credibility="unverified", confidence=0.0),
            policy.Candidate("p2", "c2", predicted_impact=4.0, credibility="unverified", confidence=0.0),
        ]
        result1 = policy.select_for_checkpoint(first_batch, checkpoint_minutes=10, already_intervened_count=0, total_budget=50)
        self.assertEqual(2, len(result1.selected))
        second_batch = [
            policy.Candidate(f"q{i}", f"cq{i}", predicted_impact=float(10 - i), credibility="unverified", confidence=0.0)
            for i in range(20)
        ]
        result2 = policy.select_for_checkpoint(second_batch, checkpoint_minutes=20, already_intervened_count=len(result1.selected), total_budget=50)
        self.assertEqual(16, len(result2.selected))

    def test_no_duplicate_selection_across_checkpoints(self) -> None:
        candidates_cp10 = [policy.Candidate("p1", "c1", predicted_impact=5.0, credibility="unverified", confidence=0.0)]
        result1 = policy.select_for_checkpoint(candidates_cp10, checkpoint_minutes=10, already_intervened_count=0, total_budget=50)
        selected_ids = {c.post_uri for c in result1.selected}
        candidates_cp20 = [c for c in candidates_cp10 if c.post_uri not in selected_ids]
        result2 = policy.select_for_checkpoint(candidates_cp20, checkpoint_minutes=20, already_intervened_count=len(result1.selected), total_budget=50)
        self.assertEqual(set(), {c.post_uri for c in result2.selected} & selected_ids)


class RecommendationTierTest(unittest.TestCase):
    def test_unverified_gets_soft_tier(self) -> None:
        candidates = [policy.Candidate("p1", "c1", predicted_impact=10.0, credibility="unverified", confidence=0.0)]
        result = policy.select_for_checkpoint(candidates, checkpoint_minutes=10, already_intervened_count=0, total_budget=50)
        self.assertEqual("soft", result.tier_by_uri["p1"])

    def test_confirmed_false_gets_hard_tier(self) -> None:
        candidates = [policy.Candidate("p1", "c1", predicted_impact=10.0, credibility="likely_false", confidence=0.9)]
        result = policy.select_for_checkpoint(candidates, checkpoint_minutes=10, already_intervened_count=0, total_budget=50)
        self.assertEqual("hard", result.tier_by_uri["p1"])

    def test_unconfirmed_low_confidence_false_gets_soft_tier_not_hard(self) -> None:
        candidates = [policy.Candidate("p1", "c1", predicted_impact=10.0, credibility="likely_false", confidence=0.2)]
        result = policy.select_for_checkpoint(candidates, checkpoint_minutes=10, already_intervened_count=0, total_budget=50)
        self.assertEqual("soft", result.tier_by_uri["p1"])


class PriorityWeightTest(unittest.TestCase):
    def test_false_outranks_unverified_at_equal_predicted_impact(self) -> None:
        false_priority = policy.priority_for(10.0, "likely_false", 0.9)
        unverified_priority = policy.priority_for(10.0, "unverified", 0.0)
        self.assertGreater(false_priority, unverified_priority)

    def test_confirmed_true_returns_none_via_exclusion_path(self) -> None:
        self.assertTrue(policy.is_confirmed_true("likely_true", 0.9))
        self.assertFalse(policy.is_confirmed_true("likely_true", 0.5))
        self.assertFalse(policy.is_confirmed_true("likely_false", 0.9))


class CacheRecordTest(unittest.TestCase):
    def test_needs_check_true_for_never_seen_thread(self) -> None:
        self.assertTrue(policy.needs_check(None, checkpoint_minutes=10))

    def test_resolved_thread_within_ttl_never_needs_recheck(self) -> None:
        record = policy.record_check(None, checkpoint_minutes=10, credibility="likely_false", confidence=0.9, is_high_risk=True)
        self.assertTrue(record.resolved)
        self.assertFalse(policy.needs_check(record, checkpoint_minutes=60))

    def test_unresolved_high_risk_recheck_next_checkpoint(self) -> None:
        record = policy.record_check(None, checkpoint_minutes=10, credibility="unverified", confidence=0.0, is_high_risk=True)
        self.assertFalse(policy.needs_check(record, checkpoint_minutes=10))
        self.assertTrue(policy.needs_check(record, checkpoint_minutes=20))

    def test_unresolved_low_risk_recheck_skips_a_checkpoint(self) -> None:
        record = policy.record_check(None, checkpoint_minutes=10, credibility="unverified", confidence=0.0, is_high_risk=False)
        self.assertFalse(policy.needs_check(record, checkpoint_minutes=20))
        self.assertTrue(policy.needs_check(record, checkpoint_minutes=30))

    def test_status_version_bumps_only_on_effective_status_change(self) -> None:
        record = policy.record_check(None, checkpoint_minutes=10, credibility="unverified", confidence=0.0, is_high_risk=True)
        self.assertEqual(1, record.status_version)
        record = policy.record_check(record, checkpoint_minutes=20, credibility="unverified", confidence=0.0, is_high_risk=True)
        self.assertEqual(1, record.status_version, "unchanged status must not bump status_version")
        record = policy.record_check(record, checkpoint_minutes=30, credibility="likely_false", confidence=0.9, is_high_risk=True)
        self.assertEqual(2, record.status_version, "status change must bump status_version")


class BatchVerificationRetryTest(unittest.TestCase):
    """Exercises intervention_agent.run_batch_verification's retry-missing
    and agent_failure marking with an injected stub -- no real network,
    LLM, or DB call. db=None is passed throughout (the function's own
    contract: it must not require a DB session to run, and must not open
    one internally either -- see _verify_with_own_session's use_db guard)."""

    def test_all_succeed_first_try(self) -> None:
        async def stub(db, key, event_id, claim_text, post_date):
            return {"credibility": "likely_false", "confidence": 0.9, "summary": "stub"}

        candidates = [{"cluster_thread_id": f"c{i}", "claim_text": "x", "event_id": "e"} for i in range(3)]
        result = asyncio.run(intervention_agent.run_batch_verification(None, candidates, checkpoint_minutes=10, verify_fn=stub))
        self.assertEqual(set(f"c{i}" for i in range(3)), set(result["checked_now"]))
        self.assertEqual([], result["agent_failures"])

    def test_missing_candidate_retried_then_succeeds(self) -> None:
        attempts: dict[str, int] = {}

        async def flaky_stub(db, key, event_id, claim_text, post_date):
            attempts[key] = attempts.get(key, 0) + 1
            if key == "flaky" and attempts[key] == 1:
                raise RuntimeError("simulated transient failure")
            return {"credibility": "unverified", "confidence": 0.0, "summary": "stub"}

        candidates = [
            {"cluster_thread_id": "ok1", "claim_text": "x", "event_id": "e"},
            {"cluster_thread_id": "flaky", "claim_text": "x", "event_id": "e"},
        ]
        result = asyncio.run(intervention_agent.run_batch_verification(None, candidates, checkpoint_minutes=10, verify_fn=flaky_stub))
        self.assertEqual([], result["agent_failures"], "must recover after one retry")
        self.assertEqual(2, attempts["flaky"], "flaky candidate must be retried exactly once")
        self.assertEqual("unverified", result["results"]["flaky"]["credibility"])

    def test_permanently_failing_candidate_marked_agent_failure_not_hold(self) -> None:
        async def always_fails(db, key, event_id, claim_text, post_date):
            raise RuntimeError("simulated permanent failure")

        candidates = [{"cluster_thread_id": "dead", "claim_text": "x", "event_id": "e"}]
        result = asyncio.run(intervention_agent.run_batch_verification(None, candidates, checkpoint_minutes=10, verify_fn=always_fails))
        self.assertEqual(["dead"], result["agent_failures"])
        self.assertEqual("agent_failure", result["results"]["dead"]["credibility"])
        self.assertNotEqual("hold", result["results"]["dead"]["credibility"])
        self.assertNotEqual("unverified", result["results"]["dead"]["credibility"],
                             "a genuine failure must not be silently indistinguishable from a real unverified check")

    def test_only_missing_subset_is_retried_not_the_whole_batch(self) -> None:
        call_log: list[str] = []

        async def stub(db, key, event_id, claim_text, post_date):
            call_log.append(key)
            if key == "flaky" and call_log.count("flaky") == 1:
                raise RuntimeError("fail once")
            return {"credibility": "unverified", "confidence": 0.0, "summary": "stub"}

        candidates = [{"cluster_thread_id": k, "claim_text": "x", "event_id": "e"} for k in ("a", "b", "flaky", "d")]
        asyncio.run(intervention_agent.run_batch_verification(None, candidates, checkpoint_minutes=10, verify_fn=stub))
        self.assertEqual(5, len(call_log))
        self.assertEqual(1, call_log.count("a"))
        self.assertEqual(1, call_log.count("b"))
        self.assertEqual(1, call_log.count("d"))
        self.assertEqual(2, call_log.count("flaky"))

    def test_same_cluster_multiple_posts_verified_once(self) -> None:
        call_log: list[str] = []

        async def stub(db, key, event_id, claim_text, post_date):
            call_log.append(key)
            return {"credibility": "likely_false", "confidence": 0.9, "summary": "stub"}

        # Three posts (different post_uri) sharing one cluster_thread_id.
        candidates = [
            {"post_uri": "post1", "cluster_thread_id": "shared_story", "claim_text": "x", "event_id": "e"},
            {"post_uri": "post2", "cluster_thread_id": "shared_story", "claim_text": "x", "event_id": "e"},
            {"post_uri": "post3", "cluster_thread_id": "shared_story", "claim_text": "x", "event_id": "e"},
        ]
        result = asyncio.run(intervention_agent.run_batch_verification(None, candidates, checkpoint_minutes=10, verify_fn=stub))
        self.assertEqual(1, len(call_log), "one shared cluster_thread_id must be verified exactly once, not once per post")
        self.assertEqual("likely_false", result["results"]["shared_story"]["credibility"])

    def test_concurrent_calls_receive_independent_db_when_db_enabled(self) -> None:
        # db=None mode must never construct a real session -- verify_fn must
        # see db=None, not a SessionLocal() instance, when the parent had no DB.
        seen_db_args = []

        async def stub(db, key, event_id, claim_text, post_date):
            seen_db_args.append(db)
            return {"credibility": "unverified", "confidence": 0.0, "summary": "stub"}

        candidates = [{"cluster_thread_id": f"c{i}", "claim_text": "x", "event_id": "e"} for i in range(3)]
        asyncio.run(intervention_agent.run_batch_verification(None, candidates, checkpoint_minutes=10, verify_fn=stub))
        self.assertTrue(all(db_arg is None for db_arg in seen_db_args),
                         "db=None (offline/smoke mode) must never open a real DB session internally")


if __name__ == "__main__":
    unittest.main()
