"""Unit tests for verification_agent.py's static/prompt-level contracts and
the VERIFICATION_PROMPT_VERSION cache-key invalidation mechanism.

These deliberately do NOT call the real agent loop (no Ollama, no network) --
they check the things that are actually testable without a live LLM: that
the anti-mismatch reasoning rule is present in the prompt text (so it can't
be silently dropped in a future edit) and that bumping the version constant
actually changes every cache key derived from it.

Regression context: a real Bluesky post reporting a genuine, USGS-confirmed
M3.2 earthquake ("27 km N of Isabela, Puerto Rico") was verified via the
"query this exact text" flow and came back "likely_false" with a fabricated-
sounding refutation ("the closest recorded event was 59 km away on a
different date") -- the agent had found a DIFFERENT, nearby-but-distinct
earthquake record and treated the mismatch as a contradiction, instead of
recognizing it only proves its search tools (no direct USGS/authoritative
API access) didn't happen to surface the exact same event. See
SYSTEM_PROMPT's rule 6 and VERIFICATION_PROMPT_VERSION's own docstring.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.services import verification_agent
from app.routers import factcheck, radar


class DifferentEventIsNotRefutationPromptRuleTest(unittest.TestCase):
    """Requirement: the prompt must explicitly forbid treating a different-
    but-similar record (e.g. a nearby earthquake on a different date) as
    evidence the claimed event is false, and must explicitly steer that
    situation to "unverified" instead."""

    def test_prompt_forbids_treating_a_different_event_as_a_refutation(self) -> None:
        prompt = verification_agent.SYSTEM_PROMPT
        self.assertIn("NOT evidence the claimed event is false", prompt)
        self.assertIn("SAME specific event", prompt)

    def test_prompt_names_the_coverage_gap_reason(self) -> None:
        # The rule must explain WHY an exact match failing to turn up is not
        # itself suspicious -- otherwise a future edit could "fix" the
        # wording in a way that loses the actual reasoning.
        self.assertIn("no access to authoritative structured databases", verification_agent.SYSTEM_PROMPT)

    def test_prompt_still_has_the_pre_existing_zero_evidence_rule(self) -> None:
        # Guards against this edit accidentally clobbering the older,
        # separately-important "no tool results at all -> unverified" rule
        # instead of adding alongside it.
        self.assertIn('credibility MUST be "unverified"', verification_agent.SYSTEM_PROMPT)


class VerificationPromptVersionCacheKeyTest(unittest.TestCase):
    """VERIFICATION_PROMPT_VERSION must actually change every cache key
    derived from it -- both duplicated _verify_cache_key copies (radar.py's
    router-level one and intervention_agent.py's, which must stay identical
    to each other -- see their own comments) and factcheck.py's _cache_key.
    Bumping this constant is how a prompt/reasoning fix invalidates every
    VerificationReport produced under the old, since-fixed reasoning."""

    def test_factcheck_cache_key_changes_with_version(self) -> None:
        text = "same claim text"
        with patch.object(factcheck, "VERIFICATION_PROMPT_VERSION", "v_old"):
            key_old = factcheck._cache_key(text)
        with patch.object(factcheck, "VERIFICATION_PROMPT_VERSION", "v_new"):
            key_new = factcheck._cache_key(text)
        self.assertNotEqual(key_old, key_new)

    def test_radar_verify_cache_key_changes_with_version(self) -> None:
        thread_id = "some-thread-uuid"
        with patch.object(radar, "VERIFICATION_PROMPT_VERSION", "v_old"):
            key_old = radar._verify_cache_key(thread_id)
        with patch.object(radar, "VERIFICATION_PROMPT_VERSION", "v_new"):
            key_new = radar._verify_cache_key(thread_id)
        self.assertNotEqual(key_old, key_new)

    def test_radar_router_and_intervention_agent_derive_the_identical_key(self) -> None:
        # The two copies (duplicated to avoid a circular import -- see both
        # docstrings) must never drift apart, or is_confirmed_true / the
        # background scheduler would silently stop seeing manually-produced
        # verification reports (and vice versa).
        from app.services import intervention_agent
        thread_id = "some-thread-uuid"
        self.assertEqual(radar._verify_cache_key(thread_id), intervention_agent._verify_cache_key(thread_id))

    def test_version_is_a_non_empty_string(self) -> None:
        self.assertIsInstance(verification_agent.VERIFICATION_PROMPT_VERSION, str)
        self.assertTrue(verification_agent.VERIFICATION_PROMPT_VERSION)


if __name__ == "__main__":
    unittest.main()
