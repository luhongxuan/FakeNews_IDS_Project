"""Live Radar Event Discovery V2 -- unit tests (pure functions only, no DB,
no network, no real LLM call). Run directly as a script (its own directory
goes on sys.path -- see test_intervention_policy_smoke.py's own note about
this project's `python -m unittest app.services.X` path quirk) or via
`python -m unittest` from this directory.

Covers the 9 cases required by the V2 spec, plus the pure event-signature
extraction the merge/eligibility logic itself depends on.
"""
from __future__ import annotations

import inspect
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import event_clustering_v2 as v2
import radar_event_signature as sig
import radar_thread_quality as quality


def _signature(text: str) -> sig.EventSignature:
    return sig.extract_event_signature(text)


class LocationConflictBlocksMergeTest(unittest.TestCase):
    """Required cases 1 & 2: two posts about explicitly different real-world
    locations must never merge, even with high embedding similarity (both
    vectors here are IDENTICAL, i.e. best-case similarity=1.0, to isolate
    the entity-conflict veto from the embedding-similarity check)."""

    def _assert_blocked_by_location_conflict(self, text_a: str, text_b: str) -> None:
        vector = [1.0, 0.0]  # identical vectors -- similarity checks alone would always pass
        signature_a = _signature(text_a)
        signature_b = _signature(text_b)
        self.assertTrue(signature_a.locations, f"test fixture must extract a location from: {text_a!r}")
        self.assertTrue(signature_b.locations, f"test fixture must extract a location from: {text_b!r}")
        self.assertTrue(signature_a.locations.isdisjoint(signature_b.locations))

        decision = v2.classify_cluster_merge(
            candidate_vector=vector, candidate_signature=signature_b,
            cluster_centroid=vector, cluster_representative_vector=vector,
            cluster_signature=signature_a,
        )
        self.assertFalse(decision.merge)
        self.assertTrue(decision.hard_location_conflict)
        self.assertEqual(1.0, decision.similarity_to_centroid)
        self.assertEqual(1.0, decision.similarity_to_representative)

    def test_california_indiana_texas_shootings_cannot_merge(self) -> None:
        # Case 1. Real bug this guards against: v1 (pure centroid similarity, no entity
        # check) merged these three real, unrelated shootings into one RadarThread --
        # confirmed against real data by diagnose_radar_cluster_quality.py.
        sacramento = "Four people injured in a shooting near Granite Regional Park in Sacramento, California."
        indianapolis = "Indiana: 3 injured in a shooting on Indianapolis's near west side."
        pasadena_tx = "Mass shooting: shooter dead, 3 wounded in a shooting spree in Pasadena, Texas."
        self._assert_blocked_by_location_conflict(sacramento, indianapolis)
        self._assert_blocked_by_location_conflict(indianapolis, pasadena_tx)
        self._assert_blocked_by_location_conflict(sacramento, pasadena_tx)

    def test_different_state_flood_notices_cannot_merge(self) -> None:
        # Case 2. Real bug this guards against: a second real false merge
        # diagnose_radar_cluster_quality.py found -- a Georgia flash-flood report merged
        # with a North Carolina flood-advisory cancellation under one cluster.
        georgia = "At 11:00 PM EDT, 5 NNE Valdosta [Lowndes Co, GA] Public reports Flash Flood."
        north_carolina = "RNK cancels Flood Advisory for Wilkes [NC] at Mon, 07 Sep 2026 05:09:31 +0000 via IEMbot"
        self._assert_blocked_by_location_conflict(georgia, north_carolina)


class CompatibleDuplicatesCanMergeTest(unittest.TestCase):
    """Required cases 3 & 4: genuinely-the-same real incident, reworded by a
    different poster, must still be allowed to merge -- the entity-conflict
    veto must not be so aggressive it blocks legitimate duplicates."""

    def _assert_allowed_to_merge(self, text_a: str, text_b: str) -> None:
        vector = [1.0, 0.0]
        signature_a = _signature(text_a)
        signature_b = _signature(text_b)
        decision = v2.classify_cluster_merge(
            candidate_vector=vector, candidate_signature=signature_b,
            cluster_centroid=vector, cluster_representative_vector=vector,
            cluster_signature=signature_a,
        )
        self.assertTrue(decision.merge, f"expected a merge; reasons={decision.reasons}")
        self.assertFalse(decision.hard_location_conflict)
        self.assertFalse(decision.hard_incident_conflict)

    def test_same_payson_fire_reworded_can_merge(self) -> None:
        # Case 3 -- real data (both are actual member texts of the same real cluster).
        first = "Payson Fire burns near Elk Ridge, Utah, with evacuation updates and map."
        second = "Payson Fire: Map, evacuation updates as wildfire burns near Elk Ridge, Utah."
        self._assert_allowed_to_merge(first, second)

    def test_same_san_diego_mosque_incident_reworded_can_merge(self) -> None:
        # Case 4 -- real data (both are actual member texts of the same real cluster).
        first = "Parents of San Diego mosque gunman say a mental health facility failed to heed FBI warnings."
        second = "A federal warning goes unheeded just days before gunfire shatters a San Diego mosque."
        self._assert_allowed_to_merge(first, second)


class CentroidChainingCannotBypassEntityCheckTest(unittest.TestCase):
    """Required case 5: a candidate must not be admitted into a cluster
    purely because a DRIFTED centroid (from prior merges) happens to be
    close to it -- similarity to the cluster's own REPRESENTATIVE post is
    checked independently and must also clear the threshold."""

    def test_high_centroid_similarity_alone_is_not_enough(self) -> None:
        representative_vector = [1.0, 0.0]
        drifted_centroid = [0.3, 0.95]      # drifted away from the representative by prior merges
        candidate_vector = [0.2, 0.98]      # close to the drifted centroid, NOT close to the representative
        empty_signature = sig.EventSignature()

        similarity_to_centroid = v2.cosine_similarity(candidate_vector, drifted_centroid)
        similarity_to_representative = v2.cosine_similarity(candidate_vector, representative_vector)
        self.assertGreaterEqual(similarity_to_centroid, v2.SIMILARITY_THRESHOLD_V2, "test fixture must actually exercise the centroid-passes case")
        self.assertLess(similarity_to_representative, v2.SIMILARITY_THRESHOLD_V2, "test fixture must actually exercise the representative-fails case")

        decision = v2.classify_cluster_merge(
            candidate_vector=candidate_vector, candidate_signature=empty_signature,
            cluster_centroid=drifted_centroid, cluster_representative_vector=representative_vector,
            cluster_signature=empty_signature,
        )
        self.assertFalse(decision.merge, "centroid-only similarity must not be sufficient to admit a chained-in post")
        self.assertFalse(decision.hard_location_conflict)
        self.assertFalse(decision.hard_incident_conflict)


class SingletonDisplayEligibilityTest(unittest.TestCase):
    """Required case 6: a singleton (member_count=1) with a real reply
    cascade must still be display_eligible -- eligibility is not purely
    about member_count."""

    def test_singleton_with_enough_replies_is_display_eligible(self) -> None:
        thread = {"member_count": 1, "members": [{"uri": "a", "reply_count": 64, "repost_count": 0}]}
        result = quality.classify_radar_thread_quality(thread)
        self.assertEqual(quality.DISPLAY_ELIGIBLE, result["display_status"])
        self.assertEqual(64, result["reply_signal"])

    def test_singleton_with_low_signal_is_singleton_monitoring(self) -> None:
        thread = {"member_count": 1, "members": [{"uri": "a", "reply_count": 0, "repost_count": 0}],
                  "matched_keywords": ["breaking news"], "representative_text": "some generic post"}
        result = quality.classify_radar_thread_quality(thread)
        self.assertEqual(quality.SINGLETON_MONITORING, result["display_status"])

    def test_multi_member_is_always_display_eligible_regardless_of_engagement(self) -> None:
        thread = {"member_count": 2, "members": [
            {"uri": "a", "reply_count": 0, "repost_count": 0}, {"uri": "b", "reply_count": 0, "repost_count": 0},
        ]}
        result = quality.classify_radar_thread_quality(thread)
        self.assertEqual(quality.DISPLAY_ELIGIBLE, result["display_status"])


class SearchNoiseNotForceMergedTest(unittest.TestCase):
    """Required case 7: two posts that merely share a search keyword, with
    unrelated actual content, must not merge just because of that shared
    keyword -- clustering here never even looks at matched_keywords, only
    embeddings + entity signatures, so this also documents that fact
    directly."""

    def test_unrelated_posts_sharing_a_keyword_are_not_merged_by_keyword_alone(self) -> None:
        # Deliberately DISSIMILAR vectors (unlike the location-conflict tests above, which
        # used identical vectors to isolate the entity veto) -- the point here is that
        # classify_cluster_merge has no matched_keywords parameter at all, so two posts
        # that only share a search keyword and have low embedding similarity fail on
        # similarity alone, the same as any other unrelated pair would.
        basketball = "Steph Curry's 3-ball magic is REAL! On 4/20, the Warriors dropped 20 threes."
        gun_violence = "Three people are injured after a shooting on Indy's near west side."
        vector_a, vector_b = [1.0, 0.0], [0.0, 1.0]  # orthogonal -- cosine similarity 0.0
        decision = v2.classify_cluster_merge(
            candidate_vector=vector_b, candidate_signature=_signature(gun_violence),
            cluster_centroid=vector_a, cluster_representative_vector=vector_a,
            cluster_signature=_signature(basketball),
        )
        self.assertFalse(decision.merge)
        self.assertLess(decision.similarity_to_centroid, v2.SIMILARITY_THRESHOLD_V2)

    def test_classify_radar_thread_quality_flags_a_keyword_that_never_reappears_as_noise(self) -> None:
        thread = {
            "member_count": 1, "members": [{"uri": "a", "reply_count": 0, "repost_count": 0}],
            "matched_keywords": ["shooting"], "representative_text": "Grap me please! #topless #boudoir #akt",
        }
        result = quality.classify_radar_thread_quality(thread)
        self.assertEqual(quality.SEARCH_NOISE_CANDIDATE, result["display_status"])


class AmbiguousClusterVerificationCacheTest(unittest.TestCase):
    """Required case 8: an ambiguous cluster (its own members disagree on
    explicit location/incident-type signals) must be detected so a caller
    can route it to per-post verification instead of one shared cache
    entry -- see app/routers/radar.py's manual_policy_decision, which
    checks is_cluster_claim_ambiguous() before deciding the verification
    cache key. This test covers the pure detection function directly; the
    router-level wiring (does it actually pick a DIFFERENT cache key) is
    covered by the DB smoke check, not by a pure unit test."""

    def test_cluster_with_conflicting_member_locations_is_ambiguous(self) -> None:
        member_signatures = [
            _signature("Indiana: 3 injured in a shooting on Indianapolis's near west side."),
            _signature("Mass shooting: shooter dead, 3 wounded in a shooting spree in Pasadena, Texas."),
        ]
        self.assertTrue(v2.is_cluster_claim_ambiguous(member_signatures))

    def test_cluster_with_consistent_member_locations_is_not_ambiguous(self) -> None:
        member_signatures = [
            _signature("Payson Fire burns near Elk Ridge, Utah, with evacuation updates and map."),
            _signature("Payson Fire: Map, evacuation updates as wildfire burns near Elk Ridge, Utah."),
        ]
        self.assertFalse(v2.is_cluster_claim_ambiguous(member_signatures))


class InsufficientCorroboratingSignalTest(unittest.TestCase):
    """Regression tests for review round 2: shared_entity_signal was computed
    but never actually gated `merge` -- high similarity plus zero
    extractable location/incident-type on EITHER side used to merge on
    embedding similarity alone."""

    def test_similar_generic_phrasing_with_no_shared_entity_and_no_location_does_not_merge(self) -> None:
        # Neither signature is empty (both have an incident_type), but they share
        # NOTHING (no location, no incident-type conflict either since both are
        # "shooting", no shared named entity) -- previously this merged on embedding
        # similarity alone.
        candidate_text = "A shooting outside a downtown venue left three people hurt, police say."
        cluster_text = "Gunfire reported near a nightclub, multiple victims taken to hospital."
        candidate_signature = _signature(candidate_text)
        cluster_signature = _signature(cluster_text)
        self.assertFalse(candidate_signature.is_empty())
        self.assertFalse(cluster_signature.is_empty())
        vector = [1.0, 0.0]
        decision = v2.classify_cluster_merge(
            candidate_vector=vector, candidate_signature=candidate_signature,
            cluster_centroid=vector, cluster_representative_vector=vector,
            cluster_signature=cluster_signature,
        )
        self.assertFalse(decision.merge)
        self.assertTrue(decision.insufficient_corroborating_signal)
        self.assertFalse(decision.hard_location_conflict)
        self.assertFalse(decision.hard_incident_conflict)

    def test_both_signatures_fully_empty_falls_back_to_embedding_similarity(self) -> None:
        # Truly nothing to check on either side -- allowed to merge on embedding
        # similarity alone, same as before this fix (there is nothing more this
        # module can do without NER).
        empty = sig.EventSignature()
        self.assertTrue(empty.is_empty())
        vector = [1.0, 0.0]
        decision = v2.classify_cluster_merge(
            candidate_vector=vector, candidate_signature=empty,
            cluster_centroid=vector, cluster_representative_vector=vector,
            cluster_signature=empty,
        )
        self.assertTrue(decision.merge)
        self.assertFalse(decision.insufficient_corroborating_signal)

    def test_shared_entity_signal_still_overrides_insufficient_signal(self) -> None:
        first = "Payson Fire burns near Elk Ridge, Utah, with evacuation updates and map."
        second = "Payson Fire: Map, evacuation updates as wildfire burns near Elk Ridge, Utah."
        vector = [1.0, 0.0]
        decision = v2.classify_cluster_merge(
            candidate_vector=vector, candidate_signature=_signature(second),
            cluster_centroid=vector, cluster_representative_vector=vector,
            cluster_signature=_signature(first),
        )
        self.assertTrue(decision.merge)
        self.assertFalse(decision.insufficient_corroborating_signal)
        self.assertTrue(decision.shared_entity_signal)

    def test_shared_location_with_compatible_incident_type_is_corroboration_without_a_shared_phrase(self) -> None:
        # Fixed per review (round 3): "都是 shooting 且都是 Sacramento" must corroborate
        # even with no capitalized-phrase overlap at all -- neither text below shares a
        # named entity/distinctive phrase, but both explicitly say Sacramento AND shooting.
        first = "A shooting was reported in Sacramento, California this evening."
        second = "Sacramento, California police are investigating a shooting near downtown."
        candidate_signature = _signature(second)
        cluster_signature = _signature(first)
        self.assertFalse(candidate_signature.named_entities & cluster_signature.named_entities)
        self.assertFalse(candidate_signature.distinctive_phrases & cluster_signature.distinctive_phrases)
        vector = [1.0, 0.0]
        decision = v2.classify_cluster_merge(
            candidate_vector=vector, candidate_signature=candidate_signature,
            cluster_centroid=vector, cluster_representative_vector=vector,
            cluster_signature=cluster_signature,
        )
        self.assertTrue(decision.merge)
        self.assertTrue(decision.corroboration_signal)
        self.assertFalse(decision.shared_entity_signal)

    def test_incident_type_agreement_alone_is_not_corroboration(self) -> None:
        # "都是 shooting" alone (no location, no entity, no domain, no hashtag in common)
        # must NOT count as corroboration -- it is the weakest possible signal and would
        # make this check nearly always pass if allowed.
        first = "A shooting was reported this evening, police say."
        second = "Witnesses described a shooting near a local business."
        vector = [1.0, 0.0]
        decision = v2.classify_cluster_merge(
            candidate_vector=vector, candidate_signature=_signature(second),
            cluster_centroid=vector, cluster_representative_vector=vector,
            cluster_signature=_signature(first),
        )
        self.assertFalse(decision.corroboration_signal)
        self.assertTrue(decision.insufficient_corroborating_signal)
        self.assertFalse(decision.merge)

    def test_shared_domain_is_corroboration(self) -> None:
        first = "Details here: https://news.example.com/incident-1"
        second = "More on this developing story: https://news.example.com/incident-2"
        vector = [1.0, 0.0]
        decision = v2.classify_cluster_merge(
            candidate_vector=vector, candidate_signature=_signature(second),
            cluster_centroid=vector, cluster_representative_vector=vector,
            cluster_signature=_signature(first),
        )
        self.assertTrue(decision.corroboration_signal)
        self.assertTrue(decision.merge)

    def test_shared_hashtag_is_corroboration(self) -> None:
        first = "Stay safe out there #PaysonFire"
        second = "Evacuation orders lifted #PaysonFire"
        vector = [1.0, 0.0]
        decision = v2.classify_cluster_merge(
            candidate_vector=vector, candidate_signature=_signature(second),
            cluster_centroid=vector, cluster_representative_vector=vector,
            cluster_signature=_signature(first),
        )
        self.assertTrue(decision.corroboration_signal)
        self.assertTrue(decision.merge)


class AmbiguousClusterNotAutoDisplayEligibleTest(unittest.TestCase):
    """Regression test for review round 2: an ambiguous multi-member cluster
    (v1 already mis-merged it) must not be promoted to display_eligible on
    member_count alone -- it must fall through to the same checks a
    singleton gets, and `ambiguous=True` must always be reported."""

    def test_ambiguous_multi_member_cluster_with_low_signal_is_not_display_eligible(self) -> None:
        thread = {
            "member_count": 2,
            "members": [
                {"uri": "a", "reply_count": 0, "repost_count": 0, "text": "Indiana: 3 injured in a shooting on Indianapolis's near west side."},
                {"uri": "b", "reply_count": 0, "repost_count": 0, "text": "Mass shooting: shooter dead, 3 wounded in a shooting spree in Pasadena, Texas."},
            ],
        }
        result = quality.classify_radar_thread_quality(thread)
        self.assertTrue(result["ambiguous"])
        self.assertEqual(quality.AMBIGUOUS_REVIEW, result["display_status"])

    def test_ambiguous_multi_member_cluster_is_ambiguous_review_even_with_high_engagement(self) -> None:
        # Fixed per review (round 3): a real reply cascade must NOT promote an ambiguous
        # cluster to display_eligible -- that would silently conflate it with a genuinely
        # consistent event. It is ALWAYS ambiguous_review, a status of its own, regardless
        # of engagement; callers decide whether/how to still surface it (with a warning).
        thread = {
            "member_count": 2,
            "members": [
                {"uri": "a", "reply_count": 50, "repost_count": 0, "text": "Indiana: 3 injured in a shooting on Indianapolis's near west side."},
                {"uri": "b", "reply_count": 0, "repost_count": 0, "text": "Mass shooting: shooter dead, 3 wounded in a shooting spree in Pasadena, Texas."},
            ],
        }
        result = quality.classify_radar_thread_quality(thread)
        self.assertTrue(result["ambiguous"])
        self.assertEqual(quality.AMBIGUOUS_REVIEW, result["display_status"])
        self.assertNotEqual(quality.DISPLAY_ELIGIBLE, result["display_status"])

    def test_non_ambiguous_multi_member_cluster_reports_ambiguous_false(self) -> None:
        thread = {
            "member_count": 2,
            "members": [
                {"uri": "a", "reply_count": 0, "repost_count": 0, "text": "Payson Fire burns near Elk Ridge, Utah, with evacuation updates and map."},
                {"uri": "b", "reply_count": 0, "repost_count": 0, "text": "Payson Fire: Map, evacuation updates as wildfire burns near Elk Ridge, Utah."},
            ],
        }
        result = quality.classify_radar_thread_quality(thread)
        self.assertFalse(result["ambiguous"])
        self.assertEqual(quality.DISPLAY_ELIGIBLE, result["display_status"])


class ExpansionQueryDeterminismTest(unittest.TestCase):
    """Regression test for review round 2: build_expansion_queries iterated
    sets directly (distinctive_phrases/casualty_expressions/hashtags),
    whose iteration order depends on Python's per-PROCESS hash seed, not
    guaranteed stable ACROSS processes/runs -- calling it 20x within one
    process (the same interpreter, one fixed hash seed) would NOT have
    caught the original bug, so this spawns two real subprocesses with
    DIFFERENT explicit PYTHONHASHSEED values and compares their output."""

    def test_same_signature_produces_the_same_query_list_across_different_hash_seeds(self) -> None:
        import json
        import subprocess

        script = (
            "import sys; sys.path.insert(0, r'" + str(Path(__file__).resolve().parent) + "'); "
            "sys.path.insert(0, r'" + str(Path(__file__).resolve().parents[2]) + "'); "
            "import json, radar_event_signature as sig; "
            "s = sig.EventSignature(incident_types={'flood'}, locations={'CO'}, "
            "distinctive_phrases={'Flood Advisory', 'Larimer County', 'Denver Metro'}, "
            "casualty_expressions={'2 injured', '5 evacuated'}, "
            "hashtags={'#flood', '#wx', '#COwx'}); "
            "print(json.dumps(sig.build_expansion_queries(s)))"
        )
        python = sys.executable
        results = []
        for seed in ("1", "12345"):
            proc = subprocess.run(
                [python, "-c", script], capture_output=True, text=True,
                env={**os.environ, "PYTHONHASHSEED": seed}, timeout=30,
            )
            self.assertEqual(0, proc.returncode, proc.stderr)
            results.append(json.loads(proc.stdout.strip()))
        self.assertEqual(results[0], results[1], "build_expansion_queries must not depend on PYTHONHASHSEED")


class PureFunctionsNeverTakeADbSessionTest(unittest.TestCase):
    """Required case 9 (a proxy for it): the V2 phase 2/3/4 decision
    functions this diagnostic-and-classification work added must be
    provably pure -- none of them accept (and therefore none of them can
    themselves perform I/O through) a DB session or ORM object, so calling
    them can never modify or delete an existing RadarThread row or any
    other experiment/data artifact. The actual "did a real diagnostic run
    modify anything" claim is additionally checked by the DB smoke test,
    which asserts specific rows are byte-identical before/after."""

    def test_no_pure_function_signature_mentions_db_or_session(self) -> None:
        pure_functions = [
            sig.extract_event_signature, sig.build_expansion_queries,
            v2.classify_cluster_merge, v2.is_cluster_claim_ambiguous, v2.cosine_similarity,
            quality.classify_radar_thread_quality,
        ]
        for fn in pure_functions:
            params = " ".join(inspect.signature(fn).parameters)
            self.assertNotIn("db", params.lower())
            self.assertNotIn("session", params.lower())


if __name__ == "__main__":
    unittest.main()
