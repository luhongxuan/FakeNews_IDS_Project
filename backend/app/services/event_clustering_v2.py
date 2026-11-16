"""Live Radar Event Discovery V2 -- versioned clustering DECISION logic.

Deliberately a NEW module, not an in-place edit of event_clustering.py's
`assign_post_to_cluster` (still v1, still embedding-centroid-only, still
used by the live ingestion path today -- see that module's own docstring).
V1 is left completely unmodified; nothing here deletes, reclassifies, or
re-merges an existing RadarThread row. This module only adds a decision
function that CAN be wired in later (a separate, explicit follow-up change)
once its threshold has been chosen against real data (see
diagnose_radar_cluster_quality.py's threshold-sensitivity table).

Why v1 alone is not enough (confirmed against real data, see the diagnostic
report): pure centroid-similarity clustering merged THREE unrelated
shootings (Sacramento CA / Indianapolis IN / Pasadena TX) into one cluster,
because short, formulaic breaking-news phrasing ("N injured in a shooting
in X") embeds similarly regardless of which real event it describes. This
is an OVER-merge risk, not under-merging -- see PROJECT_RESULTS note in the
diagnostic conclusion.md.

Two structural fixes over v1, both required together (see
`classify_cluster_merge`'s own docstring for why "either one alone" is not
enough):
  1. Compare the candidate to the cluster's REPRESENTATIVE post's own
     embedding, not only to the (potentially drifted) running centroid --
     prevents "centroid chaining": post C merging into a cluster whose
     centroid has drifted toward it through prior merges, despite C not
     actually being close to the cluster's own original representative
     post.
  2. A hard, symmetric entity/incident-type conflict veto, computed from
     event_signature.py's cutoff-independent EventSignature extraction --
     if both posts name an EXPLICIT but DIFFERENT location, or both are
     confidently typed into DIFFERENT, non-overlapping incident types,
     merging is refused regardless of how high the embedding similarity is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.services.radar_event_signature import EventSignature

SIMILARITY_THRESHOLD_V2 = 0.55  # same starting point as v1's SIMILARITY_THRESHOLD -- an
# ENGINEERING default carried over unchanged, NOT a value chosen by validating against a
# labeled ground truth (there is no ground-truth "same real-world event" label available
# for live Bluesky posts). See diagnose_radar_cluster_quality.py's threshold-sensitivity
# table (0.55/0.60/0.65/0.70/0.75) for how singleton rate, multi-member cluster count, and
# rule-flagged false-merge count trade off at each value -- that table is descriptive of
# this specific pull of real data, not a claim that any one value is "correct".


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    a, b = np.array(vector_a), np.array(vector_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class MergeDecisionV2:
    merge: bool
    similarity_to_centroid: float
    similarity_to_representative: float
    hard_location_conflict: bool
    hard_incident_conflict: bool
    shared_entity_signal: bool
    corroboration_signal: bool
    insufficient_corroborating_signal: bool
    reasons: list[str] = field(default_factory=list)


def classify_cluster_merge(
    candidate_vector: list[float],
    candidate_signature: EventSignature,
    cluster_centroid: list[float],
    cluster_representative_vector: list[float],
    cluster_signature: EventSignature,
    similarity_threshold: float = SIMILARITY_THRESHOLD_V2,
) -> MergeDecisionV2:
    """Pure function -- every embedding vector and EventSignature must
    already be computed by the caller (see event_clustering.embed and
    radar_event_signature.extract_event_signature). No DB access, no model
    load, so this is directly and cheaply unit-testable with synthetic
    vectors/signatures.

    merge = (similarity_to_centroid >= threshold)
        AND (similarity_to_representative >= threshold)   -- anti-chaining
        AND NOT (hard_location_conflict OR hard_incident_conflict)
        AND NOT insufficient_corroborating_signal

    Requiring BOTH similarity checks (not just centroid) is the anti-
    chaining fix: a centroid drifts toward whatever has already been
    merged into it, so a post could clear the centroid-similarity bar
    purely by resembling an EARLIER merge, without ever being similar to
    the cluster's own original representative post. Requiring
    similarity_to_representative >= threshold too closes that gap -- see
    CentroidChainingCannotBypassEntityCheckTest for the regression case
    this specifically guards against.

    A hard conflict is symmetric and location/incident-type only (not
    named-entity mismatch alone -- two genuinely-the-same-story posts
    routinely name different secondary people/quotes without that meaning
    they're different events, so named-entity overlap alone is recorded as
    a supporting `shared_entity_signal`, never a veto by itself). Only
    fires when BOTH sides have an explicit, non-empty signal that turns
    out to be fully disjoint -- an empty/undetected signal on either side
    is treated as "unknown", never as evidence of a conflict.

    Fixed per review (round 2): `shared_entity_signal` was computed but
    never actually GATED merge -- a candidate and cluster with high
    embedding similarity and no extractable named entity/phrase overlap on
    EITHER side could merge on embedding similarity alone, with no
    corroborating signal at all. Fixed via `insufficient_corroborating_
    signal`, gated on the BROADER `corroboration_signal` (round-3 review
    fix -- the round-2 version only checked named_entities/distinctive_
    phrases, which misses two posts that plainly agree on other signals,
    e.g. same explicit location + same incident type, or the same
    hashtag/domain, without sharing a specific capitalized phrase):

    corroboration_signal = shared named entity / distinctive phrase
        OR shared external domain
        OR shared specific hashtag
        OR (overlapping explicit location AND compatible incident type)

    "都是 shooting" alone is NOT corroboration (incident type compatible
    but no location signal) -- "都是 shooting 且都是 Sacramento" IS (an
    actual location in common, not just "not conflicting"). Plain
    incident-type-only agreement is intentionally excluded: it is the
    weakest possible signal (most breaking-news posts share SOME incident
    type with most others), so counting it alone would make this check
    nearly always pass, defeating its purpose.

    `insufficient_corroborating_signal` fires when NEITHER signature is
    fully empty (EventSignature.is_empty()) AND no hard conflict already
    decided the question AND there is no `corroboration_signal` -- i.e.
    "I extracted SOMETHING from at least one side, but found nothing in
    common", treated as NOT enough to merge on embedding similarity alone.
    Only when BOTH sides are fully empty (truly nothing to check) does
    this fall back to trusting embedding similarity alone -- deliberately
    erring toward under-merging over the false-merge risk this whole
    module exists to reduce.
    """
    similarity_to_centroid = cosine_similarity(candidate_vector, cluster_centroid)
    similarity_to_representative = cosine_similarity(candidate_vector, cluster_representative_vector)

    hard_location_conflict = bool(
        candidate_signature.locations and cluster_signature.locations
        and candidate_signature.locations.isdisjoint(cluster_signature.locations)
    )
    hard_incident_conflict = bool(
        candidate_signature.incident_types and cluster_signature.incident_types
        and candidate_signature.incident_types.isdisjoint(cluster_signature.incident_types)
    )
    shared_entity_signal = bool(
        (candidate_signature.named_entities & cluster_signature.named_entities)
        or (candidate_signature.distinctive_phrases & cluster_signature.distinctive_phrases)
    )
    shared_domain = bool(candidate_signature.domains & cluster_signature.domains)
    shared_hashtag = bool(candidate_signature.hashtags & cluster_signature.hashtags)
    shared_location_with_compatible_incident = bool(
        candidate_signature.locations & cluster_signature.locations  # an ACTUAL overlap, not just "not disjoint"
        and candidate_signature.incident_types and cluster_signature.incident_types
        and not candidate_signature.incident_types.isdisjoint(cluster_signature.incident_types)
    )
    corroboration_signal = shared_entity_signal or shared_domain or shared_hashtag or shared_location_with_compatible_incident
    both_signatures_empty = candidate_signature.is_empty() and cluster_signature.is_empty()
    insufficient_corroborating_signal = (
        not hard_location_conflict and not hard_incident_conflict
        and not corroboration_signal and not both_signatures_empty
    )

    reasons: list[str] = []
    if similarity_to_centroid < similarity_threshold:
        reasons.append(f"similarity_to_centroid {similarity_to_centroid:.3f} < threshold {similarity_threshold}")
    if similarity_to_representative < similarity_threshold:
        reasons.append(f"similarity_to_representative {similarity_to_representative:.3f} < threshold {similarity_threshold}")
    if hard_location_conflict:
        reasons.append(f"conflicting explicit locations: {sorted(candidate_signature.locations)} vs {sorted(cluster_signature.locations)}")
    if hard_incident_conflict:
        reasons.append(f"conflicting incident types: {sorted(candidate_signature.incident_types)} vs {sorted(cluster_signature.incident_types)}")
    if shared_entity_signal:
        reasons.append("shared named entity / distinctive phrase")
    if shared_domain:
        reasons.append(f"shared external domain: {sorted(candidate_signature.domains & cluster_signature.domains)}")
    if shared_hashtag:
        reasons.append(f"shared hashtag: {sorted(candidate_signature.hashtags & cluster_signature.hashtags)}")
    if shared_location_with_compatible_incident:
        reasons.append(
            f"shared explicit location {sorted(candidate_signature.locations & cluster_signature.locations)} "
            "with a compatible incident type"
        )
    if insufficient_corroborating_signal:
        reasons.append(
            "no corroboration_signal and no hard conflict either -- extracted SOME signal from at least one "
            "side but found nothing in common to corroborate the embedding similarity; refusing to merge on "
            "embedding similarity alone (see insufficient_corroborating_signal)"
        )

    merge = (
        similarity_to_centroid >= similarity_threshold
        and similarity_to_representative >= similarity_threshold
        and not hard_location_conflict
        and not hard_incident_conflict
        and not insufficient_corroborating_signal
    )
    if merge and not reasons:
        reasons.append(
            "similarity to both centroid and representative met the threshold; no entity/incident conflict "
            "detected; both signatures were empty, so embedding similarity alone was the only signal available"
        )

    return MergeDecisionV2(
        merge=merge,
        similarity_to_centroid=similarity_to_centroid,
        similarity_to_representative=similarity_to_representative,
        hard_location_conflict=hard_location_conflict,
        hard_incident_conflict=hard_incident_conflict,
        insufficient_corroborating_signal=insufficient_corroborating_signal,
        shared_entity_signal=shared_entity_signal,
        corroboration_signal=corroboration_signal,
        reasons=reasons,
    )


def is_cluster_claim_ambiguous(member_signatures: list[EventSignature]) -> bool:
    """True if a cluster's own members' signatures disagree enough that
    treating the cluster as ONE claim (one shared verification result, one
    representative claim_text) would be unsafe -- see
    manual_intervention_mvp.py / intervention_agent.run_batch_verification
    for where this must force PER-POST verification instead of a
    cluster-shared one (phase 5, decision safety).

    Conservative on purpose: only flags ambiguity on the same hard signals
    `classify_cluster_merge` vetoes on (explicit, disjoint locations or
    incident types across the cluster's own members) -- a cluster that
    passed v1's single-centroid-similarity gate could still contain such a
    conflict (v1 has no entity check at all), which is exactly the
    real-world case this function exists to catch after the fact for
    ALREADY-formed v1 clusters, not only to prevent new v2 merges.
    """
    locations_seen: set[str] = set()
    incident_types_seen: set[str] = set()
    for signature in member_signatures:
        if signature.locations and locations_seen and signature.locations.isdisjoint(locations_seen):
            return True
        if signature.incident_types and incident_types_seen and signature.incident_types.isdisjoint(incident_types_seen):
            return True
        locations_seen |= signature.locations
        incident_types_seen |= signature.incident_types
    return False
