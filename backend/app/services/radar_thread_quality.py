"""Live Radar Event Discovery V2 -- display-eligibility classification
(phase 2). Pure function, no DB access -- takes a plain dict shaped like
app/routers/radar.py's `_serialize(thread)` output (or any dict with the
same `member_count`/`members`/`total_engagement` fields) and returns a
classification. Never modifies or deletes the RadarThread it was given;
the caller decides what (if anything) to do with the result.

The Radar API itself keeps returning every RadarThread regardless of this
classification -- this module only exists to let a FRONTEND VIEW decide
what to show by default (see InterventionReviewPanel.jsx's "真實事件預覽"
tab) versus what to leave available behind an explicit "show singletons"
expansion, per the V2 spec's phase 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.services.event_clustering_v2 import is_cluster_claim_ambiguous
from app.services.radar_event_signature import INCIDENT_TYPE_KEYWORDS, extract_event_signature

DISPLAY_ELIGIBLE = "display_eligible"
SINGLETON_MONITORING = "singleton_monitoring"
SEARCH_NOISE_CANDIDATE = "search_noise_candidate"
AMBIGUOUS_REVIEW = "ambiguous_review"

# ENGINEERING display thresholds ONLY -- chosen to be a visibly reasonable cutoff for a
# demo/ops view, NOT a value optimized or validated against any labeled ground truth
# (there is no "this cluster was actually worth showing" label for live Bluesky data).
# Configurable per call; the module-level defaults below are what the frontend uses
# unless told otherwise.
DEFAULT_MIN_REPLY_SIGNAL = 5
DEFAULT_MIN_TOTAL_ENGAGEMENT = 15

# Reverse of radar_event_signature.INCIDENT_TYPE_KEYWORDS: which matched-keyword search
# term (bluesky_source.DEFAULT_KEYWORDS) belongs to which incident_type bucket. Keywords
# with no clean incident-type mapping (e.g. "breaking news" -- deliberately generic) are
# left unmapped, and a thread whose ONLY matched keywords are unmapped never gets the
# search_noise_candidate label -- there's no incident-type signal to compare against.
_KEYWORD_TO_INCIDENT_TYPE: dict[str, str] = {}
for _incident_type, _phrases in INCIDENT_TYPE_KEYWORDS.items():
    for _phrase in _phrases:
        _KEYWORD_TO_INCIDENT_TYPE.setdefault(_phrase, _incident_type)
# bluesky_source.DEFAULT_KEYWORDS entries that don't literally appear in
# INCIDENT_TYPE_KEYWORDS' own phrase lists but obviously mean the same thing:
_KEYWORD_TO_INCIDENT_TYPE.setdefault("wildfire evacuation", "wildfire")


@dataclass
class RadarThreadQuality:
    display_status: str
    member_count: int
    reply_signal: int
    total_engagement: int
    reasons: list[str] = field(default_factory=list)
    ambiguous: bool = False

    def as_dict(self) -> dict:
        return {
            "display_status": self.display_status,
            "member_count": self.member_count,
            "reply_signal": self.reply_signal,
            "total_engagement": self.total_engagement,
            "ambiguous": self.ambiguous,
            "reasons": self.reasons,
        }


def _total_engagement(members: list[dict]) -> int:
    # Same weighting radar.py's own _serialize() uses -- kept identical so this
    # module's total_engagement always matches what the API already reports,
    # rather than silently introducing a second, slightly different definition.
    return sum(m.get("reply_count", 0) * 3 + m.get("repost_count", 0) for m in members)


def _looks_like_search_noise(matched_keywords: list[str], representative_text: str) -> bool:
    """Heuristic, not a claim of certainty -- see module docstring's honest limits
    (documented fully in the V2 diagnostic conclusion.md): this can only catch a
    keyword match that does NOT reappear, in any incident-type-recognizable form,
    in the post's own visible text (e.g. Bluesky's search matched something not
    present in the `text` field this module can see, or matched a wholly
    different sense than the keyword's disaster/incident meaning that this
    module's phrase list would also recognize). It systematically MISSES a
    keyword used in its literal disaster sense as an aside inside otherwise
    unrelated content (e.g. a pun using the literal word "flood") -- there is no
    NER/context model here to catch that; see radar_event_signature.py."""
    mapped_incident_types = {_KEYWORD_TO_INCIDENT_TYPE[kw] for kw in matched_keywords if kw in _KEYWORD_TO_INCIDENT_TYPE}
    if not mapped_incident_types:
        return False
    detected = extract_event_signature(representative_text or "").incident_types
    return detected.isdisjoint(mapped_incident_types)


def classify_radar_thread_quality(
    thread: dict,
    min_reply_signal: int = DEFAULT_MIN_REPLY_SIGNAL,
    min_total_engagement: int = DEFAULT_MIN_TOTAL_ENGAGEMENT,
) -> dict:
    """`thread` needs at least `members` (list of {reply_count, repost_count,
    text, ...}); `member_count` and `total_engagement` are read if already
    present (matching radar.py's _serialize output) and otherwise derived
    from `members` directly, so this also works on a raw un-serialized
    shape. `matched_keywords` and `representative_text` are optional --
    without them, the SEARCH_NOISE_CANDIDATE check is simply skipped
    (never guessed at from a member's own text, since the search-noise
    question is specifically about the field Bluesky's OWN search matched
    on).

    Fixed per review, round 2: the ambiguity check now runs BEFORE the
    member_count>=2 fast path, not after. member_count>=2 alone used to be
    sufficient for DISPLAY_ELIGIBLE, which meant a cluster v1's
    entity-blind clustering had already mis-merged (see event_clustering_
    v2.is_cluster_claim_ambiguous -- e.g. the real three-state-shooting
    merge this diagnostic found) would be PROMOTED for display on the
    strength of the very mistake this whole V2 effort exists to catch,
    with no visible warning.

    Fixed per review, round 3: an ambiguous cluster reaching DISPLAY_
    ELIGIBLE anyway (via reply_signal/total_engagement, since round 2's
    fix only skipped the member_count-alone fast path) is STILL wrong --
    a caller filtering on `display_status == "display_eligible"` would
    show it with no visible distinction from a genuinely-consistent
    event. An ambiguous cluster is now ALWAYS classified AMBIGUOUS_REVIEW,
    a status of its own, regardless of engagement -- it can still be
    surfaced (it may be genuinely graph-rich and worth a human's
    attention), but never silently conflated with DISPLAY_ELIGIBLE. The
    caller (see radar.py's _serialize and InterventionReviewPanel.jsx)
    decides whether/how to show AMBIGUOUS_REVIEW threads; this function
    only refuses to call one "eligible" without qualification."""
    members = thread.get("members") or []
    member_count = int(thread.get("member_count", len(members)))
    reply_signal = max((m.get("reply_count", 0) for m in members), default=0)
    total_engagement = thread.get("total_engagement")
    total_engagement = int(total_engagement) if total_engagement is not None else _total_engagement(members)
    member_signatures = [extract_event_signature(m.get("text", "")) for m in members if m.get("text")]
    ambiguous = member_count >= 2 and is_cluster_claim_ambiguous(member_signatures)

    reasons: list[str] = []
    if ambiguous:
        reasons.append(
            f"cluster's own members disagree on explicit location/incident-type signals "
            f"(is_cluster_claim_ambiguous=True) -- classified {AMBIGUOUS_REVIEW}, never "
            f"display_eligible, regardless of engagement (reply_signal={reply_signal}, "
            f"total_engagement={total_engagement})"
        )
        return RadarThreadQuality(AMBIGUOUS_REVIEW, member_count, reply_signal, total_engagement, reasons, ambiguous).as_dict()

    if member_count >= 2:
        reasons.append(f"member_count={member_count} >= 2: multiple independent source posts clustered under the same event, no internal conflict detected")
        return RadarThreadQuality(DISPLAY_ELIGIBLE, member_count, reply_signal, total_engagement, reasons, ambiguous).as_dict()

    if reply_signal >= min_reply_signal:
        reasons.append(f"max single-post reply_count={reply_signal} >= min_reply_signal={min_reply_signal}: a meaningful reply cascade exists to render")
        return RadarThreadQuality(DISPLAY_ELIGIBLE, member_count, reply_signal, total_engagement, reasons, ambiguous).as_dict()

    if total_engagement >= min_total_engagement:
        reasons.append(f"total_engagement={total_engagement} >= min_total_engagement={min_total_engagement}")
        return RadarThreadQuality(DISPLAY_ELIGIBLE, member_count, reply_signal, total_engagement, reasons, ambiguous).as_dict()

    matched_keywords = thread.get("matched_keywords") or []
    representative_text = thread.get("representative_text")
    if representative_text is not None and _looks_like_search_noise(matched_keywords, representative_text):
        reasons.append(
            f"matched_keywords={matched_keywords} map to an incident type not detected anywhere in representative_text "
            "-- likely a coincidental/unrelated keyword match, not a real incident (heuristic, see module docstring for its limits)"
        )
        return RadarThreadQuality(SEARCH_NOISE_CANDIDATE, member_count, reply_signal, total_engagement, reasons, ambiguous).as_dict()

    reasons.append(
        f"member_count={member_count} with low signal (reply_signal={reply_signal}, "
        f"total_engagement={total_engagement}) below display thresholds; kept tracked, not yet promoted"
    )
    return RadarThreadQuality(SINGLETON_MONITORING, member_count, reply_signal, total_engagement, reasons, ambiguous).as_dict()
