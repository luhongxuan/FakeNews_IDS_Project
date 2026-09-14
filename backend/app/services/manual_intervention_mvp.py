"""Pure decision formatting for the manually-triggered live-policy MVP.

The router performs I/O (live graph, RF, verification and persistence).  This
module only turns an already-computed SelectionResult into one auditable
recommendation.  Keeping this mapping pure lets the safety-critical cases be
tested without Bluesky, an LLM or a database.
"""
from __future__ import annotations

from app.services import intervention_policy

POLICY_VERSION = "manual_live_policy_mvp_v1"

ACTION_STRENGTH = {
    "hard": "75-100%",
    "soft": "10-25%",
    "none": "0%",
    "deferred_agent_failure": "0%",
    "deferred_inflight": "0%",
    "deferred_budget": "0%",
}


def recommendation_from_selection(
    candidate: intervention_policy.Candidate,
    selection: intervention_policy.SelectionResult,
    evidence: dict,
) -> dict:
    """Return one recommendation for one manually reviewed live post.

    Failure/lock/confirmed-true paths are checked before selected/budget paths
    so lack of evidence can never silently become an intervention.
    """
    uri = candidate.post_uri
    effective = intervention_policy.effective_credibility(
        candidate.credibility, candidate.confidence,
    )

    if uri in selection.deferred_agent_failure:
        action = "deferred_agent_failure"
        reasoning = "Verification failed after retry. No restrictive action is recommended; retry evidence gathering later."
    elif uri in selection.deferred_inflight:
        action = "deferred_inflight"
        reasoning = "Another worker is verifying this claim. No restrictive action is recommended until that evidence is available."
    elif uri in selection.excluded_confirmed_true:
        action = "none"
        reasoning = "Reliable evidence currently supports the claim, so it is excluded from restrictive intervention."
    elif any(item.post_uri == uri for item in selection.selected):
        action = selection.tier_by_uri[uri]
        if action == "hard":
            reasoning = "Reliable evidence currently refutes the claim and the post was selected under the released intervention budget."
        else:
            reasoning = "The claim remains unresolved and the RF propagation estimate placed it within the released budget; only a reversible soft action is recommended."
    else:
        action = "deferred_budget"
        reasoning = "The post is eligible, but no released intervention capacity remains at this checkpoint. Re-evaluate at a later checkpoint."

    return {
        "policy_version": POLICY_VERSION,
        "action": action,
        "action_strength": ACTION_STRENGTH[action],
        "recommendation_only": True,
        "effective_credibility": effective,
        "verification_confidence": candidate.confidence,
        "evidence_summary": evidence.get("summary"),
        "verification_from_cache": bool(evidence.get("from_cache")),
        "priority": selection.priority_by_uri.get(uri),
        "risk_weight": (
            intervention_policy.RISK_WEIGHTS.get(effective)
            if uri in selection.priority_by_uri else None
        ),
        "reasoning": reasoning,
    }
