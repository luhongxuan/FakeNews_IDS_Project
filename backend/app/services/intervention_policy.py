"""Deterministic intervention priority + budget selection.

Responsibility split (per review after the PHEME agent_vs_rf_replay
analysis, effective_models/pheme_multicheckpoint_rf/experiments/
20260904_193125_signal_quota_cache_decomposition -- the only mechanism
that analysis found to reliably improve false-thread capture and reduce
harm to true content, once a ranking-signal confound was removed, was
RF-predicted-impact x verification-risk-weight ranking; the quota
mechanism itself (fixed vs. carry-over/release-ceiling) only matters when
it changes how many total interventions are spent, not detection quality
at a fixed budget):

- The verification Agent's job is ONLY to report what the evidence says:
  credibility, confidence, reasoning -- see
  intervention_agent.run_batch_verification. It does not decide who gets
  an intervention slot.
- This module's job is to turn (RF prediction, evidence) into a priority
  score, and to run budget-constrained selection deterministically. No LLM
  call happens in this file. A confirmed-true thread is excluded here in
  plain Python, not left to any model's judgment to honor.

This intentionally does NOT include a utility threshold (accept/reject
gate) -- the PHEME analysis found no stable advantage for one over plain
top-priority selection up to the budget. Note precisely what that means:
select_for_checkpoint takes the top-N eligible candidates by priority up
to the current released cap, with NO quality floor -- it does NOT skip a
checkpoint just because every remaining candidate has low priority. "Low
quality -> fewer selected" only happens when there are literally fewer
eligible candidates than budget, not because of any threshold on priority
itself. A minimum-quality gate is still explicitly not being integrated
(unproven in the PHEME analysis).

v2 (this revision) fixes issues a read-only review found in v1 that the
original stub-only smoke tests did not cover:
  - agent_failure could still be ranked and selected like "unverified"
    (RISK_WEIGHTS had an entry for it) -- a verification failure is not
    evidence of anything and must never compete for a slot. Fixed: handled
    as a hard exclusion in select_for_checkpoint, routed to a separate
    deferred_agent_failure list, never assigned a priority at all.
  - likely_true / likely_false were treated as permanently "resolved" the
    instant they were reported, regardless of confidence -- a
    confidence=0.2 likely_false got the same permanent risk_weight=1.0 and
    cache lock-in as a confidence=0.95 one. Fixed: added
    CONFIRMED_FALSE_CONFIDENCE symmetric to CONFIRMED_TRUE_CONFIDENCE;
    below-threshold true/false is downgraded to "unverified" (still
    eligible, still gets re-checked) via effective_credibility(), the
    single function everything else in this module goes through. (v2.1:
    an earlier version of this fix downgraded to "disputed" instead --
    caught by review as itself a bug, since disputed's weight of 0.75 is
    HIGHER than unverified's 0.5, making a weak "probably true" verdict
    outrank genuinely-unknown content. See effective_credibility's own
    docstring.)
  - "resolved" had no expiry -- a genuinely correct verdict from hours ago
    could go stale (retraction, new evidence) with no mechanism to ever
    look again. Fixed: RESOLVED_TTL_MINUTES: needs_check() re-opens a
    resolved record once the TTL has elapsed since resolution.
  - evidence_version only tracked credibility-label changes, not evidence
    content -- renamed to status_version to say exactly what it measures;
    a true content-fingerprint version is still out of scope (evidence is
    presently just a free-text summary, nothing structured to fingerprint
    yet).
  - Added recommendation_tier(): "hard" only for a genuinely confirmed
    likely_false; "soft" for everything else selected (disputed,
    unverified) -- callers must not apply the same intervention strength
    to an unresolved candidate as to a confirmed-false one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

CONFIRMED_TRUE_CONFIDENCE = 0.7   # must match intervention_agent.CONFIRMED_TRUE_CONFIDENCE
CONFIRMED_FALSE_CONFIDENCE = 0.7  # symmetric threshold -- a low-confidence "likely_false" is not yet a confirmed finding either

# Placeholder priors, NOT learned or tuned -- see
# signal_quota_cache_decomposition.py's run_record.json "known_limitations".
# A future sensitivity analysis should sweep UNVERIFIED_RISK_WEIGHT instead
# of treating 0.5 as settled. No entry for agent_failure: it is never
# scored at all, see select_for_checkpoint. deferred_inflight is likewise a
# scheduler state rather than evidence and is excluded before scoring.
RISK_WEIGHTS = {
    "likely_false": 1.0,
    "disputed": 0.75,
    "unverified": 0.5,
}

# Cumulative fraction of the total per-event budget released by checkpoint
# index, duplicated from effective_models/pheme_multicheckpoint_rf/training/
# opportunity_reserve_policy_common.py RELEASE_PROFILES["balanced"] -- backend/
# and effective_models/training/ are separate, unconnected packages (the
# backend does not import research code), so this is a deliberate constant
# duplication, not an oversight. Keep in sync if the source changes.
RELEASE_PROFILE_BALANCED = (0.18, 0.36, 0.52, 0.68, 0.84, 1.00)
CHECKPOINTS_MINUTES = (10, 20, 30, 40, 50, 60)

HIGH_RISK_COOLDOWN_MINUTES = 10  # re-check next checkpoint
LOW_RISK_COOLDOWN_MINUTES = 20   # re-check the checkpoint after that
# Placeholder split for which unresolved candidates get the shorter cooldown
# -- currently "top half of this checkpoint's shortlist by RF score", not
# tuned. See module docstring.

RESOLVED_TTL_MINUTES = 180  # placeholder -- not tuned against any real retraction/update-rate data yet


def effective_credibility(credibility: str | None, confidence: float | None) -> str:
    """The single source of truth for "what does this evidence actually
    support", after applying the confidence gate. Every other function in
    this module that looks at credibility goes through this -- never
    compare raw credibility strings directly.

    A below-threshold likely_true/likely_false downgrades to "unverified",
    NOT "disputed" -- disputed carries RISK_WEIGHTS["disputed"]=0.75, higher
    than unverified's 0.5, which would have made a weak "probably true"
    verdict outrank genuinely-unknown content for intervention priority
    (backwards: weak support for truth should never raise urgency above the
    neutral baseline). This is a temporary blanket choice, not a claim that
    weak-support and weak-refutation are equivalent -- a future experiment
    should split them into distinct weights instead of collapsing both to
    unverified.
    """
    confidence = confidence or 0.0
    if credibility == "likely_true" and confidence < CONFIRMED_TRUE_CONFIDENCE:
        return "unverified"
    if credibility == "likely_false" and confidence < CONFIRMED_FALSE_CONFIDENCE:
        return "unverified"
    return credibility or "unverified"


def is_confirmed_true(credibility: str | None, confidence: float | None) -> bool:
    return effective_credibility(credibility, confidence) == "likely_true"


def is_confirmed_false(credibility: str | None, confidence: float | None) -> bool:
    return effective_credibility(credibility, confidence) == "likely_false"


def recommendation_tier(credibility: str | None, confidence: float | None) -> str:
    """"hard" = strong intervention warranted (genuinely confirmed false).
    "soft" = everything else that still got selected (disputed/unverified
    at high enough priority) -- must be handled with a lighter touch, not
    the same action strength as a hard case. No tiered action-effectiveness
    model exists yet (deferred, see prior P2 discussion) -- this is only
    the two-level distinction needed to stop treating them identically.
    """
    return "hard" if is_confirmed_false(credibility, confidence) else "soft"


def priority_for(predicted_impact: float, credibility: str | None, confidence: float | None) -> float | None:
    """Returns None if this candidate must be excluded outright (confirmed
    true) -- callers must check for None, not treat it as priority 0.
    agent_failure is NOT handled here -- it must never reach this function;
    see select_for_checkpoint's hard exclusion.
    """
    effective = effective_credibility(credibility, confidence)
    if effective == "likely_true":
        return None
    weight = RISK_WEIGHTS.get(effective, RISK_WEIGHTS["unverified"])
    return max(predicted_impact, 0.0) * weight


def released_cap(checkpoint_minutes: int, total_budget: int, release_profile: tuple[float, ...] = RELEASE_PROFILE_BALANCED) -> int:
    if checkpoint_minutes not in CHECKPOINTS_MINUTES:
        raise ValueError(f"checkpoint_minutes must be one of {CHECKPOINTS_MINUTES}, got {checkpoint_minutes}")
    index = CHECKPOINTS_MINUTES.index(checkpoint_minutes)
    return min(total_budget, math.ceil(total_budget * release_profile[index]))


@dataclass
class Candidate:
    post_uri: str
    cluster_thread_id: str
    predicted_impact: float          # RF's predicted_impact, already in real node units (see intervention_model.predict_impact)
    credibility: str | None          # evidence status, or scheduler-only "agent_failure" / "deferred_inflight"
    confidence: float | None
    evidence_reasoning: str = ""


@dataclass
class SelectionResult:
    selected: list[Candidate] = field(default_factory=list)
    excluded_confirmed_true: list[str] = field(default_factory=list)
    deferred_agent_failure: list[str] = field(default_factory=list)
    deferred_inflight: list[str] = field(default_factory=list)
    priority_by_uri: dict[str, float] = field(default_factory=dict)
    tier_by_uri: dict[str, str] = field(default_factory=dict)


def select_for_checkpoint(
    candidates: list[Candidate],
    checkpoint_minutes: int,
    already_intervened_count: int,
    total_budget: int,
    release_profile: tuple[float, ...] = RELEASE_PROFILE_BALANCED,
) -> SelectionResult:
    """Pure function, no DB/LLM access.

    Exclusions, in order:
      1. credibility == "agent_failure" -> deferred_agent_failure. A failed
         verification is not evidence of anything; it must be retried, not
         treated as risk. See module docstring.
      2. credibility == "deferred_inflight" -> deferred_inflight. Another
         scheduler is still producing the evidence, so selection must wait.
      3. effective_credibility == "likely_true" -> excluded_confirmed_true.

    Everything else is ranked by priority_for() and the top-N (by priority,
    no quality floor) that fit under this checkpoint's released cumulative
    cap MINUS already_intervened_count is selected. An empty or
    small-candidate checkpoint can select fewer than the cap allows simply
    because fewer candidates exist -- NOT because of any quality-based
    cutoff (there isn't one here). Unused capacity is implicitly carried
    forward: the next checkpoint's remaining_budget is computed the same
    way against the same running already_intervened_count.
    """
    result = SelectionResult()
    cap = released_cap(checkpoint_minutes, total_budget, release_profile)
    remaining_budget = max(cap - already_intervened_count, 0)

    ranked: list[tuple[Candidate, float]] = []
    for c in candidates:
        if c.credibility == "agent_failure":
            result.deferred_agent_failure.append(c.post_uri)
            continue
        if c.credibility == "deferred_inflight":
            result.deferred_inflight.append(c.post_uri)
            continue
        if is_confirmed_true(c.credibility, c.confidence):
            result.excluded_confirmed_true.append(c.post_uri)
            continue
        priority = priority_for(c.predicted_impact, c.credibility, c.confidence)
        ranked.append((c, priority))
        result.priority_by_uri[c.post_uri] = priority
        result.tier_by_uri[c.post_uri] = recommendation_tier(c.credibility, c.confidence)

    ranked.sort(key=lambda pair: -pair[1])
    result.selected = [c for c, _priority in ranked[:remaining_budget]]
    return result


# --- Verification cache -----------------------------------------------------
# Pure, DB-independent state so the same logic can run against a live
# VerificationReport-backed store (see intervention_agent.py's DB adapter)
# or an in-memory dict for offline replay / smoke tests.

@dataclass
class CacheRecord:
    last_status: str | None = None           # the EFFECTIVE (post-confidence-gate) status, not the raw reported one
    last_checked_at_minutes: int | None = None
    next_check_due_minutes: int | None = None
    verification_attempts: int = 0
    status_version: int = 0                  # counts credibility-LABEL changes only -- see module docstring, not a content fingerprint
    resolved: bool = False
    resolved_at_minutes: int | None = None


def needs_check(record: CacheRecord | None, checkpoint_minutes: int) -> bool:
    if record is None:
        return True
    if record.resolved:
        if record.resolved_at_minutes is not None and (checkpoint_minutes - record.resolved_at_minutes) < RESOLVED_TTL_MINUTES:
            return False
        return True  # TTL expired -- a "resolved" verdict is not exempt from re-verification forever
    if record.next_check_due_minutes is None:
        return True
    return checkpoint_minutes >= record.next_check_due_minutes


def record_check(record: CacheRecord | None, checkpoint_minutes: int, credibility: str | None,
                  confidence: float | None, is_high_risk: bool) -> CacheRecord:
    """credibility/confidence here are the RAW verification result --
    effective_credibility() is applied internally so the cache always
    stores and reasons about the confidence-gated status. Must never be
    called with credibility == "agent_failure" -- a failed check has
    nothing to record; see intervention_agent.run_batch_verification,
    which deliberately skips this call for failures so next_check_due is
    left untouched and a real retry happens later.
    """
    effective = effective_credibility(credibility, confidence)
    record = record or CacheRecord()
    status_changed = record.last_status != effective
    record.verification_attempts += 1
    if status_changed:
        record.status_version += 1
    record.last_status = effective
    record.last_checked_at_minutes = checkpoint_minutes
    if effective in ("likely_true", "likely_false"):
        record.resolved = True
        record.resolved_at_minutes = checkpoint_minutes
        record.next_check_due_minutes = None
    else:
        record.resolved = False
        record.resolved_at_minutes = None
        cooldown = HIGH_RISK_COOLDOWN_MINUTES if is_high_risk else LOW_RISK_COOLDOWN_MINUTES
        record.next_check_due_minutes = checkpoint_minutes + cooldown
    return record
