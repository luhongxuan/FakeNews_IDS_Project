"""Live radar: keyword-search recent public Bluesky posts, cluster them into
same-event "threads" (app/services/event_clustering.py), and persist to a
"pending review" queue -- replacing the earlier pure in-memory/ephemeral
version, which could drop a post mid-verification when the next 30s poll
reshuffled the list.

Two frontend surfaces share this same queue:
- general users: read-only "what's trending right now" + on-demand verify
- research/ops: same list + a live graph + an estimated deletion impact
  (see bluesky_source.estimate_deletion_impact)

Still keyless, still read-only against Bluesky, still never touches the
early-detection models' training/evaluation data.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import SessionLocal, get_db
from app.models.schema import InterventionDecision, RadarThread, VerificationReport
from app.services import (
    bluesky_source,
    event_clustering_v2,
    intervention_agent,
    intervention_features,
    intervention_model,
    intervention_policy,
    manual_intervention_mvp,
    radar_thread_quality,
)
from app.services.bluesky_source import DEFAULT_KEYWORDS
from app.services.event_clustering import assign_post_to_cluster
from app.services.radar_event_signature import extract_event_signature
from app.services.verification_agent import OLLAMA_MODEL, VERIFICATION_PROMPT_VERSION, run_verification

router = APIRouter()

_INGEST_INTERVAL_SECONDS = 45
_last_ingest_at = 0.0


def _serialize(thread: RadarThread) -> dict:
    total_engagement = sum(m.get("reply_count", 0) * 3 + m.get("repost_count", 0) for m in thread.members)
    # Live Radar Event Discovery V2, phase 2: computed server-side (not reimplemented in
    # JS -- this codebase already has too many independent frontend reimplementations of
    # backend decision logic, see InterventionReviewPanel.jsx's own pre-existing ad-hoc
    # elevated/reasoning heuristic) so every caller sees the SAME classification. The API
    # itself still returns every thread regardless of this field -- filtering by it is a
    # frontend display choice (see InterventionReviewPanel.jsx's "真實事件預覽" tab), not
    # something this endpoint enforces.
    quality = radar_thread_quality.classify_radar_thread_quality({
        "member_count": len(thread.members), "members": thread.members,
        "matched_keywords": thread.matched_keywords, "representative_text": thread.representative_text,
        "total_engagement": total_engagement,
    })
    return {
        "id": str(thread.id),
        "representative_text": thread.representative_text,
        "representative_uri": thread.representative_uri,
        "representative_created_at": thread.representative_created_at.isoformat() if thread.representative_created_at else None,
        "matched_keywords": thread.matched_keywords,
        "member_count": len(thread.members),
        "members": thread.members,
        "reviewed": thread.reviewed,
        "first_seen_at": thread.first_seen_at.isoformat() if thread.first_seen_at else None,
        "last_seen_at": thread.last_seen_at.isoformat() if thread.last_seen_at else None,
        "total_engagement": total_engagement,
        "display_status": quality["display_status"],
        "display_status_reasons": quality["reasons"],
        # Fixed per review (round 3): `ambiguous` was computed by classify_radar_thread_
        # quality but never actually reached the API response -- the frontend had no way
        # to render a warning at all. Always present (not only when True) so the frontend
        # never has to special-case a missing field.
        "ambiguous": quality["ambiguous"],
    }


_SORT_MODES = ("latest", "top")
_MAX_POST_AGE = timedelta(days=3)


# "top"-sort search results Bluesky itself already judges as engaged-with,
# but plenty still have few or no replies (a like-heavy post can rank "top"
# without much of a reply cascade). Filtering to a reply-count floor here
# --only for "top", never for "latest" (see below) -- biases what actually
# gets stored toward posts with a real propagation tree, which is what the
# RF/graph/verification pipeline actually needs to have something to reason
# about. Not tuned against any target distribution -- just large enough to
# exclude the common near-zero-reply "top" result.
_MIN_REPLIES_FOR_TOP_SORT_INGEST = 5
_TOP_SORT_SEARCH_LIMIT = 100  # Bluesky's searchPosts max page size -- up from the shared default of 10, so the reply-count floor above still leaves enough posts to cluster


async def _ingest_new_posts(db: Session) -> None:
    async def _safe_search(keyword: str, sort: str, limit: int) -> list[dict]:
        try:
            return await bluesky_source.search_recent_posts(keyword, limit=limit, sort=sort)
        except httpx.HTTPError:
            return []

    # Ingest both sort modes: "latest" catches things the moment they're
    # posted (almost always zero replies -- no time has passed yet), "top"
    # catches things Bluesky already judges as engaged-with (a real
    # propagation tree to look at). Without "top", the whole pool skews
    # toward brand-new posts with nothing to show yet -- confirmed directly
    # in testing (157/166 tracked posts sitting at 0 replies). Ingesting
    # both means a caller can filter by engagement afterward without this
    # background job having to pick one mode for everyone. "latest" is
    # deliberately NEVER reply-count-filtered -- catching a real zero-reply
    # post the moment it's posted is this mode's entire purpose (early
    # detection), so filtering it would defeat it.
    #
    # "top" is not time-bounded, though -- confirmed directly in testing it
    # surfaced a post from 176 days ago. That's a poor fit for something
    # framed as a *live* radar (and it broke the deletion-impact estimate,
    # whose rate-since-creation math assumes recent activity), so anything
    # older than _MAX_POST_AGE is dropped here before it ever gets clustered.
    age_cutoff = datetime.now(timezone.utc) - _MAX_POST_AGE
    jobs = [(kw, sort, _TOP_SORT_SEARCH_LIMIT if sort == "top" else 10) for kw in DEFAULT_KEYWORDS for sort in _SORT_MODES]
    results = await asyncio.gather(*(_safe_search(kw, sort, limit) for kw, sort, limit in jobs))
    for (keyword, sort, _limit), posts in zip(jobs, results):
        for post in posts:
            if not post["text"]:
                continue
            if sort == "top" and post.get("reply_count", 0) < _MIN_REPLIES_FOR_TOP_SORT_INGEST:
                continue
            created_at = bluesky_source.parse_iso(post.get("created_at"))
            if created_at and created_at < age_cutoff:
                continue
            assign_post_to_cluster(db, post, keyword)


def _prune_stale_threads(db: Session) -> None:
    """Drop clusters whose representative post is now older than
    _MAX_POST_AGE. Needed on top of the ingestion-time filter because that
    filter only stops *new* old posts from being added -- it does nothing
    about a cluster that was ingested when its post was still fresh and has
    since aged past the cutoff, or ones ingested before this filter existed.
    """
    # representative_created_at is stored naive (the DateTime column has no
    # timezone), always as a UTC-equivalent value elsewhere in this module --
    # match that here rather than comparing against a tz-aware cutoff, which
    # some DB drivers handle inconsistently.
    cutoff = (datetime.now(timezone.utc) - _MAX_POST_AGE).replace(tzinfo=None)
    (
        db.query(RadarThread)
        .filter(RadarThread.representative_created_at < cutoff)
        .delete(synchronize_session=False)
    )
    db.commit()


async def _maybe_ingest(db: Session) -> None:
    global _last_ingest_at
    now = time.monotonic()
    if now - _last_ingest_at < _INGEST_INTERVAL_SECONDS:
        return
    _last_ingest_at = now
    _prune_stale_threads(db)
    await _ingest_new_posts(db)


@router.get("/api/radar/threads")
async def list_radar_threads(
    limit: int = Query(30, ge=1, le=100),
    min_engagement: int = Query(0, ge=0, description="Only include threads with at least this much total_engagement (reply_count*3 + repost_count, summed across members). 0 = everything, including brand-new posts with no replies yet."),
    db: Session = Depends(get_db),
):
    await _maybe_ingest(db)
    threads = (
        db.query(RadarThread)
        .order_by(RadarThread.last_seen_at.desc())
        .limit(limit * 3 if min_engagement > 0 else limit)  # over-fetch since some will be filtered out below
        .all()
    )
    serialized = [_serialize(t) for t in threads]
    if min_engagement > 0:
        serialized = [t for t in serialized if t["total_engagement"] >= min_engagement][:limit]
    return {"threads": serialized, "keywords": DEFAULT_KEYWORDS}


@router.get("/api/radar/threads/{thread_id}")
def get_radar_thread(thread_id: str, db: Session = Depends(get_db)):
    thread = db.query(RadarThread).filter_by(id=thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Radar thread not found")
    return _serialize(thread)


@router.post("/api/radar/threads/{thread_id}/mark_reviewed")
def mark_reviewed(thread_id: str, reviewed: bool = Query(True), db: Session = Depends(get_db)):
    thread = db.query(RadarThread).filter_by(id=thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Radar thread not found")
    thread.reviewed = reviewed
    db.commit()
    return _serialize(thread)


def _verify_cache_key(thread_id: str) -> str:
    # VERIFICATION_PROMPT_VERSION folded in for the same reason as
    # factcheck.py's _cache_key -- see that constant's docstring. Must match
    # app/services/intervention_agent.py's own copy of this formula exactly
    # (duplicated there, not imported, to avoid a router->service->router
    # import cycle) -- see that copy's comment.
    return "radar_" + hashlib.sha256(f"{VERIFICATION_PROMPT_VERSION}|{thread_id}".encode("utf-8")).hexdigest()[:24]


def _verification_context(thread: RadarThread, thread_id: str, target_uri: str, created_at) -> tuple[str, str | None, str, bool]:
    """Claim text / post_date / cache key for ONE specific post inside this
    radar cluster -- shared by the plain /verify endpoints below AND
    manual_policy_decision so both ALWAYS check the exact same evidence for
    the exact same claim.

    Before this existed, the two panels disagreed constantly: this plain
    /verify endpoint always checked thread.representative_text (the whole
    cluster's representative post, which may not even be the post on
    screen) with NO post_date, while manual_policy_decision checked the
    SPECIFIC selected post's own text with its real post_date. post_date
    matters a lot, not just for bookkeeping -- run_verification anchors its
    search queries to that year (see _heuristic_query's docstring); without
    it, an ambiguous/old-sounding claim (e.g. one that name-drops a JFK-era
    figure) tends to pull up unrelated historical search results and the
    agent lands on "insufficient evidence", where the date-anchored search
    used by manual_policy_decision correctly finds the real, recent story
    and confidently refutes. Two different inputs to the same underlying
    agent, hashed into two different cache keys, will keep disagreeing no
    matter how good the agent is -- the fix is to stop asking it two
    different questions, not to make the model more consistent.
    """
    member = next((m for m in thread.members if m.get("uri") == target_uri), None)
    claim_text = (member or {}).get("text") or thread.representative_text
    post_date = created_at.strftime("%Y-%m-%d") if created_at else None
    # Same ambiguous-cluster guard as manual_policy_decision (see its own
    # comment) -- an ambiguous cluster's posts must never share one cache
    # key, or one post's evidence could silently "verify" a different post's
    # claim.
    member_signatures = [extract_event_signature(m.get("text", "")) for m in thread.members]
    is_ambiguous_cluster = event_clustering_v2.is_cluster_claim_ambiguous(member_signatures)
    cache_key = _verify_cache_key(f"{thread_id}:{target_uri}") if is_ambiguous_cluster else _verify_cache_key(thread_id)
    return claim_text, post_date, cache_key, is_ambiguous_cluster


@router.get("/api/radar/threads/{thread_id}/verify")
def get_radar_verification(thread_id: str, post_uri: str | None = Query(None), db: Session = Depends(get_db)):
    thread = db.query(RadarThread).filter_by(id=thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Radar thread not found")
    target_uri, created_at = _resolve_post(thread, post_uri)
    _claim_text, _post_date, key, _ambiguous = _verification_context(thread, thread_id, target_uri, created_at)
    existing = db.query(VerificationReport).filter_by(thread_id=key).first()
    if not existing:
        raise HTTPException(status_code=404, detail="No verification yet")
    return {**existing.report_jsonb, "cached": True, "generated_at": existing.created_at.isoformat() if existing.created_at else None}


@router.post("/api/radar/threads/{thread_id}/verify")
async def verify_radar_thread(
    thread_id: str, post_uri: str | None = Query(None), force: bool = Query(False),
    request_id: str | None = Query(None, description="Frontend-minted id to poll live progress at GET /api/verification_progress/{request_id} while this call is in flight."),
    db: Session = Depends(get_db),
):
    thread = db.query(RadarThread).filter_by(id=thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Radar thread not found")

    target_uri, created_at = _resolve_post(thread, post_uri)
    claim_text, post_date, key, _ambiguous = _verification_context(thread, thread_id, target_uri, created_at)
    if not force:
        existing = db.query(VerificationReport).filter_by(thread_id=key).first()
        if existing:
            return {**existing.report_jsonb, "cached": True, "generated_at": existing.created_at.isoformat() if existing.created_at else None}

    try:
        report = await run_verification(db, key, "radar", claim_text, post_date=post_date, progress_id=request_id)
    except httpx.ConnectError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach the local Ollama server (model={OLLAMA_MODEL}). Is `ollama serve` running?",
        ) from error
    except httpx.HTTPStatusError as error:
        raise HTTPException(status_code=502, detail=f"Ollama returned an error: {error}") from error

    # Persisted through the SAME cooldown-bookkeeping helper manual_policy_decision's
    # run_batch_verification uses (intervention_agent.persist_verification_check) --
    # a bare `existing.report_jsonb = report` overwrite here used to wipe the
    # next_check_due_minutes/resolved fields the MVP path depends on, making it
    # treat every check as "never verified" and re-run for real again even
    # moments later. See that helper's docstring for the full story.
    checkpoint_seconds = intervention_features.nearest_usable_checkpoint(
        (datetime.now(timezone.utc) - created_at).total_seconds(),
    ) if created_at else None
    checkpoint_minutes = int(checkpoint_seconds // 60) if checkpoint_seconds is not None else 0
    report = intervention_agent.persist_verification_check(db, key, "radar", checkpoint_minutes, True, report)

    return {**report, "cached": False, "generated_at": None}


@router.get("/api/radar/threads/{thread_id}/intervention_score")
async def get_radar_intervention_score(thread_id: str, post_uri: str | None = Query(None), db: Session = Depends(get_db)):
    """Balanced Cumulative RF impact score for this thread at whatever
    checkpoint it has actually reached (10/20/30/40/50/60 minutes since the
    post). See intervention_model.py for what the score does and doesn't
    mean -- it's a propagation-structure signal only, blind to whether the
    content is true.
    """
    thread = db.query(RadarThread).filter_by(id=thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Radar thread not found")

    target_uri, created_at = _resolve_post(thread, post_uri)
    if created_at is None:
        raise HTTPException(status_code=422, detail="This post has no known creation time to measure elapsed time from")

    elapsed_seconds = (datetime.now(timezone.utc) - created_at).total_seconds()
    checkpoint = intervention_features.nearest_usable_checkpoint(elapsed_seconds)
    if checkpoint is None:
        return {
            "thread_id": thread_id,
            "post_uri": target_uri,
            "scoreable": False,
            "reason": "This thread is younger than the first checkpoint (10 minutes) -- not enough observation yet.",
            "elapsed_seconds": round(elapsed_seconds),
        }

    try:
        thread_json = await bluesky_source.get_post_thread(target_uri)
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"Could not fetch Bluesky thread: {error}") from error

    graph = bluesky_source.build_cascade_graph(thread_json, target_uri, created_at)
    features = intervention_features.build_cumulative_features(graph["nodes"], checkpoint)
    result = intervention_model.predict_impact(features)

    return {
        "thread_id": thread_id,
        "post_uri": target_uri,
        "scoreable": True,
        "checkpoint_seconds": checkpoint,
        "elapsed_seconds": round(elapsed_seconds),
        **result,
    }


async def _review_and_persist(db: Session, thread_id: str, target_uri: str, created_at) -> dict:
    """Shared by the on-demand endpoint and the background scheduler below:
    fetch the live thread, run the triage agent, persist the decision.
    """
    elapsed_minutes = (datetime.now(timezone.utc) - created_at).total_seconds() / 60
    thread_json = await bluesky_source.get_post_thread(target_uri)
    graph = bluesky_source.build_cascade_graph(thread_json, target_uri, created_at)
    decision = await intervention_agent.run_intervention_review(db, thread_id, graph["nodes"], elapsed_minutes)

    checkpoints_reviewed = decision.get("checkpoints_reviewed") or []
    if checkpoints_reviewed:
        latest_checkpoint = max(checkpoints_reviewed)
    else:
        usable = intervention_features.nearest_usable_checkpoint(elapsed_minutes * 60)
        latest_checkpoint = int(usable // 60) if usable is not None else 0

    # Surface the RF's raw score as a plain "predicted_reach" number even
    # for the single-thread path -- pulled from its first (automatic)
    # get_rf_prediction tool call, so the frontend can show it without a
    # second request. See intervention_model.py for what this number
    # actually means (predicted future reply-cascade growth, not literal
    # impressions).
    predicted_reach = None
    for call in decision.get("tool_calls", []):
        if call.get("name") == "get_rf_prediction" and isinstance(call.get("result"), dict):
            predicted_reach = call["result"].get("predicted_impact")
            break
    decision["predicted_reach"] = predicted_reach

    db.add(InterventionDecision(
        thread_id=thread_id,
        post_uri=target_uri,
        checkpoint_minutes=latest_checkpoint,
        action=decision.get("action", "hold"),
        confidence=float(decision.get("confidence") or 0.0),
        reasoning=decision.get("reasoning", ""),
        report_jsonb=decision,
    ))
    db.commit()
    return decision


@router.post("/api/radar/threads/{thread_id}/intervention_review")
async def review_radar_thread_intervention(thread_id: str, post_uri: str | None = Query(None), db: Session = Depends(get_db)):
    """Runs intervention_agent.py's triage loop (RF score + verification
    status + budget state) for this thread and persists the decision to
    InterventionDecision. Recommendation only -- nothing here removes or
    hides content; a human still acts on "escalate_now".
    """
    thread = db.query(RadarThread).filter_by(id=thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Radar thread not found")

    target_uri, created_at = _resolve_post(thread, post_uri)
    if created_at is None:
        raise HTTPException(status_code=422, detail="This post has no known creation time to measure elapsed time from")

    try:
        decision = await _review_and_persist(db, thread_id, target_uri, created_at)
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"Could not fetch Bluesky thread: {error}") from error

    return {"thread_id": thread_id, "post_uri": target_uri, **decision}


_MVP_INTERVENTION_ACTIONS = ("hard", "soft")


def _manual_policy_decision_to_response(row: InterventionDecision) -> dict:
    report = row.report_jsonb or {}
    return {
        "id": str(row.id),
        "thread_id": row.thread_id,
        "post_uri": row.post_uri,
        "checkpoint_minutes": row.checkpoint_minutes,
        "action": row.action,
        "confidence": row.confidence,
        "reasoning": row.reasoning,
        "decided_at": row.decided_at.isoformat() if row.decided_at else None,
        **report,
    }


@router.post("/api/radar/threads/{thread_id}/manual_policy_decision")
async def manual_policy_decision(
    thread_id: str,
    post_uri: str | None = Query(None),
    force: bool = Query(False, description="Create a fresh audit row even when this post already has an MVP decision at the same checkpoint."),
    # db kept in its original 4th position -- test_manual_policy_endpoint.py
    # calls this positionally as (thread_id, post_uri, force, db); request_id
    # goes AFTER it so that existing positional call keeps working unchanged.
    db: Session = Depends(get_db),
    request_id: str | None = Query(None, description="Frontend-minted id to poll live progress at GET /api/verification_progress/{request_id} while this call is in flight."),
):
    """One-post, manually-triggered live decision-support MVP.

    This performs real live-data scoring and evidence gathering, but only
    persists a recommendation. It never calls a platform moderation API.
    """
    thread = db.query(RadarThread).filter_by(id=thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Radar thread not found")

    target_uri, created_at = _resolve_post(thread, post_uri)
    if created_at is None:
        raise HTTPException(status_code=422, detail="This post has no known creation time")

    # Live Radar Event Discovery V2, phase 5 (decision safety): if this cluster's own
    # members disagree on explicit location/incident-type signals, it must NOT be
    # treated as one shared claim -- v1's clustering (event_clustering.py) never checks
    # entity conflicts at merge time (see event_clustering_v2.py's own docstring for a
    # real example this caught: three unrelated shootings merged into one cluster), so
    # an ALREADY-formed cluster can still be ambiguous even though it passed the v1
    # similarity gate. An ambiguous cluster gets its own PER-POST verification cache key
    # (never the cluster-shared one) so this post's evidence can never be silently
    # satisfied by, or leak into, a different real event's verification result.
    # claim_text/post_date/verification_cache_key come from the SAME
    # _verification_context helper the plain /verify endpoints above use --
    # this used to compute its own copy inline (representative_text-only,
    # no date anchor was never the issue here, but keeping two independent
    # copies of this logic is how it drifted from /verify in the first
    # place), so this endpoint and /verify can never again check two
    # different claims/dates for what looks like the same post on screen.
    claim_text, post_date, verification_cache_key, is_ambiguous_cluster = _verification_context(
        thread, thread_id, target_uri, created_at,
    )

    elapsed_seconds = (datetime.now(timezone.utc) - created_at).total_seconds()
    checkpoint_seconds = intervention_features.nearest_usable_checkpoint(elapsed_seconds)
    if checkpoint_seconds is None:
        raise HTTPException(status_code=422, detail="This post is younger than the first 10-minute checkpoint")
    checkpoint_minutes = int(checkpoint_seconds // 60)

    latest = (
        db.query(InterventionDecision)
        .filter_by(thread_id=thread_id, post_uri=target_uri, checkpoint_minutes=checkpoint_minutes)
        .order_by(InterventionDecision.decided_at.desc())
        .first()
    )
    if latest and (latest.report_jsonb or {}).get("policy_version") == manual_intervention_mvp.POLICY_VERSION and not force:
        return {**_manual_policy_decision_to_response(latest), "cached_decision": True}

    try:
        thread_json = await bluesky_source.get_post_thread(target_uri)
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"Could not fetch Bluesky thread: {error}") from error
    graph = bluesky_source.build_cascade_graph(thread_json, target_uri, created_at)
    rf_result = intervention_agent.score_thread_at_checkpoint(
        graph["nodes"], elapsed_seconds / 60, checkpoint_minutes,
    )
    predicted_impact = max(float(rf_result.get("predicted_impact") or 0.0), 0.0)

    # Fixed per review (Live Radar Event Discovery V2, phase 5): this endpoint used to
    # pass the RAW `thread_id` here as cluster_thread_id, so run_batch_verification would
    # read/write VerificationReport.thread_id=<raw UUID> -- a DIFFERENT key from every
    # other verification pathway in this file (get_radar_verification/verify_radar_thread
    # and intervention_agent.get_verification_status_for/is_confirmed_true all key off
    # _verify_cache_key(thread_id), a SHA256 hash). The two caches never saw each other's
    # writes: a thread confirmed true via THIS endpoint would still show as "unverified"
    # to is_confirmed_true's hard safety gate (and vice versa), and calling both endpoints
    # for the same thread paid for two independent real verification calls instead of one
    # cached one. Now uses the same _verify_cache_key(...) formula as everything else
    # (or, for an ambiguous cluster, the per-post variant computed above).
    verification_input = [{
        "post_uri": target_uri,
        "cluster_thread_id": verification_cache_key,
        "claim_text": claim_text,
        "event_id": "radar",
        # run_verification's post_date param is documented (and, via
        # _heuristic_query/its prompt, actually used) as a "YYYY-MM-DD"
        # STRING -- passing the raw datetime object here made every single
        # manual MVP verification blow up inside the agent loop with
        # "'datetime.datetime' object is not subscriptable" (post_date[:4]),
        # silently landing as a generic agent_failure with zero tool calls
        # ever made. Never caught by the smoke tests because those stub out
        # verify_fn entirely and never touch this string-vs-datetime seam.
        "post_date": post_date,
        # Manual MVP reviews are conservatively rechecked at the next
        # checkpoint when unresolved. This affects cache cooldown only, not
        # the current priority or intervention tier.
        "is_high_risk": True,
    }]
    verification_batch = await intervention_agent.run_batch_verification(
        db, verification_input, checkpoint_minutes, progress_id=request_id,
    )
    evidence = verification_batch["results"][verification_cache_key]

    since = datetime.now(timezone.utc) - timedelta(hours=24)
    prior_rows = (
        db.query(InterventionDecision)
        .filter(
            InterventionDecision.thread_id == thread_id,
            InterventionDecision.action.in_(_MVP_INTERVENTION_ACTIONS),
            InterventionDecision.decided_at >= since,
        )
        .all()
    )
    already_intervened_count = len({row.post_uri for row in prior_rows})
    candidate = intervention_policy.Candidate(
        post_uri=target_uri,
        cluster_thread_id=thread_id,
        predicted_impact=predicted_impact,
        credibility=evidence.get("credibility"),
        confidence=evidence.get("confidence"),
        evidence_reasoning=evidence.get("summary") or "",
    )
    selection = intervention_policy.select_for_checkpoint(
        [candidate], checkpoint_minutes, already_intervened_count,
        intervention_model.TOTAL_BUDGET,
    )
    recommendation = manual_intervention_mvp.recommendation_from_selection(
        candidate, selection, evidence,
    )
    # Phase 5 decision safety: an ambiguous cluster must never receive a "hard"
    # (restrictive) recommendation on the strength of a claim that might not even be
    # THIS post's own claim -- cap at "soft" and say why, rather than trusting the
    # evidence-status alone. This is deliberately below the policy layer's own hard/soft
    # tier decision (intervention_policy.recommendation_tier), not a replacement for it --
    # a per-post-verified, genuinely confirmed-false post in an ambiguous cluster is still
    # exactly what "soft" (reversible, re-checked) is for.
    if is_ambiguous_cluster and recommendation["action"] == "hard":
        recommendation["action"] = "soft"
        recommendation["action_strength"] = manual_intervention_mvp.ACTION_STRENGTH["soft"]
        recommendation["reasoning"] += (
            " (Downgraded from hard to soft: this cluster's own members disagree on explicit "
            "location/incident-type signals, so a single shared claim cannot be trusted for a "
            "restrictive action -- see is_cluster_claim_ambiguous.)"
        )
    release_cap = intervention_policy.released_cap(
        checkpoint_minutes, intervention_model.TOTAL_BUDGET,
    )
    report = {
        **recommendation,
        "auto": False,
        "source": "manual_live_policy_mvp",
        "predicted_reach": predicted_impact,
        "predicted_log1p_impact": rf_result.get("predicted_log1p_impact"),
        "rf_top_features": rf_result.get("top_features", []),
        "verification": evidence,
        "verification_checked_now": verification_cache_key in verification_batch.get("checked_now", []),
        "verification_cache_key": verification_cache_key,
        "is_ambiguous_cluster": is_ambiguous_cluster,
        "released_cap": release_cap,
        "already_intervened_count": already_intervened_count,
        "remaining_capacity_before_decision": max(release_cap - already_intervened_count, 0),
        "observation_cutoff_seconds": checkpoint_seconds,
    }
    row = InterventionDecision(
        thread_id=thread_id,
        post_uri=target_uri,
        checkpoint_minutes=checkpoint_minutes,
        action=recommendation["action"],
        confidence=float(recommendation.get("verification_confidence") or 0.0),
        reasoning=recommendation["reasoning"],
        report_jsonb=report,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {**_manual_policy_decision_to_response(row), "cached_decision": False}


@router.get("/api/radar/interventions")
def list_intervention_decisions(
    limit: int = Query(50, ge=1, le=200),
    action: str | None = Query(None),
    thread_id: str | None = Query(None, description="Filter to one radar cluster's decision history (all posts in it)."),
    post_uri: str | None = Query(None, description="Filter to one individual post's own decision history -- the actual unit a decision is made about."),
    db: Session = Depends(get_db),
):
    """Feeds a 'pending human review' surface: the auto-scheduler's (or a
    manually triggered) triage decisions, newest first. Every row here is a
    recommendation the agent already reasoned through -- nothing was
    auto-acted on.

    The intervention budget (intervention_model.TOTAL_BUDGET, rolling 24h)
    is scoped PER EVENT -- see _remaining_budget -- not a single pool
    shared across every event, so this summary intentionally does NOT
    report one global "X remaining out of 50": that framing would imply a
    single shared budget that doesn't exist. escalated_all_time/_last_24h
    here are just raw activity counts across every event, for visibility;
    check GET /api/radar/threads/{thread_id}/budget_state for one event's
    own actual remaining budget.

    `thread_id` + `post_uri` together let RadarEventDetail.jsx show the
    decision history for whichever single post (discussion thread) is
    currently selected, instead of merging every post in the cluster
    together or only being visible in the separate InterventionReviewPanel
    list.
    """
    query = db.query(InterventionDecision)
    if action:
        query = query.filter(InterventionDecision.action == action)
    if thread_id:
        query = query.filter(InterventionDecision.thread_id == thread_id)
    if post_uri:
        query = query.filter(InterventionDecision.post_uri == post_uri)
    rows = query.order_by(InterventionDecision.decided_at.desc()).limit(limit).all()

    since = datetime.now(timezone.utc) - timedelta(hours=24)
    escalated_last_24h = (
        db.query(InterventionDecision)
        .filter(InterventionDecision.action == "escalate_now", InterventionDecision.decided_at >= since)
        .count()
    )
    escalated_all_time = db.query(InterventionDecision).filter(InterventionDecision.action == "escalate_now").count()

    return {
        "summary": {
            "escalated_last_24h": escalated_last_24h,
            "escalated_all_time": escalated_all_time,
        },
        "decisions": [
            {
                "id": str(row.id),
                "thread_id": row.thread_id,
                "post_uri": row.post_uri,
                "checkpoint_minutes": row.checkpoint_minutes,
                "action": row.action,
                "confidence": row.confidence,
                "reasoning": row.reasoning,
                "predicted_reach": (row.report_jsonb or {}).get("predicted_reach"),
                "policy_version": (row.report_jsonb or {}).get("policy_version"),
                "action_strength": (row.report_jsonb or {}).get("action_strength"),
                "effective_credibility": (row.report_jsonb or {}).get("effective_credibility"),
                "priority": (row.report_jsonb or {}).get("priority"),
                "risk_weight": (row.report_jsonb or {}).get("risk_weight"),
                "recommendation_only": (row.report_jsonb or {}).get("recommendation_only", True),
                "decided_at": row.decided_at.isoformat() if row.decided_at else None,
            }
            for row in rows
        ],
    }


@router.get("/api/radar/threads/{thread_id}/budget_state")
def get_radar_thread_budget_state(thread_id: str, db: Session = Depends(get_db)):
    """This event's own remaining intervention budget, scoped per event --
    see _remaining_budget for why. Matches the original PHEME policy design
    ('50 threads per eligible event', quotas [9,9,8,8,8,8] per event): an
    event's own threads compete for this event's own budget, never against
    an unrelated event's threads.
    """
    thread = db.query(RadarThread).filter_by(id=thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Radar thread not found")

    since = datetime.now(timezone.utc) - timedelta(hours=24)
    total_used = (
        db.query(InterventionDecision)
        .filter(
            InterventionDecision.thread_id == thread_id,
            InterventionDecision.action == "escalate_now",
            InterventionDecision.decided_at >= since,
        )
        .count()
    )
    return {
        "thread_id": thread_id,
        "total_budget": intervention_model.TOTAL_BUDGET,
        "total_used_last_24h": total_used,
        "total_remaining": max(intervention_model.TOTAL_BUDGET - total_used, 0),
        "checkpoint_tiers": {
            str(checkpoint_minutes): {
                "quota": quota,
                "remaining": _remaining_budget(db, thread_id, checkpoint_minutes),
            }
            for checkpoint_minutes, quota in intervention_model.CHECKPOINT_QUOTAS.items()
        },
    }


_REVIEW_INTERVAL_SECONDS = 180
# How many different events (RadarThread clusters) to process per tick --
# NOT how many individual threads. Each event's own due threads are all
# processed together in one batch (see _due_for_review), since the whole
# point of the ranking is to compare threads *within* the same event.
_REVIEW_EVENTS_PER_TICK = 3


def _remaining_budget(db: Session, cluster_thread_id: str, checkpoint_minutes: int) -> int:
    """The tighter of two limits, both scoped to ONE event (RadarThread
    cluster): this checkpoint tier's own quota, and the event's overall
    budget across every tier (intervention_model.TOTAL_BUDGET).

    This mirrors the original PHEME policy exactly: "50 threads per
    eligible event" and quotas [9,9,8,8,8,8] were always per event -- an
    event's threads compete for that event's own budget, never against an
    unrelated event's threads. Earlier this budget was implemented as one
    pool shared across every event, which was wrong: a 39-thread event
    could get starved of its own quota by an unrelated event's threads
    happening to look more urgent in the same tick. Adapted to a rolling
    24h window per event, instead of PHEME's single full-replay window.
    """
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    tier_used = (
        db.query(InterventionDecision)
        .filter(
            InterventionDecision.thread_id == cluster_thread_id,
            InterventionDecision.checkpoint_minutes == checkpoint_minutes,
            InterventionDecision.action == "escalate_now",
            InterventionDecision.decided_at >= since,
        )
        .count()
    )
    total_used = (
        db.query(InterventionDecision)
        .filter(
            InterventionDecision.thread_id == cluster_thread_id,
            InterventionDecision.action == "escalate_now",
            InterventionDecision.decided_at >= since,
        )
        .count()
    )
    tier_quota = intervention_model.CHECKPOINT_QUOTAS.get(checkpoint_minutes, 0)
    return max(min(tier_quota - tier_used, intervention_model.TOTAL_BUDGET - total_used), 0)


def _post_velocity(member: dict, created_at) -> float:
    """Engagement accumulated so far, per minute the post has existed --
    catches posts getting heavy discussion FAST, not just posts that are
    merely old enough to have slowly accumulated some replies over hours.
    Same reply-weighted-3x convention as total_engagement elsewhere in this
    module (a reply signals more active engagement than a repost).
    """
    age_minutes = max((datetime.now(timezone.utc) - created_at).total_seconds() / 60, 1.0)
    engagement = member.get("reply_count", 0) * 3 + member.get("repost_count", 0)
    return engagement / age_minutes


def _member_candidates(thread: RadarThread) -> list[tuple[str, object, float]]:
    """Every individual post in this cluster eligible for AUTOMATIC
    intervention -- each its own discussion thread with its own reply
    cascade, not the cluster as a whole -- paired with its velocity score.
    The representative post is always also one of thread.members (see
    event_clustering.py's assign_post_to_cluster), so it's covered here
    too, just not singled out.

    Posts with fewer than intervention_model.MIN_REPLIES_FOR_AUTO_INTERVENTION
    replies are excluded entirely -- see that constant for why: a post
    with no observed replies gives the RF no real propagation structure to
    score, so its prediction ends up driven by superficial source-text
    formatting instead of actual spread. This only gates the SCHEDULER's
    candidate pool; on-demand manual checks (get_radar_intervention_score,
    the manual /intervention_review endpoint) still work on any post,
    since a human explicitly asking is a different situation from the
    scheduler guessing where to spend its limited budget. reply_count here
    is the snapshot captured when this post was first ingested (radar
    threads never re-scrape an already-known member), so it can undercount
    a post that has since gained replies -- an accepted staleness, same as
    the velocity score below which uses the same snapshot.
    """
    out: list[tuple[str, object, float]] = []
    seen: set[str] = set()
    for member in thread.members:
        uri = member.get("uri")
        if not uri or uri in seen:
            continue
        if member.get("reply_count", 0) < intervention_model.MIN_REPLIES_FOR_AUTO_INTERVENTION:
            continue
        created_at = bluesky_source.parse_iso(member.get("created_at"))
        if created_at is None:
            continue
        created_at = _normalize_dt(created_at)
        seen.add(uri)
        out.append((uri, created_at, _post_velocity(member, created_at)))
    return out


def _due_for_review(db: Session) -> list[tuple[RadarThread, int, list[tuple[str, object]]]]:
    """(event, checkpoint_minutes, [(post_uri, created_at), ...]) -- one
    entry per (event, checkpoint) pair that has at least one of its own
    threads newly eligible for review.

    This groups strictly BY EVENT (one RadarThread cluster) because that's
    how the underlying policy was actually designed: PHEME's reference
    bundle budget ("50 threads per eligible event") and checkpoint quotas
    were always scoped to one event's own threads ranked against each
    other, never pooled across unrelated events. A 39-thread earthquake
    event and a 1-thread meme post are not competing for the same slots --
    each event gets its own budget, so run_batch_selection is always
    called once per (event, checkpoint) with only that event's own due
    threads.

    Events are prioritized by their highest-velocity due thread and capped
    to _REVIEW_EVENTS_PER_TICK per tick (each event's batch still involves
    live Bluesky fetches plus an LLM call, unattended) -- but once an
    event is selected, ALL of its own due threads at that checkpoint are
    included, not an arbitrary global cap, since the ranking needs to see
    the whole competing set to mean anything.

    Verification (is_confirmed_true) stays at the event level -- threads
    clustered together are near-duplicates of the same underlying claim by
    construction (event_clustering.py's cosine-similarity threshold), so
    an event confirmed true makes every thread under it ineligible too.
    Those get auto-dismissed right here with no LLM call: this project
    only spends its limited intervention budget on rumours (PHEME's sense:
    unverified or contested claims), never on content already confirmed
    accurate, enforced in plain Python rather than left to the agent to
    honor as an instruction.
    """
    event_groups: list[tuple[RadarThread, float, dict[int, list[tuple[str, object]]]]] = []
    for thread in db.query(RadarThread).order_by(RadarThread.last_seen_at.desc()).limit(30).all():
        if intervention_agent.is_confirmed_true(db, str(thread.id)):
            for post_uri, _created_at, _velocity in _member_candidates(thread):
                already = db.query(InterventionDecision).filter_by(thread_id=str(thread.id), post_uri=post_uri).first()
                if already:
                    continue
                db.add(InterventionDecision(
                    thread_id=str(thread.id), post_uri=post_uri,
                    checkpoint_minutes=0, action="dismiss", confidence=1.0,
                    reasoning="Auto-dismissed without an LLM call: verification agent already confirmed this event as true, so no thread under it is eligible for intervention.",
                    report_jsonb={"auto": True},
                ))
            db.commit()
            continue

        by_checkpoint: dict[int, list[tuple[str, object]]] = {}
        max_velocity = 0.0
        for post_uri, created_at, velocity in _member_candidates(thread):
            elapsed_seconds = (datetime.now(timezone.utc) - created_at).total_seconds()
            checkpoint_seconds = intervention_features.nearest_usable_checkpoint(elapsed_seconds)
            if checkpoint_seconds is None:
                continue
            checkpoint_minutes = int(checkpoint_seconds // 60)
            latest = (
                db.query(InterventionDecision)
                .filter_by(thread_id=str(thread.id), post_uri=post_uri)
                .order_by(InterventionDecision.checkpoint_minutes.desc())
                .first()
            )
            if latest and latest.checkpoint_minutes >= checkpoint_minutes:
                continue  # already reviewed at this or a later checkpoint
            by_checkpoint.setdefault(checkpoint_minutes, []).append((post_uri, created_at))
            max_velocity = max(max_velocity, velocity)

        if by_checkpoint:
            event_groups.append((thread, max_velocity, by_checkpoint))

    event_groups.sort(key=lambda item: item[1], reverse=True)

    result: list[tuple[RadarThread, int, list[tuple[str, object]]]] = []
    for thread, _velocity, by_checkpoint in event_groups[:_REVIEW_EVENTS_PER_TICK]:
        for checkpoint_minutes, items in by_checkpoint.items():
            result.append((thread, checkpoint_minutes, items))
    return result


async def _review_due_threads_tick() -> None:
    db = SessionLocal()
    try:
        due_groups = _due_for_review(db)
        if not due_groups:
            return

        for thread, checkpoint_minutes, items in due_groups:
            candidates = []
            for post_uri, created_at in items:
                try:
                    thread_json = await bluesky_source.get_post_thread(post_uri)
                except httpx.HTTPError as error:
                    print(f"[intervention scheduler] skipped {post_uri} (Bluesky fetch failed: {error})")
                    continue
                graph = bluesky_source.build_cascade_graph(thread_json, post_uri, created_at)
                age_minutes = (datetime.now(timezone.utc) - created_at).total_seconds() / 60
                candidates.append({
                    "post_uri": post_uri,
                    "cluster_thread_id": str(thread.id),
                    "nodes": graph["nodes"],
                    "age_minutes": age_minutes,
                    "initial_rf": intervention_agent.score_thread_at_checkpoint(graph["nodes"], age_minutes, checkpoint_minutes),
                    "verification": intervention_agent.get_verification_status_for(db, str(thread.id)),
                })
            if not candidates:
                continue

            remaining_budget = _remaining_budget(db, str(thread.id), checkpoint_minutes)
            if remaining_budget <= 0:
                for c in candidates:
                    db.add(InterventionDecision(
                        thread_id=c["cluster_thread_id"], post_uri=c["post_uri"],
                        checkpoint_minutes=checkpoint_minutes, action="hold", confidence=0.0,
                        reasoning="This event's budget is exhausted for this checkpoint tier (or overall) in the last 24h; deferred to the next checkpoint without spending an LLM call.",
                        report_jsonb={"auto": True},
                    ))
                db.commit()
                continue

            batch_result = await intervention_agent.run_batch_selection(db, candidates, checkpoint_minutes, remaining_budget)
            cluster_by_uri = {c["post_uri"]: c["cluster_thread_id"] for c in candidates}
            reach_by_uri = {c["post_uri"]: c["initial_rf"].get("predicted_impact") for c in candidates}
            for decision in batch_result["decisions"]:
                db.add(InterventionDecision(
                    thread_id=cluster_by_uri[decision["post_uri"]],
                    post_uri=decision["post_uri"],
                    checkpoint_minutes=checkpoint_minutes,
                    action=decision["action"],
                    confidence=decision["confidence"],
                    reasoning=decision["reasoning"],
                    report_jsonb={
                        **decision,
                        "predicted_reach": reach_by_uri[decision["post_uri"]],
                        "model": batch_result["model"],
                        "tool_calls": batch_result["tool_calls"],
                    },
                ))
            db.commit()
    finally:
        db.close()


async def run_intervention_scheduler_forever() -> None:
    """Background loop, started once at app startup (see main.py). Reviews
    a small batch of newly-checkpoint-eligible threads every
    _REVIEW_INTERVAL_SECONDS.

    Deliberately NOT folded into _maybe_ingest's request-triggered pattern:
    that pattern is fine for ingestion (a couple of fast HTTP searches), but
    a review involves a local-LLM tool-calling loop that can take
    10-60+ seconds per thread -- piggybacking that onto a dashboard page
    load would make the UI feel broken for whoever's request happened to
    trigger it.
    """
    while True:
        try:
            await _review_due_threads_tick()
        except Exception as error:  # one bad tick must not kill the loop
            print(f"[intervention scheduler] tick failed: {error}")
        await asyncio.sleep(_REVIEW_INTERVAL_SECONDS)


def _normalize_dt(value):
    if value and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _resolve_post(thread: RadarThread, post_uri: str | None) -> tuple[str, object]:
    """Resolve which post within this cluster to act on, and its known
    created_at. Every member (including the representative) is independently
    clickable, mirroring how picking a thread inside a PHEME event swaps the
    cascade view in InterventionDashboard.
    """
    if not post_uri or post_uri == thread.representative_uri:
        return thread.representative_uri, _normalize_dt(thread.representative_created_at)
    member = next((m for m in thread.members if m["uri"] == post_uri), None)
    if member is None:
        raise HTTPException(status_code=404, detail="That post is not part of this radar thread")
    return post_uri, _normalize_dt(bluesky_source.parse_iso(member.get("created_at")))


@router.get("/api/radar/threads/{thread_id}/graph")
async def get_radar_thread_graph(thread_id: str, post_uri: str | None = Query(None), db: Session = Depends(get_db)):
    thread = db.query(RadarThread).filter_by(id=thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Radar thread not found")

    target_uri, created_at = _resolve_post(thread, post_uri)
    try:
        thread_json = await bluesky_source.get_post_thread(target_uri)
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"Could not fetch Bluesky thread: {error}") from error

    graph = bluesky_source.build_cascade_graph(thread_json, target_uri, created_at)
    return {"thread_id": thread_id, "post_uri": target_uri, "cutoff_seconds": bluesky_source.CUTOFF_SECONDS, **graph}


@router.post("/api/radar/threads/{thread_id}/estimate_intervene")
async def estimate_intervene_radar_thread(thread_id: str, post_uri: str | None = Query(None), db: Session = Depends(get_db)):
    """Deliberately an estimate, not a replay -- see
    bluesky_source.estimate_deletion_impact for why: this content is live,
    so unlike the PHEME-backed /api/threads/{id}/intervene endpoint there is
    no already-known future to show.
    """
    thread = db.query(RadarThread).filter_by(id=thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Radar thread not found")

    target_uri, created_at = _resolve_post(thread, post_uri)
    try:
        thread_json = await bluesky_source.get_post_thread(target_uri)
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"Could not fetch Bluesky thread: {error}") from error

    current_reply_count = ((thread_json.get("thread") or {}).get("post") or {}).get("replyCount", 0)
    estimate = bluesky_source.estimate_deletion_impact(created_at, current_reply_count)
    return {"thread_id": thread_id, "post_uri": target_uri, **estimate}
