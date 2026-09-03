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

from app.database import get_db
from app.models.schema import RadarThread, VerificationReport
from app.services import bluesky_source
from app.services.bluesky_source import DEFAULT_KEYWORDS
from app.services.event_clustering import assign_post_to_cluster
from app.services.verification_agent import OLLAMA_MODEL, run_verification

router = APIRouter()

_INGEST_INTERVAL_SECONDS = 45
_last_ingest_at = 0.0


def _serialize(thread: RadarThread) -> dict:
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
        "total_engagement": sum(m.get("reply_count", 0) * 3 + m.get("repost_count", 0) for m in thread.members),
    }


_SORT_MODES = ("latest", "top")
_MAX_POST_AGE = timedelta(days=3)


async def _ingest_new_posts(db: Session) -> None:
    async def _safe_search(keyword: str, sort: str) -> list[dict]:
        try:
            return await bluesky_source.search_recent_posts(keyword, limit=10, sort=sort)
        except httpx.HTTPError:
            return []

    # Ingest both sort modes: "latest" catches things the moment they're
    # posted (almost always zero replies -- no time has passed yet), "top"
    # catches things Bluesky already judges as engaged-with (a real
    # propagation tree to look at). Without "top", the whole pool skews
    # toward brand-new posts with nothing to show yet -- confirmed directly
    # in testing (157/166 tracked posts sitting at 0 replies). Ingesting
    # both means a caller can filter by engagement afterward without this
    # background job having to pick one mode for everyone.
    #
    # "top" is not time-bounded, though -- confirmed directly in testing it
    # surfaced a post from 176 days ago. That's a poor fit for something
    # framed as a *live* radar (and it broke the deletion-impact estimate,
    # whose rate-since-creation math assumes recent activity), so anything
    # older than _MAX_POST_AGE is dropped here before it ever gets clustered.
    age_cutoff = datetime.now(timezone.utc) - _MAX_POST_AGE
    jobs = [(kw, sort) for kw in DEFAULT_KEYWORDS for sort in _SORT_MODES]
    results = await asyncio.gather(*(_safe_search(kw, sort) for kw, sort in jobs))
    for (keyword, sort), posts in zip(jobs, results):
        for post in posts:
            if not post["text"]:
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
    return "radar_" + hashlib.sha256(str(thread_id).encode("utf-8")).hexdigest()[:24]


@router.get("/api/radar/threads/{thread_id}/verify")
def get_radar_verification(thread_id: str, db: Session = Depends(get_db)):
    existing = db.query(VerificationReport).filter_by(thread_id=_verify_cache_key(thread_id)).first()
    if not existing:
        raise HTTPException(status_code=404, detail="No verification yet")
    return {**existing.report_jsonb, "cached": True, "generated_at": existing.created_at.isoformat() if existing.created_at else None}


@router.post("/api/radar/threads/{thread_id}/verify")
async def verify_radar_thread(thread_id: str, force: bool = Query(False), db: Session = Depends(get_db)):
    thread = db.query(RadarThread).filter_by(id=thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Radar thread not found")

    key = _verify_cache_key(thread_id)
    if not force:
        existing = db.query(VerificationReport).filter_by(thread_id=key).first()
        if existing:
            return {**existing.report_jsonb, "cached": True, "generated_at": existing.created_at.isoformat() if existing.created_at else None}

    try:
        report = await run_verification(db, key, "radar", thread.representative_text)
    except httpx.ConnectError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach the local Ollama server (model={OLLAMA_MODEL}). Is `ollama serve` running?",
        ) from error
    except httpx.HTTPStatusError as error:
        raise HTTPException(status_code=502, detail=f"Ollama returned an error: {error}") from error

    existing = db.query(VerificationReport).filter_by(thread_id=key).first()
    if existing:
        existing.report_jsonb = report
        existing.model_name = OLLAMA_MODEL
    else:
        db.add(VerificationReport(thread_id=key, event_id="radar", model_name=OLLAMA_MODEL, report_jsonb=report))
    db.commit()

    return {**report, "cached": False, "generated_at": None}


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
