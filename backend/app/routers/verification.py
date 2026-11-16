"""Post-hoc verification agent endpoints.

Runs strictly after the early-detection ranking; see the module docstring
in app/services/verification_agent.py for the leakage boundary this must
respect.
"""
from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.schema import VerificationReport
from app.routers.intervention import _pheme_base
from app.services.pheme_cascade import load_source_date, load_source_preview
from app.services.verification_agent import OLLAMA_MODEL, run_verification
from app.services import verification_progress

router = APIRouter()


@router.get("/api/verification_progress/{request_id}")
def get_verification_progress(request_id: str):
    """Polled by the frontend while a verification is in flight (see
    run_verification's `progress_id` param) so it can show each tool call
    as the agent makes it instead of only after the whole ~10-90s run
    finishes. Best-effort and in-memory only -- see verification_progress.py
    for why that's fine here. 404 just means "nothing recorded under this
    id (yet, or ever)", not an error -- the caller may be polling slightly
    before the POST that starts the run has reached the server.
    """
    entry = verification_progress.get(request_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="No progress recorded for this request_id")
    return entry


def _cached_report(db: Session, thread_id: str) -> dict | None:
    row = db.query(VerificationReport).filter_by(thread_id=thread_id).first()
    if not row:
        return None
    return {**row.report_jsonb, "cached": True, "generated_at": row.created_at.isoformat() if row.created_at else None}


@router.get("/api/threads/{thread_id}/verify")
def get_cached_verification(thread_id: str, db: Session = Depends(get_db)):
    cached = _cached_report(db, thread_id)
    if not cached:
        raise HTTPException(status_code=404, detail="No verification report yet for this thread")
    return cached


@router.post("/api/threads/{thread_id}/verify")
async def run_thread_verification(
    thread_id: str, event_id: str = Query(...), force: bool = Query(False),
    request_id: str | None = Query(None, description="Frontend-minted id to poll live progress at GET /api/verification_progress/{request_id} while this call is in flight."),
    db: Session = Depends(get_db),
):
    if not force:
        cached = _cached_report(db, thread_id)
        if cached:
            return cached

    base = _pheme_base()
    claim_text = load_source_preview(base, event_id, thread_id, max_chars=2000)
    if not claim_text:
        raise HTTPException(status_code=404, detail=f"Could not load source text for thread {thread_id}")
    post_date = load_source_date(base, event_id, thread_id)

    try:
        report = await run_verification(db, thread_id, event_id, claim_text, post_date=post_date, progress_id=request_id)
    except httpx.ConnectError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach the local Ollama server (model={OLLAMA_MODEL}). Is `ollama serve` running?",
        ) from error
    except httpx.HTTPStatusError as error:
        raise HTTPException(status_code=502, detail=f"Ollama returned an error: {error}") from error

    existing = db.query(VerificationReport).filter_by(thread_id=thread_id).first()
    if existing:
        existing.report_jsonb = report
        existing.model_name = OLLAMA_MODEL
    else:
        db.add(VerificationReport(thread_id=thread_id, event_id=event_id, model_name=OLLAMA_MODEL, report_jsonb=report))
    db.commit()

    return {**report, "cached": False, "generated_at": None}
