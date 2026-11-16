"""General-purpose fact-check endpoint for arbitrary user-supplied text.

Deliberately decoupled from the early-detection / intervention pipeline:
/api/threads/{id}/verify answers "should an operator prioritize this
already-known PHEME thread"; this answers a completely different question
that a general user actually has -- "is this specific thing I'm reading
true" -- for any text they paste in, not tied to any known thread_id or
event_id. Same underlying verification agent, no new leakage surface: this
still never touches the early-detection models' training/evaluation data.
"""
from __future__ import annotations

import hashlib

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.schema import VerificationReport
from app.services.verification_agent import OLLAMA_MODEL, VERIFICATION_PROMPT_VERSION, run_verification

router = APIRouter()

MAX_TEXT_LENGTH = 2000


class FactCheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    request_id: str | None = Field(
        None, description="Frontend-minted id to poll live progress at GET /api/verification_progress/{request_id} while this call is in flight.",
    )


def _cache_key(text: str) -> str:
    # Deterministic per exact text so identical lookups hit cache instead of
    # re-running the agent; reuses VerificationReport's existing thread_id
    # column as a general cache key, not an actual PHEME thread id.
    # VERIFICATION_PROMPT_VERSION is folded in so a prompt/reasoning-rule fix
    # (see its own docstring -- e.g. the Isabela earthquake false-refutation
    # case) can never keep being served from a report generated under the
    # old, since-fixed reasoning: bumping it makes every prior cache entry
    # unreachable under the new key, forcing a fresh check next time.
    return "adhoc_" + hashlib.sha256(f"{VERIFICATION_PROMPT_VERSION}|{text.strip()}".encode("utf-8")).hexdigest()[:24]


@router.post("/api/factcheck")
async def factcheck(payload: FactCheckRequest, force: bool = Query(False), db: Session = Depends(get_db)):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")
    key = _cache_key(text)

    if not force:
        existing = db.query(VerificationReport).filter_by(thread_id=key).first()
        if existing:
            return {**existing.report_jsonb, "cached": True, "generated_at": existing.created_at.isoformat() if existing.created_at else None}

    try:
        report = await run_verification(db, key, "adhoc", text, progress_id=payload.request_id)
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
        db.add(VerificationReport(thread_id=key, event_id="adhoc", model_name=OLLAMA_MODEL, report_jsonb=report))
    db.commit()

    return {**report, "cached": False, "generated_at": None}
