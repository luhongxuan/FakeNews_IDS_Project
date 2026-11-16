"""Real-model-backed intervention demo endpoints.

All data here comes from two already-existing, already-validated artifacts:
- the leakage-audited nested-LOEO OOF predictions in
  effective_models/pheme_v5_text_rf/reference_result/... (see
  app/services/model_scores.py)
- the raw PHEME reply-tree cascade in data/raw/pheme (see
  app/services/pheme_cascade.py)

No model is trained or modified by these endpoints. The "intervene" endpoint
is a retrospective replay: it reports how many of a thread's real historical
replies occurred after the 30-minute observation cutoff, i.e. how many would
not have appeared had the source tweet been removed at that cutoff. It does
not delete or modify anything in data/raw/pheme.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from app.services import model_scores, pheme_replay
from app.services.pheme_cascade import load_cascade, load_source_preview

router = APIRouter()


def _pheme_base() -> Path:
    root = Path(__file__).resolve().parents[3]
    env_value = os.getenv("PHEME_PATH")
    if env_value:
        # backend/.env sets this relative to the backend/ directory (docker
        # WORKDIR, and where PHEME_PATH's own comment/history assumes cwd is
        # /app == backend/). Resolve relative paths against backend/, not
        # against whatever cwd this process happened to be launched from.
        candidate = Path(env_value)
        if not candidate.is_absolute():
            candidate = (root / "backend" / candidate).resolve()
        if candidate.exists():
            return candidate
    return root / "data" / "raw" / "pheme"


@router.get("/api/events/{event_id}/threads")
def get_ranked_threads(event_id: str, limit: int = Query(1000, ge=1, le=1000)):
    rows = model_scores.ranked_threads(event_id, limit)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No scored threads for event_id={event_id}")
    base = _pheme_base()
    for row in rows:
        row["preview_text"] = load_source_preview(base, event_id, row["thread_id"])
    return {"event_id": event_id, "cutoff_seconds": model_scores.CUTOFF_SECONDS, "threads": rows}


@router.get("/api/threads/{thread_id}/cascade")
def get_cascade(thread_id: str, event_id: str = Query(...)):
    base = _pheme_base()
    try:
        return load_cascade(base, event_id, thread_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/api/events/{event_id}/replay_manifest")
def get_replay_manifest(event_id: str, limit: int = Query(1000, ge=1, le=1000)):
    """Every scored thread's real arrival order on ONE shared event clock --
    the data source for the 即時情境模擬 command-center MVP's alert queue and
    simulated clock. See pheme_replay.build_event_manifest's own docstring
    for exactly what "arrival_offset_seconds" means and why RF-rank order
    (the OLD simulator's ordering) is not the same thing as real arrival
    order.
    """
    rows = model_scores.ranked_threads(event_id, limit)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No scored threads for event_id={event_id}")
    base = _pheme_base()
    manifest = pheme_replay.build_event_manifest(base, event_id, rows)
    for row in manifest["threads"]:
        row["preview_text"] = load_source_preview(base, event_id, row["thread_id"])
    return manifest


@router.get("/api/threads/{thread_id}/cascade_window")
def get_cascade_window(
    thread_id: str,
    event_id: str = Query(...),
    thread_arrival_offset_seconds: float = Query(..., description="From replay_manifest's arrival_offset_seconds for this thread."),
    sim_time_seconds: float = Query(..., ge=0, description="Current simulated event clock, in seconds since the event's earliest source tweet."),
):
    """Time-gated version of /cascade: only nodes/edges whose real event-time
    has already been reached by sim_time_seconds. Enforced here, in the data
    layer -- see pheme_replay.cascade_window's own docstring for the exact
    invariant. The frontend must call this instead of the plain /cascade
    endpoint while a replay is actually running.
    """
    base = _pheme_base()
    try:
        cascade = load_cascade(base, event_id, thread_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return pheme_replay.cascade_window(cascade, thread_arrival_offset_seconds, sim_time_seconds)


@router.get("/api/threads/{thread_id}/intervene_window")
def get_intervene_window(
    thread_id: str,
    event_id: str = Query(...),
    thread_arrival_offset_seconds: float = Query(...),
    sim_time_seconds: float = Query(..., ge=0),
):
    """Progressive reveal of realized-vs-pending counterfactually-blocked
    nodes for a thread the operator already chose to intervene on. Call
    this on every simulated-clock tick after intervening (GET, not POST --
    it doesn't record the intervention decision itself, only reports how
    much of it has "come true" as sim time advances); see
    pheme_replay.realized_blocked_nodes for the exact semantics, which
    match the existing /intervene endpoint's blocked-node definition
    exactly, just revealed over simulated time instead of all at once.
    """
    base = _pheme_base()
    try:
        cascade = load_cascade(base, event_id, thread_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return pheme_replay.realized_blocked_nodes(cascade, thread_arrival_offset_seconds, sim_time_seconds)


@router.get("/api/threads/{thread_id}/auto_decision")
def get_auto_decision(
    thread_id: str,
    predicted_impact: float = Query(...),
    credibility: str | None = Query(None),
    confidence: float | None = Query(None),
):
    """"Agent 自動決策" mode for the 即時情境模擬指揮台: given this thread's
    already-known RF score and verification result, what would the SAME
    deterministic policy the live radar pipeline uses actually decide? Pure
    passthrough to pheme_replay.auto_decision -- see its docstring for the
    exact (deliberately conservative) rule. Never runs the verification
    agent itself; the caller must have already checked (or explicitly chosen
    not to check) this thread.
    """
    return {"thread_id": thread_id, **pheme_replay.auto_decision(predicted_impact, credibility, confidence)}


@router.post("/api/threads/{thread_id}/intervene")
def intervene(thread_id: str, event_id: str = Query(...)):
    base = _pheme_base()
    try:
        cascade = load_cascade(base, event_id, thread_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    nodes = cascade["nodes"]
    blocked = [node for node in nodes if not node["observed_by_cutoff"]]
    return {
        "thread_id": thread_id,
        "event_id": event_id,
        "cutoff_seconds": cascade["cutoff_seconds"],
        "total_nodes_full_cascade": len(nodes),
        "observed_nodes_at_cutoff": len(nodes) - len(blocked),
        "blocked_node_count": len(blocked),
        "blocked_node_ids": [node["id"] for node in blocked],
        "note": (
            "回溯重播示範：以下節點皆為歷史真實資料。blocked_node_count 是其中發生於"
            "30 分鐘截止點之後的節點數——若在截止當下阻擋此則來源貼文，這些節點在"
            "歷史上原本就不會出現。"
        ),
    }
