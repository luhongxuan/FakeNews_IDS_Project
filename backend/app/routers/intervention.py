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

from app.services import model_scores
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
