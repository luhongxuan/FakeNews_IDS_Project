"""In-memory, best-effort live progress feed for the verification agent's
tool-call loop.

A real verification call is 1424 real inference calls' worth of experience
telling us this can take anywhere from ~10s to 90s+ (see the historical
analogy tournament's own call accounting). Without this, the frontend can
only show a generic spinner for that whole window. This module lets
run_verification() push each tool call into a shared dict AS IT HAPPENS, so
a concurrently-polling frontend request can show "已呼叫 search_web 查 ...,
拿回 0 筆" while the agent is still running its 4th search, not just after
it finally returns.

Deliberately NOT persisted anywhere (no DB table, no Redis, no websocket) --
this is purely a same-process UI convenience:
- Safe to lose on a server restart; nothing here is ever the source of
  truth for a verification result (VerificationReport.report_jsonb is).
- Keyed by a short-lived, frontend-generated request_id (a UUID minted
  right before the POST that starts a verification), never by any
  dataset/thread/event id -- so a stale or reused progress entry can never
  be mistaken for cached verification state, and this file never becomes
  part of the leakage surface described in verification_agent.py's own
  module docstring.
- A single dict is fine here (no cross-worker fan-out): this backend is
  run as one uvicorn process for local dev/demo use, per AGENTS.md's
  Network Access rules (no real deployment infra to keep in sync).
"""
from __future__ import annotations

import time

_STORE: dict[str, dict] = {}

# Best-effort cleanup so this dict can't grow forever if a caller starts a
# progress feed (POSTs with a request_id) and the frontend never polls it,
# or polls it and then the tab is closed before the run finishes.
_MAX_AGE_SECONDS = 600


def start(request_id: str | None) -> None:
    if not request_id:
        return
    _STORE[request_id] = {"tool_calls": [], "done": False, "error": None, "_started_at": time.monotonic()}
    _sweep()


def append_tool_call(request_id: str | None, call: dict) -> None:
    if not request_id or request_id not in _STORE:
        return
    _STORE[request_id]["tool_calls"].append(call)


def finish(request_id: str | None, error: str | None = None) -> None:
    if not request_id or request_id not in _STORE:
        return
    _STORE[request_id]["done"] = True
    _STORE[request_id]["error"] = error


def get(request_id: str) -> dict | None:
    entry = _STORE.get(request_id)
    if entry is None:
        return None
    return {"tool_calls": entry["tool_calls"], "done": entry["done"], "error": entry["error"]}


def _sweep() -> None:
    now = time.monotonic()
    stale = [key for key, entry in _STORE.items() if now - entry["_started_at"] > _MAX_AGE_SECONDS]
    for key in stale:
        _STORE.pop(key, None)
