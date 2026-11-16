"""WebSocket endpoint for social-frontend's event bridge.

Contract is fixed by social-frontend/src/lib/backend.js (createEventBridge,
makePostEvent/makeShareEvent/makeCommentEvent): the frontend connects to
ws://localhost:8000/ws/events, sends JSON-encoded post/share/comment events
as text frames, queues locally and retries with backoff while
disconnected, and flushes the queue on (re)connect. This endpoint accepts
that connection, persists every event, and broadcasts it to any other
connected clients (e.g. a future relationship-graph frontend listening on
the same endpoint) so state stays live across all connected views.

This does not yet compute event_cluster_id (semantic clustering) or serve
a "current full state" snapshot for a client connecting after events have
already been sent -- both are called out as separate follow-ups in
social-frontend/README.md's "待後端提供" list.
"""
from __future__ import annotations

from datetime import datetime
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.database import SessionLocal
from app.models.schema import PropagationEvent

router = APIRouter()


class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active:
            self.active.remove(websocket)

    async def broadcast(self, message: dict, exclude: WebSocket | None = None) -> None:
        dead = []
        for connection in self.active:
            if connection is exclude:
                continue
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        for connection in dead:
            self.disconnect(connection)


manager = ConnectionManager()


def _parse_ts(value) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _persist(event: dict) -> None:
    db = SessionLocal()
    try:
        db.add(PropagationEvent(
            client_event_id=event.get("id"),
            type=event.get("type"),
            thread_id=event.get("thread_id"),
            source_post_id=event.get("source_post_id"),
            from_user=event.get("from"),
            to_user=event.get("to"),
            author_user=event.get("user"),
            content=event.get("content"),
            is_rumour=event.get("is_rumour"),
            event_cluster_id=event.get("event_cluster_id"),
            event_ts=_parse_ts(event.get("ts")),
            raw_jsonb=event,
        ))
        db.commit()
    finally:
        db.close()


@router.websocket("/ws/events")
async def events_websocket(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or "type" not in event:
                continue
            _persist(event)
            await manager.broadcast(event, exclude=websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
