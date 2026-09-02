"""Reads one real PHEME thread's reply-tree cascade from the raw dataset.

This is a read-only, per-tweet view (id, parent, text, timestamp offset from
the source tweet) built from data/raw/pheme/{event}-all-rnr-threads. It never
writes to the raw dataset. `observed_by_cutoff` marks whether a tweet is
within the same 30-minute (1800s) observation window used throughout the
research pipeline, so the frontend can render the same early/late distinction
the models are trained and evaluated on.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

CUTOFF_SECONDS = 1800
TWITTER_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def _parse_twitter_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, TWITTER_DATE_FORMAT)
    except ValueError:
        return None


def _thread_dir(base_path: Path, event_id: str, thread_id: str) -> Path | None:
    event_dir = base_path / f"{event_id}-all-rnr-threads"
    for label in ("rumours", "non-rumours"):
        candidate = event_dir / label / thread_id
        if candidate.exists():
            return candidate
    return None


def _load_tweets(thread_dir: Path) -> dict[str, dict]:
    tweets: dict[str, dict] = {}
    for folder_name in ("source-tweets", "reactions"):
        folder = thread_dir / folder_name
        if not folder.exists():
            continue
        for path in folder.glob("*.json"):
            if path.name.startswith("._"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            except (json.JSONDecodeError, OSError):
                continue
            tweet_id = data.get("id_str")
            if not tweet_id:
                continue
            tweets[tweet_id] = {
                "id": tweet_id,
                "text": data.get("text", ""),
                "screen_name": (data.get("user") or {}).get("screen_name", "unknown"),
                "created_at": _parse_twitter_date(data.get("created_at")),
            }
    return tweets


def _load_source_tweet(base_path: Path, event_id: str, thread_id: str) -> dict | None:
    thread_dir = _thread_dir(base_path, event_id, thread_id)
    if thread_dir is None:
        return None
    source_path = thread_dir / "source-tweets" / f"{thread_id}.json"
    if not source_path.exists():
        return None
    try:
        return json.loads(source_path.read_text(encoding="utf-8", errors="ignore"))
    except (json.JSONDecodeError, OSError):
        return None


def load_source_preview(base_path: Path, event_id: str, thread_id: str, max_chars: int = 140) -> str:
    data = _load_source_tweet(base_path, event_id, thread_id)
    return (data.get("text") or "")[:max_chars] if data else ""


def load_source_date(base_path: Path, event_id: str, thread_id: str) -> str | None:
    """Source tweet's own post date (YYYY-MM-DD), used to anchor search queries in time."""
    data = _load_source_tweet(base_path, event_id, thread_id)
    if not data:
        return None
    created_at = _parse_twitter_date(data.get("created_at"))
    return created_at.strftime("%Y-%m-%d") if created_at else None


def load_cascade(base_path: Path, event_id: str, thread_id: str) -> dict:
    thread_dir = _thread_dir(base_path, event_id, thread_id)
    if thread_dir is None:
        raise FileNotFoundError(f"Thread {thread_id} not found under event {event_id}")
    structure_path = thread_dir / "structure.json"
    if not structure_path.exists():
        raise FileNotFoundError(f"structure.json missing for thread {thread_id}")

    tweets = _load_tweets(thread_dir)
    structure = json.loads(structure_path.read_text(encoding="utf-8", errors="ignore"))
    source_created_at = tweets.get(thread_id, {}).get("created_at")

    nodes: list[dict] = []
    edges: list[dict] = []

    def traverse(node_dict, parent_id: str | None) -> None:
        if not node_dict:
            return
        for tweet_id, children in node_dict.items():
            info = tweets.get(tweet_id)
            if info is not None:
                offset_sec = None
                if info["created_at"] is not None and source_created_at is not None:
                    offset_sec = (info["created_at"] - source_created_at).total_seconds()
                is_source = tweet_id == thread_id
                observed = is_source or (offset_sec is not None and 0 <= offset_sec <= CUTOFF_SECONDS)
                nodes.append({
                    "id": tweet_id,
                    "parent_id": parent_id,
                    "is_source": is_source,
                    "screen_name": info["screen_name"],
                    "text": info["text"],
                    "offset_sec": offset_sec,
                    "observed_by_cutoff": observed,
                })
                if parent_id is not None:
                    edges.append({"source": parent_id, "target": tweet_id})
            traverse(children, tweet_id)

    traverse(structure, None)
    return {
        "thread_id": thread_id,
        "event_id": event_id,
        "cutoff_seconds": CUTOFF_SECONDS,
        "nodes": nodes,
        "edges": edges,
    }
