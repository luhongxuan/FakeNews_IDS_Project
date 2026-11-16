"""Free, keyless live post source via Bluesky's public AppView (AT Protocol).

`https://api.bsky.app` serves public reads (search, thread lookup) with no
API key and no account required -- confirmed reachable and unauthenticated
in this environment, unlike Twitter/X (paid API), Reddit (now requires
approval under its Responsible Builder Policy), and GDELT (network-
unreachable here). Bluesky's reply structure (root/parent/replies, each
with a timestamp) is topologically the same shape as the Twitter reply
trees PHEME was built from, so this is the closest available live analogue
-- not a replacement for real propagation-model retraining, just a
keyword-search radar over live public posts.

This module is a live monitoring data source only. It is never used to
train or evaluate the early-detection models, and nothing here writes back
into effective_models/ or the PHEME pipeline.
"""
from __future__ import annotations

import httpx

APPVIEW_BASE = "https://api.bsky.app/xrpc"
USER_AGENT = "Mozilla/5.0 (compatible; MisinfoResearchRadar/1.0; research use)"

# English breaking-news / disaster / attack keywords, matching the event
# types PHEME itself covers (shootings, hostage sieges, plane crashes,
# explosions) so the radar's content domain lines up with what the v5
# model was actually trained to recognize risk patterns for.
DEFAULT_KEYWORDS = [
    "breaking news",
    "shooting",
    "explosion",
    "gunman",
    "hostage",
    "plane crash",
    "earthquake",
    "wildfire evacuation",
    # Added 2026-09-07 (user request) -- more surface area for the live radar
    # to catch a story that's actually blowing up right now, still within
    # PHEME's own event domain: "riot"/"protest" match Ferguson's own event
    # type (protest/unrest following a shooting), "flood"/"wildfire" extend
    # the existing natural-disaster coverage (earthquake, wildfire evacuation).
    "flood",
    "wildfire",
    "protest",
    "riot",
]


async def search_recent_posts(query: str, limit: int = 15, sort: str = "latest") -> list[dict]:
    """``sort="latest"`` catches posts the moment they're made -- genuinely
    "just happened", but almost always with zero replies yet, since no time
    has passed. ``sort="top"`` catches posts Bluesky judges as already
    engaged-with, which is what actually has a propagation tree worth
    looking at. Both are real, valid views of "what's circulating"; which
    one is more useful depends on whether the point is catching something
    early or seeing something that's already spreading, so callers choose
    rather than this module picking for them.
    """
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=15) as client:
        response = await client.get(
            f"{APPVIEW_BASE}/app.bsky.feed.searchPosts",
            params={"q": query, "limit": limit, "sort": sort},
        )
        response.raise_for_status()
    data = response.json()
    posts = []
    for post in data.get("posts", []):
        record = post.get("record") or {}
        if record.get("reply"):
            continue  # keep source posts only; replies are counted via replyCount
        posts.append({
            "uri": post["uri"],
            "author_handle": (post.get("author") or {}).get("handle", "unknown"),
            "text": record.get("text", ""),
            "created_at": record.get("createdAt"),
            "indexed_at": post.get("indexedAt"),
            "reply_count": post.get("replyCount", 0),
            "repost_count": post.get("repostCount", 0),
            "like_count": post.get("likeCount", 0),
        })
    return posts


async def get_post_thread(uri: str) -> dict:
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=15) as client:
        response = await client.get(
            f"{APPVIEW_BASE}/app.bsky.feed.getPostThread",
            params={"uri": uri},
        )
        response.raise_for_status()
    return response.json()


CUTOFF_SECONDS = 1800


def parse_iso(value: str | None):
    from datetime import datetime
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _walk_replies(
    node: dict, parent_id: str | None, root_created_at, nodes: list[dict], edges: list[dict], depth: int = 1
) -> None:
    for reply in node.get("replies", []) or []:
        post = reply.get("post")
        if not post:
            continue  # blocked/not-found reply, no content to reason about
        record = post.get("record") or {}
        created_at = parse_iso(record.get("createdAt"))
        offset_sec = (created_at - root_created_at).total_seconds() if (created_at and root_created_at) else None
        observed = offset_sec is not None and 0 <= offset_sec <= CUTOFF_SECONDS
        nodes.append({
            "id": post["uri"],
            "parent_id": parent_id,
            "is_source": False,
            "screen_name": (post.get("author") or {}).get("handle", "unknown"),
            "text": record.get("text", ""),
            "offset_sec": offset_sec,
            "observed_by_cutoff": observed,
            "depth": depth,
        })
        if parent_id is not None:
            edges.append({"source": parent_id, "target": post["uri"]})
        _walk_replies(reply, post["uri"], root_created_at, nodes, edges, depth + 1)


def build_cascade_graph(thread_json: dict, root_uri: str, root_created_at) -> dict:
    """Same {nodes, edges} shape as app.services.pheme_cascade.load_cascade,
    built from Bluesky's real reply tree instead of PHEME's archived one --
    lets the existing cascade graph UI (react-cytoscapejs + graphConfig.js)
    render live threads with zero new frontend graph code. This is just a
    picture of what has actually happened so far; it carries no claim about
    the future (see estimate_deletion_impact for that distinction).

    Each node also carries `depth` (reply-chain distance from the root) --
    unused by the graph UI itself, but needed as-is by
    intervention_features.py, which feeds this same structure into the
    Hawkes-RF's cumulative feature set and expects a depth field matching
    the one PHEME's reply table provides.
    """
    root = (thread_json.get("thread") or {}).get("post") or {}
    root_record = root.get("record") or {}
    nodes: list[dict] = [{
        "id": root_uri,
        "parent_id": None,
        "is_source": True,
        "screen_name": (root.get("author") or {}).get("handle", "unknown"),
        "text": root_record.get("text", ""),
        "offset_sec": 0,
        "observed_by_cutoff": True,
        "depth": 0,
    }]
    edges: list[dict] = []
    _walk_replies(thread_json.get("thread", {}), root_uri, root_created_at, nodes, edges)
    return {"nodes": nodes, "edges": edges}


def estimate_deletion_impact(representative_created_at, current_reply_count: int, horizon_minutes: int = 30) -> dict:
    """A genuine estimate, not a peek at the real future: extrapolates the
    reply rate observed so far and projects it forward. Deliberately does
    NOT call get_post_thread to check what "actually" happened next --
    unlike PHEME's fully-elapsed historical threads, a live post's future
    hasn't happened yet (or has only partly happened), so a monitoring
    dashboard for it should reason the same way a real operator would: from
    a trend, not from hindsight.
    """
    from datetime import datetime, timezone
    if representative_created_at is None:
        return {"estimable": False}
    if representative_created_at.tzinfo is None:
        representative_created_at = representative_created_at.replace(tzinfo=timezone.utc)
    age_minutes = max((datetime.now(timezone.utc) - representative_created_at).total_seconds() / 60, 0.5)
    rate_per_minute = current_reply_count / age_minutes
    projected_additional = round(rate_per_minute * horizon_minutes)
    return {
        "estimable": True,
        "age_minutes": round(age_minutes, 1),
        "current_reply_count": current_reply_count,
        "rate_per_minute": round(rate_per_minute, 3),
        "horizon_minutes": horizon_minutes,
        "projected_additional_replies": projected_additional,
    }
