"""Live Radar Event Discovery V2, phase 3 -- actually issues the
signature-derived search queries (radar_event_signature.build_expansion_
queries) against Bluesky for one high-engagement SEED source post, to find
OTHER source posts about the same real-world event that the original
fixed-keyword search (bluesky_source.DEFAULT_KEYWORDS) missed.

This module only FINDS and DEDUPES candidates -- it does not decide who
merges with whom (that is event_clustering_v2.classify_cluster_merge, or
v1's assign_post_to_cluster if v2 has not been wired in yet) and it never
writes to RadarThread itself.

Bounded on purpose, per the V2 spec:
- at most MAX_QUERIES_PER_SEED queries per seed (from
  radar_event_signature.build_expansion_queries's own cap);
- at most MAX_RESULTS_PER_QUERY results fetched per query;
- a seed already expanded within EXPANSION_COOLDOWN is skipped entirely
  (no network call at all) until the cooldown elapses -- prevents the same
  high-engagement seed from re-triggering a full expansion round every
  time ingestion happens to see it again.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import httpx

from app.services import bluesky_source
from app.services.radar_event_signature import build_expansion_queries, extract_event_signature

MAX_QUERIES_PER_SEED = 5
MAX_RESULTS_PER_QUERY = 15
EXPANSION_COOLDOWN = timedelta(minutes=30)


@dataclass
class ExpansionAudit:
    seed_uri: str
    queries_issued: list[str] = field(default_factory=list)
    candidate_uris_found: list[str] = field(default_factory=list)
    new_candidate_posts: list[dict] = field(default_factory=list)
    skipped_cooldown: bool = False
    error: str | None = None

    @property
    def new_uris_after_dedup(self) -> list[str]:
        return [post["uri"] for post in self.new_candidate_posts]

    def as_audit_row(self, cluster_thread_id: str | None = None) -> dict:
        """Shape matching schema.RadarQueryExpansionAudit's JSONB columns."""
        return {
            "seed_uri": self.seed_uri,
            "cluster_thread_id": cluster_thread_id,
            "queries_issued": self.queries_issued,
            "candidate_uris_found": self.candidate_uris_found,
            "new_uris_after_dedup": self.new_uris_after_dedup,
            "skipped_cooldown": self.skipped_cooldown,
            "error": self.error,
        }


async def expand_seed(
    seed_post: dict,
    known_uris: set[str],
    search_fn=None,
    last_expanded_at: datetime | None = None,
    now: datetime | None = None,
    max_queries: int = MAX_QUERIES_PER_SEED,
    max_results_per_query: int = MAX_RESULTS_PER_QUERY,
    cooldown: timedelta = EXPANSION_COOLDOWN,
) -> ExpansionAudit:
    """`seed_post` needs at least `uri`/`text`. `known_uris` is every post
    URI already ingested (across all clusters) -- a result matching one of
    these is dropped from `new_candidate_posts` (still recorded in
    `candidate_uris_found` for the audit trail, so a caller can see "found
    but already known" vs. "genuinely new" separately).

    `search_fn` is injectable (defaults to bluesky_source.search_recent_
    posts) purely so tests and the mock-search smoke check never make a
    real network call -- same pattern as intervention_agent.py's
    `verify_fn` injection.
    """
    seed_uri = seed_post["uri"]
    now = now or datetime.now(timezone.utc)
    if last_expanded_at is not None and (now - last_expanded_at) < cooldown:
        return ExpansionAudit(seed_uri=seed_uri, skipped_cooldown=True)

    search_fn = search_fn or bluesky_source.search_recent_posts
    signature = extract_event_signature(seed_post.get("text", ""))
    queries = build_expansion_queries(signature, max_queries=max_queries)

    found: list[dict] = []
    error: str | None = None
    for query in queries:
        try:
            results = await search_fn(query, limit=max_results_per_query, sort="top")
        except httpx.HTTPError as exc:
            # One failing query must not discard the queries that already
            # succeeded -- record the failure for the audit trail and move on.
            error = f"query {query!r} failed: {exc}" if error is None else error
            continue
        found.extend(results)

    seen_uris: set[str] = set()
    candidate_uris_found: list[str] = []
    new_candidate_posts: list[dict] = []
    for post in found:
        uri = post.get("uri")
        if not uri or uri in seen_uris:
            continue
        seen_uris.add(uri)
        candidate_uris_found.append(uri)
        if uri != seed_uri and uri not in known_uris:
            new_candidate_posts.append(post)

    return ExpansionAudit(
        seed_uri=seed_uri,
        queries_issued=queries,
        candidate_uris_found=candidate_uris_found,
        new_candidate_posts=new_candidate_posts,
        error=error,
    )
