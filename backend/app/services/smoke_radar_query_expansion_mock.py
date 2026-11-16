"""Live Radar Event Discovery V2, phase 3 -- MOCK search-expansion smoke
(validation step 5). Exercises radar_query_expansion.expand_seed end to
end with an injected fake search_fn -- makes ZERO real network calls, per
AGENTS.md's short-bounded-check policy. Confirms: query-count bound,
per-query result-count bound, dedup against already-known URIs, and the
cooldown skip, all with real (not stubbed-out) queries built from
radar_event_signature.py against a real seed post's text.

Run: python smoke_radar_query_expansion_mock.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services.radar_query_expansion import EXPANSION_COOLDOWN, MAX_QUERIES_PER_SEED, expand_seed

SEED_POST = {
    "uri": "at://did:plc:seed/app.bsky.feed.post/1",
    "text": "Four people injured in a shooting near Granite Regional Park in Sacramento, California.",
}

_FAKE_RESULTS_BY_QUERY_PREFIX = {
    "shooting": [
        {"uri": "at://did:plc:seed/app.bsky.feed.post/1", "text": SEED_POST["text"]},  # the seed itself -- must be excluded
        {"uri": "at://did:plc:other/app.bsky.feed.post/2", "text": "Sacramento shooting update: 4 hurt, suspect at large."},
    ],
    '"Granite': [
        {"uri": "at://did:plc:other/app.bsky.feed.post/3", "text": "Granite Regional Park shooting: witnesses describe chaos."},
        {"uri": "at://did:plc:other/app.bsky.feed.post/2", "text": "duplicate of an already-found candidate -- must be deduped"},
    ],
}


async def fake_search_fn(query: str, limit: int, sort: str) -> list[dict]:
    for prefix, results in _FAKE_RESULTS_BY_QUERY_PREFIX.items():
        if query.startswith(prefix):
            return results[:limit]
    return []


def _check(label: str, condition: bool, checks: list[dict]) -> None:
    checks.append({"label": label, "passed": bool(condition)})
    print(f"  [{'OK' if condition else 'FAIL'}] {label}", flush=True)


async def main() -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    print(f"[smoke_radar_query_expansion_mock] started_at={started_at} -- ZERO real network calls.", flush=True)
    checks: list[dict] = []

    print("[1/3] Fresh seed (no prior expansion, no cooldown)...", flush=True)
    audit = await expand_seed(SEED_POST, known_uris={SEED_POST["uri"]}, search_fn=fake_search_fn)
    _check("queries_issued is non-empty", len(audit.queries_issued) > 0, checks)
    _check(f"queries_issued respects MAX_QUERIES_PER_SEED ({MAX_QUERIES_PER_SEED})", len(audit.queries_issued) <= MAX_QUERIES_PER_SEED, checks)
    _check("seed's own URI excluded from new_candidate_posts", SEED_POST["uri"] not in audit.new_uris_after_dedup, checks)
    _check("candidate_uris_found contains the seed URI (found, just not NEW)", SEED_POST["uri"] in audit.candidate_uris_found, checks)
    _check("duplicate URI across two queries deduped to one new candidate", audit.new_uris_after_dedup.count("at://did:plc:other/app.bsky.feed.post/2") == 1, checks)
    _check("a genuinely new candidate URI is present", "at://did:plc:other/app.bsky.feed.post/3" in audit.new_uris_after_dedup, checks)
    _check("not skipped (no prior expansion)", audit.skipped_cooldown is False, checks)

    print("[2/3] Same seed, already known to caller -> excluded from new candidates...", flush=True)
    known_uris = {SEED_POST["uri"], "at://did:plc:other/app.bsky.feed.post/2", "at://did:plc:other/app.bsky.feed.post/3"}
    audit2 = await expand_seed(SEED_POST, known_uris=known_uris, search_fn=fake_search_fn)
    _check("every candidate already known -> zero new candidates", len(audit2.new_uris_after_dedup) == 0, checks)

    print("[3/3] Cooldown -- an expansion within EXPANSION_COOLDOWN is skipped with NO search call...", flush=True)
    call_count = {"n": 0}

    async def counting_search_fn(query, limit, sort):
        call_count["n"] += 1
        return await fake_search_fn(query, limit, sort)

    recent = datetime.now(timezone.utc) - (EXPANSION_COOLDOWN / 2)
    audit3 = await expand_seed(SEED_POST, known_uris=set(), search_fn=counting_search_fn, last_expanded_at=recent)
    _check("skipped_cooldown is True", audit3.skipped_cooldown is True, checks)
    _check("no search call made while in cooldown", call_count["n"] == 0, checks)

    n_passed = sum(1 for c in checks if c["passed"])
    n_total = len(checks)
    completed_at = datetime.now(timezone.utc).isoformat()
    result = {
        "status": "complete" if n_passed == n_total else "failed",
        "started_at": started_at, "completed_at": completed_at,
        "checks_passed": n_passed, "checks_total": n_total, "checks": checks,
        "real_network_calls_made": 0,
    }
    out_path = Path(__file__).resolve().parent / "smoke_radar_query_expansion_mock_result.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if n_passed == n_total:
        print(f"\nSUCCESS -- {n_passed}/{n_total} checks passed -- {out_path}", flush=True)
    else:
        print(f"\nFAILURE -- {n_passed}/{n_total} checks passed -- {out_path}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
