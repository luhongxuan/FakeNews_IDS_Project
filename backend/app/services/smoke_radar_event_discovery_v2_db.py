"""Live Radar Event Discovery V2 -- DB smoke (validation step 6). Uses the
REAL Postgres DB (VerificationReport / RadarThread tables), a STUB
verify_fn (no real LLM/network call, same injection pattern as
smoke_batch_verification_db.py), and cleans up every row it writes.

Specifically exercises the phase-5 cache-key-consistency fix in
app/routers/radar.py's manual_policy_decision: replicates that endpoint's
own verification_cache_key selection logic (ambiguous cluster -> per-post
key; consistent cluster -> the SAME hashed key intervention_agent.
is_confirmed_true/get_verification_status_for already use) against real
DB rows, without needing a live Bluesky fetch or a real Ollama call.

Run: python smoke_radar_event_discovery_v2_db.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("DATABASE_URL", "postgresql://admin:secret@localhost:15432/misinfo_db")

from app.database import SessionLocal
from app.models.schema import RadarThread, VerificationReport
from app.routers.radar import _verify_cache_key
from app.services import intervention_agent
from app.services.event_clustering_v2 import is_cluster_claim_ambiguous
from app.services.radar_event_signature import extract_event_signature

TEST_KEY_PREFIX = "smoke_test_radar_v2_20260907_"


def _test_uri(name: str) -> str:
    return f"at://did:plc:{TEST_KEY_PREFIX}/app.bsky.feed.post/{name}"


AMBIGUOUS_MEMBERS = [
    {"uri": _test_uri("indiana"), "text": "Indiana: 3 injured in a shooting on Indianapolis's near west side.",
     "created_at": "2026-09-07T05:00:00Z", "reply_count": 1, "repost_count": 0},
    {"uri": _test_uri("texas"), "text": "Mass shooting: shooter dead, 3 wounded in a shooting spree in Pasadena, Texas.",
     "created_at": "2026-09-07T05:05:00Z", "reply_count": 1, "repost_count": 0},
]
CONSISTENT_MEMBERS = [
    {"uri": _test_uri("payson_1"), "text": "Payson Fire burns near Elk Ridge, Utah, with evacuation updates and map.",
     "created_at": "2026-09-07T05:00:00Z", "reply_count": 1, "repost_count": 0},
    {"uri": _test_uri("payson_2"), "text": "Payson Fire: Map, evacuation updates as wildfire burns near Elk Ridge, Utah.",
     "created_at": "2026-09-07T05:05:00Z", "reply_count": 1, "repost_count": 0},
]


async def _stub_verify(session, key, event_id, claim_text, post_date, **_kw):
    # Deterministic, content-free stub -- no real LLM/network call. Returns a
    # DIFFERENT credibility depending on which post's text was actually passed in, so
    # the test can prove which claim_text a given verification call actually saw.
    if "Indianapolis" in claim_text:
        return {"credibility": "likely_false", "confidence": 0.9, "summary": "stub: indiana"}
    if "Pasadena" in claim_text:
        return {"credibility": "likely_true", "confidence": 0.9, "summary": "stub: texas"}
    return {"credibility": "likely_false", "confidence": 0.9, "summary": "stub: payson fire"}


def _make_test_thread(db, members: list[dict]) -> RadarThread:
    thread = RadarThread(
        representative_text=members[0]["text"], representative_uri=members[0]["uri"],
        representative_created_at=datetime(2026, 9, 7, 5, 0, 0),
        matched_keywords=[TEST_KEY_PREFIX], members=members,
        centroid_embedding=[0.0] * 384,  # never read by anything this smoke exercises
        first_seen_at=datetime.now(timezone.utc), last_seen_at=datetime.now(timezone.utc),
    )
    db.add(thread)
    db.commit()
    db.refresh(thread)
    return thread


def _check(label: str, condition: bool, checks: list[dict]) -> None:
    checks.append({"label": label, "passed": bool(condition)})
    print(f"  [{'OK' if condition else 'FAIL'}] {label}", flush=True)


async def main() -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    print(f"[smoke_radar_event_discovery_v2_db] started_at={started_at}", flush=True)
    print("      Uses the real DB; STUB verify_fn only -- zero real LLM/network calls.", flush=True)
    checks: list[dict] = []
    db = SessionLocal()
    created_thread_ids: list[str] = []
    written_verification_keys: list[str] = []

    try:
        print("[1/4] Baseline: count real (non-test) RadarThread rows before this smoke runs...", flush=True)
        baseline_count = db.query(RadarThread).filter(~RadarThread.matched_keywords.contains([TEST_KEY_PREFIX])).count()

        print("[2/4] Ambiguous cluster -> per-post verification cache keys never collide...", flush=True)
        ambiguous_thread = _make_test_thread(db, AMBIGUOUS_MEMBERS)
        created_thread_ids.append(str(ambiguous_thread.id))
        member_signatures = [extract_event_signature(m["text"]) for m in ambiguous_thread.members]
        is_ambiguous = is_cluster_claim_ambiguous(member_signatures)
        _check("cluster is detected as ambiguous", is_ambiguous, checks)

        thread_id = str(ambiguous_thread.id)
        for member in AMBIGUOUS_MEMBERS:
            key = _verify_cache_key(f"{thread_id}:{member['uri']}") if is_ambiguous else _verify_cache_key(thread_id)
            written_verification_keys.append(key)
            batch = await intervention_agent.run_batch_verification(
                db, [{"post_uri": member["uri"], "cluster_thread_id": key, "claim_text": member["text"],
                      "event_id": "radar", "post_date": None, "is_high_risk": True}],
                checkpoint_minutes=10, verify_fn=_stub_verify,
            )
            evidence = batch["results"][key]
            member["_evidence"] = evidence

        indiana_evidence = AMBIGUOUS_MEMBERS[0]["_evidence"]
        texas_evidence = AMBIGUOUS_MEMBERS[1]["_evidence"]
        _check("per-post keys are distinct", written_verification_keys[0] != written_verification_keys[1], checks)
        _check("indiana post's own evidence reflects ITS OWN claim_text (likely_false)", indiana_evidence["credibility"] == "likely_false", checks)
        _check("texas post's own evidence reflects ITS OWN claim_text (likely_true), not indiana's", texas_evidence["credibility"] == "likely_true", checks)
        rows = db.query(VerificationReport).filter(VerificationReport.thread_id.in_(written_verification_keys)).all()
        _check("two SEPARATE VerificationReport rows were written (not one shared row)", len(rows) == 2, checks)

        print("[3/4] Consistent cluster -> the SAME key intervention_agent.is_confirmed_true reads from...", flush=True)
        consistent_thread = _make_test_thread(db, CONSISTENT_MEMBERS)
        created_thread_ids.append(str(consistent_thread.id))
        consistent_thread_id = str(consistent_thread.id)
        member_signatures_2 = [extract_event_signature(m["text"]) for m in consistent_thread.members]
        is_ambiguous_2 = is_cluster_claim_ambiguous(member_signatures_2)
        _check("consistent cluster is NOT detected as ambiguous", not is_ambiguous_2, checks)

        cluster_key = _verify_cache_key(consistent_thread_id)
        written_verification_keys.append(cluster_key)
        target = CONSISTENT_MEMBERS[0]
        batch2 = await intervention_agent.run_batch_verification(
            db, [{"post_uri": target["uri"], "cluster_thread_id": cluster_key, "claim_text": target["text"],
                  "event_id": "radar", "post_date": None, "is_high_risk": True}],
            checkpoint_minutes=10, verify_fn=_stub_verify,
        )
        # This is the ACTUAL regression this phase's fix addresses: before the fix,
        # manual_policy_decision wrote under thread_id (raw) while every other pathway
        # (is_confirmed_true / get_verification_status_for / the /verify endpoint) read
        # from _verify_cache_key(thread_id) -- two caches that never saw each other. Now
        # both read the SAME row.
        status_seen_by_other_pathways = intervention_agent.get_verification_status_for(db, consistent_thread_id)
        _check(
            "get_verification_status_for (used by is_confirmed_true, the scheduler, and "
            "run_intervention_review) now sees the SAME verification this endpoint just wrote",
            status_seen_by_other_pathways.get("credibility") == batch2["results"][cluster_key]["credibility"],
            checks,
        )

        print("[4/4] Cleanup...", flush=True)
    finally:
        db.query(VerificationReport).filter(VerificationReport.thread_id.in_(written_verification_keys)).delete(synchronize_session=False)
        db.query(RadarThread).filter(RadarThread.id.in_(created_thread_ids)).delete(synchronize_session=False)
        db.commit()
        remaining_test_rows = db.query(RadarThread).filter(RadarThread.matched_keywords.contains([TEST_KEY_PREFIX])).count()
        final_baseline_count = db.query(RadarThread).filter(~RadarThread.matched_keywords.contains([TEST_KEY_PREFIX])).count()
        _check("all test RadarThread rows removed", remaining_test_rows == 0, checks)
        _check("pre-existing (non-test) RadarThread rows untouched", final_baseline_count == baseline_count, checks)
        db.close()

    n_passed = sum(1 for c in checks if c["passed"])
    n_total = len(checks)
    result = {
        "status": "complete" if n_passed == n_total else "failed",
        "started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(),
        "checks_passed": n_passed, "checks_total": n_total, "checks": checks,
        "real_llm_or_network_calls_made": 0,
    }
    out_path = Path(__file__).resolve().parent / "smoke_radar_event_discovery_v2_db_result.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if n_passed == n_total:
        print(f"\nSUCCESS -- {n_passed}/{n_total} checks passed -- {out_path}", flush=True)
    else:
        print(f"\nFAILURE -- {n_passed}/{n_total} checks passed -- {out_path}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
