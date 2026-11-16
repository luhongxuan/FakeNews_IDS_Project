"""Smoke check: run_batch_verification against the REAL Postgres DB
(VerificationReport table), using a stub verify_fn (no real LLM/network
call) -- validates persistence, the atomic upsert, cluster dedup, and
confidence-gated caching behavior that only a real DB round-trip can prove
(the pure-Python unit tests in test_intervention_policy_smoke.py cover the
same logic in-memory, but not actual read-back through Postgres). Cleans
up every row it writes. Saves a full run record per AGENTS.md Section 7,
not just a terminal success line.

Run: python smoke_batch_verification_db.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv
# This script runs on the host, while backend/.env's `db:5432` hostname is
# only resolvable inside docker-compose. Match backend/dev_server.py's
# established host-side default without overriding an explicit caller value.
os.environ.setdefault("DATABASE_URL", "postgresql://admin:secret@localhost:15432/misinfo_db")
load_dotenv(BACKEND / ".env")

from app.database import SessionLocal
from app.models.schema import VerificationReport
from app.services import intervention_agent

TEST_KEY_PREFIX = "smoke_test_quota_cache_20260905_"
STARTED_AT = datetime.now(timezone.utc).isoformat()


def _cleanup(db, keys: list[str]) -> None:
    db.query(VerificationReport).filter(VerificationReport.thread_id.in_(keys)).delete(synchronize_session=False)
    db.commit()


async def main() -> dict:
    checks: list[dict] = []
    db = SessionLocal()
    keys = [f"{TEST_KEY_PREFIX}{name}" for name in ("persist", "cluster_shared")]
    _cleanup(db, keys)

    def record(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        status = "OK" if passed else "FAIL"
        print(f"  [{status}] {name}{(' -- ' + detail) if detail else ''}", flush=True)

    try:
        # --- Check 1: persist + resolved cache hit on a high-confidence result ---
        print("[1/6] Persistence + cache-hit round trip (high-confidence likely_false)", flush=True)
        persist_key = f"{TEST_KEY_PREFIX}persist"
        call_count = {"n": 0}

        async def stub_high_confidence(db, key, event_id, claim_text, post_date):
            call_count["n"] += 1
            return {"credibility": "likely_false", "confidence": 0.9, "summary": "stub verification result"}

        candidates = [{"post_uri": "post1", "cluster_thread_id": persist_key, "claim_text": "test claim", "event_id": "smoke_test"}]
        result1 = await intervention_agent.run_batch_verification(db, candidates, checkpoint_minutes=10, verify_fn=stub_high_confidence)
        record("first_call_hits_stub_once", call_count["n"] == 1, f"call_count={call_count['n']}")
        record("first_call_returns_likely_false", result1["results"][persist_key]["credibility"] == "likely_false")
        row = db.query(VerificationReport).filter_by(thread_id=persist_key).first()
        record("row_persisted", row is not None)
        record("persisted_resolved_true", bool(row and row.report_jsonb.get("resolved") is True))

        result2 = await intervention_agent.run_batch_verification(db, candidates, checkpoint_minutes=20, verify_fn=stub_high_confidence)
        record("second_call_is_cache_hit", result2["results"][persist_key]["from_cache"] is True)
        record("stub_not_called_again", call_count["n"] == 1, f"call_count={call_count['n']}")

        # --- Check 2: low-confidence result must NOT be cached as resolved ---
        print("[2/6] Low-confidence result must not resolve/lock the cache", flush=True)
        low_conf_key = f"{TEST_KEY_PREFIX}lowconf"
        _cleanup(db, [low_conf_key])
        low_conf_calls = {"n": 0}

        async def stub_low_confidence(db, key, event_id, claim_text, post_date):
            low_conf_calls["n"] += 1
            return {"credibility": "likely_false", "confidence": 0.2, "summary": "weak evidence"}

        candidates_lc = [{"post_uri": "postlc", "cluster_thread_id": low_conf_key, "claim_text": "x", "event_id": "smoke_test"}]
        await intervention_agent.run_batch_verification(db, candidates_lc, checkpoint_minutes=10, verify_fn=stub_low_confidence)
        row_lc = db.query(VerificationReport).filter_by(thread_id=low_conf_key).first()
        record("low_confidence_not_resolved", bool(row_lc and row_lc.report_jsonb.get("resolved") is False))
        record("low_confidence_downgraded_to_unverified", bool(row_lc and row_lc.report_jsonb.get("credibility") == "unverified"))
        await intervention_agent.run_batch_verification(db, candidates_lc, checkpoint_minutes=30, verify_fn=stub_low_confidence)
        record("low_confidence_rechecked_later", low_conf_calls["n"] == 2, f"calls={low_conf_calls['n']}")

        # --- Check 3: cluster dedup persists once even with multiple posts ---
        print("[3/6] Same cluster, multiple posts -> single DB row, single stub call", flush=True)
        cluster_key = f"{TEST_KEY_PREFIX}cluster_shared"
        cluster_calls = {"n": 0}

        async def stub_cluster(db, key, event_id, claim_text, post_date):
            cluster_calls["n"] += 1
            return {"credibility": "unverified", "confidence": 0.0, "summary": "stub"}

        candidates_cluster = [
            {"post_uri": "postA", "cluster_thread_id": cluster_key, "claim_text": "x", "event_id": "smoke_test"},
            {"post_uri": "postB", "cluster_thread_id": cluster_key, "claim_text": "x", "event_id": "smoke_test"},
        ]
        await intervention_agent.run_batch_verification(db, candidates_cluster, checkpoint_minutes=10, verify_fn=stub_cluster)
        record("shared_cluster_verified_once", cluster_calls["n"] == 1, f"calls={cluster_calls['n']}")
        rows_for_cluster = db.query(VerificationReport).filter_by(thread_id=cluster_key).count()
        record("shared_cluster_single_db_row", rows_for_cluster == 1, f"rows={rows_for_cluster}")

        # --- Check 4: agent_failure never persisted as a checked/resolved row ---
        print("[4/6] Permanently failing verification must not be cached as checked", flush=True)
        fail_key = f"{TEST_KEY_PREFIX}failure"
        _cleanup(db, [fail_key])

        async def always_fails(db, key, event_id, claim_text, post_date):
            raise RuntimeError("simulated permanent failure")

        candidates_fail = [{"post_uri": "postfail", "cluster_thread_id": fail_key, "claim_text": "x", "event_id": "smoke_test"}]
        result_fail = await intervention_agent.run_batch_verification(db, candidates_fail, checkpoint_minutes=10, verify_fn=always_fails)
        record("agent_failure_reported", result_fail["results"][fail_key]["credibility"] == "agent_failure")
        row_fail = db.query(VerificationReport).filter_by(thread_id=fail_key).first()
        record("agent_failure_not_persisted", row_fail is None, "a failed verification must not create a VerificationReport row")

        # --- Check 5: concurrent verifications get distinct, real, independently-closed sessions ---
        print("[5/6] Two concurrent clusters -> two distinct DB sessions; one fails and rolls back, the other commits unaffected", flush=True)
        ok_key = f"{TEST_KEY_PREFIX}concurrent_ok"
        fail2_key = f"{TEST_KEY_PREFIX}concurrent_fail"
        _cleanup(db, [ok_key, fail2_key])
        seen_session_ids: dict[str, int] = {}
        seen_session_active: dict[str, bool] = {}

        async def concurrent_stub(session, key, event_id, claim_text, post_date):
            seen_session_ids[key] = id(session)
            seen_session_active[key] = session is not None and session.is_active
            if key == fail2_key:
                raise RuntimeError("simulated mid-verification failure on one of two concurrent calls")
            return {"credibility": "likely_false", "confidence": 0.9, "summary": "stub"}

        candidates_concurrent = [
            {"post_uri": "p_ok", "cluster_thread_id": ok_key, "claim_text": "x", "event_id": "smoke_test"},
            {"post_uri": "p_fail", "cluster_thread_id": fail2_key, "claim_text": "x", "event_id": "smoke_test"},
        ]
        result_concurrent = await intervention_agent.run_batch_verification(
            db, candidates_concurrent, checkpoint_minutes=10, verify_fn=concurrent_stub,
        )
        record("two_sessions_seen", len(seen_session_ids) >= 2, f"keys_seen={list(seen_session_ids)}")
        record("sessions_are_distinct_objects", seen_session_ids.get(ok_key) != seen_session_ids.get(fail2_key))
        record("sessions_not_the_parent_session", id(db) not in seen_session_ids.values())
        record("both_sessions_were_active_during_call", all(seen_session_active.values()))
        record("ok_cluster_committed_despite_sibling_failure", result_concurrent["results"][ok_key]["credibility"] == "likely_false")
        row_ok = db.query(VerificationReport).filter_by(thread_id=ok_key).first()
        record("ok_cluster_row_persisted", row_ok is not None)
        row_failed = db.query(VerificationReport).filter_by(thread_id=fail2_key).first()
        record("failed_cluster_not_persisted_after_retry_exhausted", row_failed is None)

        # --- Check 6: two scheduler invocations, same cluster -> one external call ---
        print("[6/6] Same cluster in two schedulers -> advisory lock permits one external verification", flush=True)
        lock_key = f"{TEST_KEY_PREFIX}inflight_lock"
        _cleanup(db, [lock_key])
        parent_one = SessionLocal()
        parent_two = SessionLocal()
        verifier_entered = asyncio.Event()
        allow_verifier_to_finish = asyncio.Event()
        inflight_calls = {"n": 0}

        async def slow_stub(session, key, event_id, claim_text, post_date):
            inflight_calls["n"] += 1
            verifier_entered.set()
            await allow_verifier_to_finish.wait()
            return {"credibility": "likely_false", "confidence": 0.9, "summary": "single winning verifier"}

        lock_candidate = [{
            "post_uri": "post-lock", "cluster_thread_id": lock_key,
            "claim_text": "x", "event_id": "smoke_test",
        }]
        winner_task = asyncio.create_task(
            intervention_agent.run_batch_verification(
                parent_one, lock_candidate, checkpoint_minutes=10, verify_fn=slow_stub,
            )
        )
        await verifier_entered.wait()
        loser_result = await intervention_agent.run_batch_verification(
            parent_two, lock_candidate, checkpoint_minutes=10, verify_fn=slow_stub,
        )
        record("second_scheduler_deferred_inflight", loser_result["inflight_deferred"] == [lock_key])
        record("inflight_is_not_agent_failure", loser_result["agent_failures"] == [])
        record("loser_did_not_start_external_call", inflight_calls["n"] == 1, f"calls={inflight_calls['n']}")
        allow_verifier_to_finish.set()
        winner_result = await winner_task
        record(
            "winner_completed_normally",
            winner_result["checked_now"] == [lock_key],
            f"winner_result={winner_result!r}",
        )
        parent_two.expire_all()
        cached_result = await intervention_agent.run_batch_verification(
            parent_two, lock_candidate, checkpoint_minutes=20, verify_fn=slow_stub,
        )
        record("later_scheduler_reads_committed_cache", cached_result["results"][lock_key]["from_cache"] is True)
        record("still_exactly_one_external_call", inflight_calls["n"] == 1, f"calls={inflight_calls['n']}")
        parent_one.close()
        parent_two.close()

        all_test_keys = [persist_key, low_conf_key, cluster_key, fail_key, ok_key, fail2_key, lock_key]
        _cleanup(db, all_test_keys)
        db.close()

    except Exception as error:
        db.rollback()
        _cleanup(db, keys + [f"{TEST_KEY_PREFIX}lowconf", f"{TEST_KEY_PREFIX}cluster_shared", f"{TEST_KEY_PREFIX}failure",
                              f"{TEST_KEY_PREFIX}concurrent_ok", f"{TEST_KEY_PREFIX}concurrent_fail",
                              f"{TEST_KEY_PREFIX}inflight_lock"])
        db.close()
        completed_at = datetime.now(timezone.utc).isoformat()
        return {"status": "failed", "started_at": STARTED_AT, "completed_at": completed_at, "checks": checks, "error": repr(error)}

    completed_at = datetime.now(timezone.utc).isoformat()
    passed = all(c["passed"] for c in checks)
    return {"status": "complete" if passed else "failed", "started_at": STARTED_AT, "completed_at": completed_at, "checks": checks}


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[3] / "effective_models" / "pheme_multicheckpoint_rf" / "experiments" / (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_intervention_pipeline_db_smoke"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Starting intervention pipeline DB smoke.", flush=True)
    print(f"Output directory: {out_dir.resolve()}", flush=True)
    running_record = {
        "status": "running",
        "started_at": STARTED_AT,
        "completed_at": None,
        "script": "backend/app/services/smoke_batch_verification_db.py",
        "db_target": "VerificationReport table; explicit DATABASE_URL if supplied, otherwise host-side localhost:15432 default matching backend/dev_server.py",
        "stub_config": "All verify_fn calls in this script are stubs -- no real LLM or web-search call was made",
        "checks": [],
        "output_files": ["run_record.json"],
    }
    (out_dir / "run_record.json").write_text(json.dumps(running_record, indent=2), encoding="utf-8")
    try:
        outcome = asyncio.run(main())
    except Exception as error:
        outcome = {
            "status": "failed", "started_at": STARTED_AT,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "checks": [], "error": repr(error),
        }
    run_record = {
        **outcome,
        "script": "backend/app/services/smoke_batch_verification_db.py",
        "db_target": "VerificationReport table; explicit DATABASE_URL if supplied, otherwise host-side localhost:15432 default matching backend/dev_server.py",
        "stub_config": "All verify_fn calls in this script are stubs -- no real LLM or web-search call was made",
        "checks_total": len(outcome["checks"]),
        "checks_passed": sum(1 for c in outcome["checks"] if c["passed"]),
        "output_files": ["run_record.json"],
    }
    (out_dir / "run_record.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    n_passed = run_record["checks_passed"]
    n_total = run_record["checks_total"]
    if outcome["status"] == "complete":
        print(f"\nSUCCESS -- {n_passed}/{n_total} checks passed -- {out_dir}")
    else:
        print(f"\nFAILURE -- {n_passed}/{n_total} checks passed -- {out_dir}")
        sys.exit(1)
