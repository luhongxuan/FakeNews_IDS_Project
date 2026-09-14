"""First real end-to-end smoke of the RF -> verify -> select pipeline for
ONE candidate. Uses a real local RF prediction (synthetic cascade input,
since there is no live radar thread for this claim -- NOT a real
propagation measurement), a REAL verification_agent.run_verification call
(real web search + real local Ollama), the real intervention_policy
selection logic, and writes to the real VerificationReport table (a
distinct test key, left in place as the audit artifact -- not cleaned up,
unlike the stub-only smoke checks).

LABEL, exactly as agreed before running this: this is a TECHNICAL
INTEGRATION SMOKE, not a model-validity result. The RF prediction is
computed from synthetic cascade data (a placeholder engagement shape), not
this claim's real propagation -- pairing a real, well-established claim
with unrelated synthetic RF features cannot demonstrate detection
capability, only that the modules correctly connect, cache, audit, and
apply the confirmed-true safety exclusion.

Produces a RECOMMENDATION ONLY -- calls no hide/downrank/execution path
anywhere. Test claim, agreed with the user specifically because it is
public, stable, and independently verifiable by authoritative sources
(WHO's own announcement), not because of its content -- no private,
project, or PHEME data is sent to any external service.

Run: python e2e_smoke_single_candidate.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv
load_dotenv(BACKEND / ".env")

from app.database import SessionLocal
from app.models.schema import VerificationReport
from app.services import intervention_agent, intervention_policy, verification_agent
from app.services.intervention_features import build_cumulative_features

STARTED_AT = datetime.now(timezone.utc).isoformat()

CLAIM_TEXT = "WHO ended COVID-19's Public Health Emergency of International Concern on 5 May 2023."
POST_DATE = "2023-05-05"
import os
TEST_KEY = os.getenv("E2E_SMOKE_TEST_KEY", "e2e_smoke_who_covid_phe_end_20260905")
CHECKPOINT_MINUTES = 10

# Synthetic cascade -- a placeholder engagement shape (1 source post + 4
# replies within the first 10 minutes), NOT this claim's real propagation.
# See module docstring: this is what makes the RF number "real" (a genuine
# model call) but not meaningful as a propagation forecast for this specific
# claim.
SYNTHETIC_NODES = [
    {"id": "source", "is_source": True, "offset_sec": 0, "parent_id": None, "text": CLAIM_TEXT, "depth": 0},
    {"id": "r1", "is_source": False, "offset_sec": 90, "parent_id": "source", "text": "Is this confirmed?", "depth": 1},
    {"id": "r2", "is_source": False, "offset_sec": 210, "parent_id": "source", "text": "Saw this on the news too.", "depth": 1},
    {"id": "r3", "is_source": False, "offset_sec": 340, "parent_id": "r1", "text": "Yes, WHO announced it.", "depth": 2},
    {"id": "r4", "is_source": False, "offset_sec": 480, "parent_id": "source", "text": "About time.", "depth": 1},
]


async def main() -> dict:
    print("[1/5] Computing real local RF prediction from synthetic cascade input", flush=True)
    rf_result = intervention_agent._run_rf_tool(SYNTHETIC_NODES, thread_age_minutes=CHECKPOINT_MINUTES, checkpoint_minutes=CHECKPOINT_MINUTES)
    print(f"      predicted_impact={rf_result.get('predicted_impact')}", flush=True)

    print("[2/5] Real verification (live web search + local Ollama) -- this is the first real network/LLM call in this pipeline", flush=True)
    db = SessionLocal()
    db.query(VerificationReport).filter_by(thread_id=TEST_KEY).delete()
    db.commit()

    candidates = [{
        "post_uri": "e2e_smoke_post_1",
        "cluster_thread_id": TEST_KEY,
        "claim_text": CLAIM_TEXT,
        "event_id": "e2e_smoke",
        "post_date": POST_DATE,
        "is_high_risk": True,
    }]
    verification_result = await intervention_agent.run_batch_verification(
        db, candidates, checkpoint_minutes=CHECKPOINT_MINUTES, verify_fn=None,  # None -> real verification_agent.run_verification
    )
    evidence = verification_result["results"][TEST_KEY]
    print(f"      credibility={evidence['credibility']} confidence={evidence['confidence']}", flush=True)

    print("[3/5] Deterministic priority + selection (no LLM)", flush=True)
    candidate = intervention_policy.Candidate(
        post_uri="e2e_smoke_post_1",
        cluster_thread_id=TEST_KEY,
        predicted_impact=float(rf_result.get("predicted_impact") or 0.0),
        credibility=evidence["credibility"],
        confidence=evidence["confidence"],
        evidence_reasoning=evidence.get("summary") or "",
    )
    selection = intervention_policy.select_for_checkpoint(
        [candidate], checkpoint_minutes=CHECKPOINT_MINUTES, already_intervened_count=0, total_budget=50,
    )
    if candidate.post_uri in selection.excluded_confirmed_true:
        recommendation = "excluded_confirmed_true (no intervention -- content already confirmed accurate)"
    elif candidate.post_uri in selection.deferred_agent_failure:
        recommendation = "deferred_agent_failure (verification failed -- retry, no decision made)"
    elif any(c.post_uri == candidate.post_uri for c in selection.selected):
        recommendation = f"selected, tier={selection.tier_by_uri.get(candidate.post_uri)}, priority={selection.priority_by_uri.get(candidate.post_uri):.3f}"
    else:
        recommendation = f"not selected this checkpoint (priority={selection.priority_by_uri.get(candidate.post_uri)}, below budget cutoff)"
    print(f"      recommendation: {recommendation}", flush=True)

    print("[4/5] Reading back persisted cache state from VerificationReport", flush=True)
    row = db.query(VerificationReport).filter_by(thread_id=TEST_KEY).first()
    cache_state = {k: row.report_jsonb.get(k) for k in (
        "last_status", "last_checked_at_minutes", "next_check_due_minutes",
        "verification_attempts", "status_version", "resolved", "resolved_at_minutes",
    )} if row else None
    evidence_sources = (row.report_jsonb.get("evidence") if row else None) or []
    queries_used = (row.report_jsonb.get("queries_used") if row else None) or []
    db.close()

    print("[5/5] No hide/downrank/execution action was taken anywhere in this run -- recommendation only.", flush=True)

    return {
        "label": "TECHNICAL INTEGRATION SMOKE -- NOT a model-validity result (RF input is synthetic, unrelated to this claim's real propagation)",
        "verification_backend": {"ollama_api_base": verification_agent.OLLAMA_API_BASE, "ollama_model": verification_agent.OLLAMA_MODEL},
        "test_key": TEST_KEY,
        "claim_text": CLAIM_TEXT,
        "post_date": POST_DATE,
        "checkpoint_minutes": CHECKPOINT_MINUTES,
        "rf_output": rf_result,
        "verification_result": {
            "credibility": evidence["credibility"],
            "confidence": evidence["confidence"],
            "summary": evidence.get("summary"),
            "from_cache": evidence.get("from_cache"),
        },
        "evidence_sources": evidence_sources,
        "queries_used": queries_used,
        "risk_weight": intervention_policy.RISK_WEIGHTS.get(intervention_policy.effective_credibility(evidence["credibility"], evidence["confidence"])),
        "priority": selection.priority_by_uri.get(candidate.post_uri),
        "recommendation": recommendation,
        "excluded_confirmed_true": candidate.post_uri in selection.excluded_confirmed_true,
        "cache_state_after_run": cache_state,
        "action_taken": "none -- recommendation only, no hide/downrank/execution call made",
    }


if __name__ == "__main__":
    outcome = asyncio.run(main())
    completed_at = datetime.now(timezone.utc).isoformat()
    out_dir = Path(__file__).resolve().parents[3] / "effective_models" / "pheme_multicheckpoint_rf" / "experiments" / (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_e2e_smoke_single_candidate"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    run_record = {
        "status": "complete",
        "started_at": STARTED_AT,
        "completed_at": completed_at,
        "script": "backend/app/services/e2e_smoke_single_candidate.py",
        **outcome,
    }
    (out_dir / "run_record.json").write_text(json.dumps(run_record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSUCCESS -- {out_dir}")
    print(json.dumps(outcome, indent=2, ensure_ascii=False))
