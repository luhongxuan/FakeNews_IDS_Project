"""Intervention-decision agent.

Where verification_agent.py decides "is this claim true", this agent
decides "how urgently does this thread's *propagation* need attention, and
is it even eligible for one of our limited intervention slots". Two modes:

- run_intervention_review: single-thread triage, used by the on-demand
  "review this one now" endpoint. Independent per-thread judgment.
- run_batch_selection: the scheduler's real mode. A fixed number of
  intervention slots exist per checkpoint tier (see
  intervention_model.CHECKPOINT_QUOTAS) and in total
  (intervention_model.TOTAL_BUDGET) -- this is a budget-constrained
  *selection* problem, not N independent yes/no decisions, so the agent
  sees every eligible candidate at once and has to choose which ones are
  worth spending scarce slots on, and say why the others didn't make the
  cut. Threads already confirmed true never reach this agent at all --
  see is_confirmed_true, enforced by the caller (radar.py) before this
  module is even invoked, so an intervention can never target content
  already known to be accurate.

Both modes only ever see reply-timing/depth/text-shape features from the
Balanced Cumulative multi-checkpoint RF (effective_models/
pheme_multicheckpoint_rf, served via intervention_model.py) -- it has no
idea whether content is true, satire, or already debunked. That's why
get_verification_status exists as a separate tool/signal.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy.orm import Session

import asyncio

from app.models.schema import InterventionDecision, VerificationReport
from app.services import intervention_policy
from app.services.intervention_features import build_cumulative_features, nearest_usable_checkpoint
from app.services.intervention_model import CHECKPOINT_QUOTAS, predict_impact
from app.services.verification_agent import VERIFICATION_PROMPT_VERSION
from app.services.verification_agent import run_verification as _default_run_verification
from app.services.verification_inflight_lock import VerificationInFlightLock

OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:8b-instruct")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
# Ollama's default is to unload a model from GPU memory after 5 minutes
# idle. Confirmed directly (SSH + nvidia-smi on a RunPod pod) that a
# cold-loaded 14B model takes 30+ seconds just to read back into VRAM
# before any real generation starts, dwarfing actual inference time (a
# warm call to the same model took 14.6s end to end, vs 42.2s cold) --
# and repeated cold starts between test runs likely confounded earlier
# batch-size comparisons. Keeping the model loaded for the whole session
# avoids re-paying that cost on every call.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
MAX_TOOL_ITERATIONS = 6
REQUEST_TIMEOUT = 120.0

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_rf_prediction",
            "description": (
                "Get the Balanced Cumulative multi-checkpoint Random Forest's predicted future "
                "propagation impact for this thread, at a given checkpoint (minutes since the "
                "source post). The model only sees reply timing/depth/text-shape features -- it "
                "has no idea whether the content is true. Higher predicted_impact means the model "
                "expects more future replies would be prevented by intervening now. You can call "
                "this at more than one checkpoint (e.g. 10 and the current one) to see whether the "
                "predicted urgency is rising or falling."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "checkpoint_minutes": {
                        "type": "integer",
                        "description": "One of 10, 20, 30, 40, 50, 60 -- must not exceed how old the thread actually is.",
                    }
                },
                "required": ["checkpoint_minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_verification_status",
            "description": (
                "Check whether the verification agent has already fact-checked this thread's claim, "
                "and if so what it found (credibility, confidence, summary). This is the ONLY source "
                "of content-truth information available to you -- the RF score knows nothing about "
                "it. Call this before deciding, since a thread already confirmed true or already "
                "debunked-and-flagged should weigh very differently than one nobody has looked at yet."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_budget_state",
            "description": (
                "Check how many 'escalate_now' slots are already used at a given checkpoint tier in "
                "the last 24 hours, out of that tier's fixed quota. Use this before choosing "
                "escalate_now if you want to know whether there is still room, versus whether the "
                "queue for that tier is already saturated (in which case escalate_next_checkpoint may "
                "be more realistic than escalate_now)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "checkpoint_minutes": {
                        "type": "integer",
                        "description": "One of 10, 20, 30, 40, 50, 60 -- the tier to check quota usage for.",
                    }
                },
                "required": ["checkpoint_minutes"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are an intervention-triage assistant. You are given one live social media thread that is currently being tracked. Your job is to decide how urgently it needs human attention, combining TWO independent signals: how fast/wide it is propagating (the RF score), and whether its content is even known to be true, false, or unverified (the verification status). Do not guess at truthfulness yourself -- get_verification_status is the only legitimate source for that.

Tools:
- get_rf_prediction: a trained model's predicted future propagation-impact score at a given checkpoint (10/20/30/40/50/60 minutes since the post). Pure propagation-structure signal, blind to content. A thread already slowing down (falling predicted_impact across checkpoints) is less urgent than a rising one even at the same absolute score.
- get_verification_status: whether the claim has already been fact-checked, and what was found. A thread already confirmed true, or already debunked-and-flagged, needs very different handling than one nobody has looked at.
- get_budget_state: how saturated a checkpoint tier's escalate_now quota already is in the last 24h. Escalating when the tier is already full is less realistic than when there is clear room.

Process:
1. You will already be given one get_rf_prediction result for the thread's current (most recent usable) checkpoint. Read it.
2. Call get_verification_status. This should almost always inform your decision -- e.g. a high RF score on an already-confirmed-true post is not an incident to escalate; a high RF score on an already-debunked post, or one still unverified, is exactly the case escalation exists for.
3. If it would help, call get_rf_prediction again at an earlier checkpoint to see the trend, or call get_budget_state before committing to escalate_now.
4. Once you have enough to decide, stop calling tools and output your verdict.

Your final answer MUST be ONLY a single JSON object (no prose before or after, no markdown fences) with exactly this shape:
{
  "action": "escalate_now" | "escalate_next_checkpoint" | "hold" | "dismiss",
  "confidence": <number between 0 and 1>,
  "reasoning": "<2-4 sentences citing the actual tool results you saw -- RF numbers, verification status, and budget state where relevant>",
  "checkpoints_reviewed": [<the checkpoint_minutes values you actually saw, as integers>]
}
"escalate_now" = act immediately; "escalate_next_checkpoint" = worth watching, revisit at the next checkpoint; "hold" = low urgency but keep tracking; "dismiss" = no meaningful propagation signal or already resolved (e.g. confirmed true). Base this only on tool results you actually saw -- do not invent numbers or a verification verdict that was never returned."""


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


async def _call_ollama(client: httpx.AsyncClient, messages: list[dict], use_tools: bool, tools: list[dict] | None = None) -> dict:
    payload = {
        "model": OLLAMA_MODEL, "stream": False, "messages": messages,
        "options": {"num_ctx": OLLAMA_NUM_CTX},
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    if use_tools:
        payload["tools"] = tools if tools is not None else TOOLS
    response = await client.post(f"{OLLAMA_API_BASE}/api/chat", json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _run_rf_tool(nodes: list[dict], thread_age_minutes: float, checkpoint_minutes: int) -> dict:
    checkpoint_seconds = checkpoint_minutes * 60
    if checkpoint_seconds > thread_age_minutes * 60:
        return {"error": f"This thread is only {thread_age_minutes:.1f} minutes old -- checkpoint {checkpoint_minutes} has not happened yet."}
    usable = nearest_usable_checkpoint(checkpoint_seconds)
    if usable is None:
        return {"error": "Thread is younger than the first checkpoint (10 minutes) -- not scoreable yet."}
    features = build_cumulative_features(nodes, usable)
    result = predict_impact(features)
    return {
        "checkpoint_minutes": int(usable // 60),
        "predicted_impact": round(result["predicted_impact"], 3),
        "predicted_log1p_impact": round(result["predicted_log1p_impact"], 3),
        "top_features": [
            {"feature": f["feature"], "importance": round(f["importance"], 4)}
            for f in result["top_features"][:5]
        ],
    }


def _verify_cache_key(thread_id: str) -> str:
    # Must match app/routers/radar.py's _verify_cache_key exactly -- both
    # derive the same VerificationReport row for a given radar thread.
    # Duplicated rather than imported to avoid a radar.py <-> this module
    # circular import (radar.py calls into this module to run the agent).
    # VERIFICATION_PROMPT_VERSION folded in for the same reason as
    # verification_agent.py's own docstring on that constant explains --
    # bumping it invalidates every VerificationReport produced under a
    # since-fixed reasoning prompt, on both sides of this duplication.
    return "radar_" + hashlib.sha256(f"{VERIFICATION_PROMPT_VERSION}|{thread_id}".encode("utf-8")).hexdigest()[:24]


def _run_verification_status_tool(db: Session, thread_id: str) -> dict:
    """Deliberately has no bare "verified: true/false" boolean -- earlier
    it did, and testing (replaying this agent against PHEME's own
    ground-truth veracity labels) caught the small local LLM reading
    "verified: false" as "this has been verified TO BE false" and treating
    that as a reason to withhold intervention, exactly backwards from the
    intent (a confirmed-false rumour, i.e. confirmed misinformation, is the
    clearest case FOR escalating, not against it). "credibility" alone,
    using the same likely_true/disputed/likely_false/unverified vocabulary
    verification_agent.py already produces, has no such ambiguity.
    """
    report = db.query(VerificationReport).filter_by(thread_id=_verify_cache_key(thread_id)).first()
    if not report:
        return {"credibility": "unverified", "confidence": 0.0, "summary": "No verification has been run yet for this thread."}
    data = report.report_jsonb or {}
    return {
        "credibility": data.get("credibility") or "unverified",
        "confidence": data.get("confidence"),
        "summary": data.get("summary"),
    }


# Below this confidence, "likely_true" is treated as genuinely unresolved
# rather than confirmed -- a 0.55-confidence "likely_true" is closer to a
# guess than a finding, and should still be eligible for intervention like
# any other unverified thread.
CONFIRMED_TRUE_CONFIDENCE = 0.7


def is_confirmed_true(db: Session, thread_id: str) -> bool:
    """Hard, server-enforced gate: a thread confirmed true above
    CONFIRMED_TRUE_CONFIDENCE is never a valid intervention target, full
    stop -- this project only intervenes on rumours (PHEME's sense: claims
    that are unverified or contested at the time, which after checking may
    turn out false, disputed, or still unverified). This is deliberately
    NOT left to the LLM to honor as an instruction (an LLM can be wrong or
    ignore a system prompt); it's checked in plain Python before a
    candidate is even shown to the agent, so a confirmed-true thread can
    physically never consume one of the scarce intervention slots.
    """
    status = _run_verification_status_tool(db, thread_id)
    return status.get("credibility") == "likely_true" and (status.get("confidence") or 0) >= CONFIRMED_TRUE_CONFIDENCE


def _run_budget_state_tool(db: Session, checkpoint_minutes: int) -> dict:
    quota = CHECKPOINT_QUOTAS.get(checkpoint_minutes)
    if quota is None:
        return {"error": f"checkpoint_minutes must be one of {sorted(CHECKPOINT_QUOTAS)}."}
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    used = (
        db.query(InterventionDecision)
        .filter(
            InterventionDecision.checkpoint_minutes == checkpoint_minutes,
            InterventionDecision.action == "escalate_now",
            InterventionDecision.decided_at >= since,
        )
        .count()
    )
    return {
        "checkpoint_minutes": checkpoint_minutes,
        "quota": quota,
        "already_escalated_last_24h": used,
        "remaining": max(quota - used, 0),
    }


# Public aliases for radar.py's batch scheduler, which needs to compute
# each candidate's initial RF/verification signal before building the batch
# prompt -- same underlying logic the tools call internally, just invoked
# directly instead of via a model tool-call.
score_thread_at_checkpoint = _run_rf_tool
get_verification_status_for = _run_verification_status_tool


async def run_intervention_review(db: Session, thread_id: str, nodes: list[dict], thread_age_minutes: float) -> dict:
    """Run the tool-calling triage loop for one live thread; returns the
    structured decision dict. `nodes` is bluesky_source.build_cascade_graph's
    node list -- the caller has already fetched the thread, this agent only
    reasons over it plus its three tools. Does not itself write to
    InterventionDecision -- the caller (radar.py) persists the result,
    matching how verify_radar_thread persists VerificationReport itself
    rather than verification_agent.py doing it.
    """
    initial_checkpoint = nearest_usable_checkpoint(thread_age_minutes * 60)
    if initial_checkpoint is None:
        return {
            "action": "hold",
            "confidence": 0.0,
            "reasoning": "Thread is younger than the first checkpoint (10 minutes); no RF score is available yet.",
            "checkpoints_reviewed": [],
            "model": OLLAMA_MODEL,
            "tool_calls": [],
        }

    tool_call_log: list[dict] = []
    initial_result = _run_rf_tool(nodes, thread_age_minutes, int(initial_checkpoint // 60))
    tool_call_log.append({"name": "get_rf_prediction", "arguments": {"checkpoint_minutes": int(initial_checkpoint // 60)}, "result": initial_result, "automatic": True})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"This thread is currently {thread_age_minutes:.1f} minutes old.\n"
                f"Automatic get_rf_prediction result at checkpoint {int(initial_checkpoint // 60)} minutes:\n"
                f"{json.dumps(initial_result, ensure_ascii=False)}\n\n"
                "Call get_verification_status next. Optionally call get_rf_prediction again at an "
                "earlier checkpoint to judge the trend, or get_budget_state before committing to "
                "escalate_now. Then give your final verdict."
            ),
        },
    ]

    async with httpx.AsyncClient() as client:
        final_content = None
        for _ in range(MAX_TOOL_ITERATIONS):
            result = await _call_ollama(client, messages, use_tools=True)
            message = result.get("message", {})
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                final_content = message.get("content", "")
                break
            messages.append({"role": "assistant", "content": message.get("content", ""), "tool_calls": tool_calls})

            for call in tool_calls:
                function = call.get("function", {})
                name = function.get("name", "")
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                if name == "get_rf_prediction":
                    checkpoint_minutes = arguments.get("checkpoint_minutes")
                    if not isinstance(checkpoint_minutes, int):
                        tool_result = {"error": "checkpoint_minutes must be an integer."}
                    else:
                        tool_result = _run_rf_tool(nodes, thread_age_minutes, checkpoint_minutes)
                elif name == "get_verification_status":
                    tool_result = _run_verification_status_tool(db, thread_id)
                elif name == "get_budget_state":
                    checkpoint_minutes = arguments.get("checkpoint_minutes")
                    if not isinstance(checkpoint_minutes, int):
                        tool_result = {"error": "checkpoint_minutes must be an integer."}
                    else:
                        tool_result = _run_budget_state_tool(db, checkpoint_minutes)
                else:
                    tool_result = {"error": f"Unknown tool '{name}'."}
                tool_call_log.append({"name": name, "arguments": arguments, "result": tool_result, "automatic": False})
                messages.append({"role": "tool", "content": json.dumps(tool_result, ensure_ascii=False)})
        else:
            messages.append({"role": "user", "content": "Stop calling tools now. Output only the final JSON verdict as specified, using the results already gathered."})
            result = await _call_ollama(client, messages, use_tools=False)
            final_content = result.get("message", {}).get("content", "")

    parsed = _extract_json(final_content or "") or {
        "action": "hold", "confidence": 0.0,
        "reasoning": "Agent did not return a parseable verdict.", "checkpoints_reviewed": [],
    }
    if parsed.get("action") not in {"escalate_now", "escalate_next_checkpoint", "hold", "dismiss"}:
        parsed["action"] = "hold"
    parsed["model"] = OLLAMA_MODEL
    parsed["tool_calls"] = tool_call_log
    return parsed


# --- Batch, budget-constrained selection (the scheduler's real mode) -------

BATCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_rf_prediction",
            "description": (
                "Re-check one candidate's RF-predicted future propagation reach at a different "
                "checkpoint than the one already given, to see whether it's trending up or down."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "post_uri": {"type": "string", "description": "One of the candidate post_uris given to you -- each is its own individual discussion thread."},
                    "checkpoint_minutes": {"type": "integer", "description": "One of 10, 20, 30, 40, 50, 60 -- must not exceed that thread's actual age."},
                },
                "required": ["post_uri", "checkpoint_minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_verification_status",
            "description": (
                "Double check a candidate's verification status if you need to re-confirm it. Verification "
                "is tracked per underlying story/event (cluster_thread_id), not per individual post, since "
                "candidates clustered under the same cluster_thread_id are near-duplicate posts about the "
                "same claim -- pass that candidate's cluster_thread_id, not its post_uri."
            ),
            "parameters": {
                "type": "object",
                "properties": {"cluster_thread_id": {"type": "string", "description": "The cluster_thread_id of the candidate you want to check."}},
                "required": ["cluster_thread_id"],
            },
        },
    },
]

SYSTEM_PROMPT_BATCH = """You are an intervention-budget allocator. You are given a batch of live threads that all just reached the same checkpoint (a fixed number of minutes since each was posted) and are eligible for intervention. Each candidate is ONE individual discussion thread: a single post plus its own reply cascade -- not a whole clustered "event". Several candidates can belong to the same underlying story (same cluster_thread_id) if multiple people posted about it independently; each is still its own thread with its own propagation.

Every candidate you see has ALREADY been screened -- none belong to a story confirmed true; that check happens before you're even called, so you will never be asked to intervene on something already known to be accurate. Everything you see is a rumour in the PHEME sense: unverified, or contested, at the time.

Each candidate's verification_status.credibility is one of:
- "likely_false": already confirmed misinformation. This is the CLEAREST case FOR escalating, not a reason to hold back -- do not confuse "not confirmed true" with "should be left alone". Spreading content already known to be false is exactly what intervention exists for.
- "disputed": contested by evidence found so far. Also a strong case for escalating.
- "unverified": no evidence checked yet either way. Judge mainly by predicted_reach -- this is genuinely uncertain, not a red flag by itself.
Do not read "not verified true" as "verified false" or as a reason to be more cautious -- an unresolved or confirmed-false claim is never a reason to escalate LESS than an otherwise-similar candidate.

You have a HARD LIMIT: only a fixed number of intervention slots ("escalate_now") are available right now, given to you as remaining_budget. You must not select more candidates for escalate_now than that number. If there are more good candidates than slots, prioritize by predicted_reach (higher = more future propagation this specific thread will produce if left alone) and by credibility per the ordering above (likely_false/disputed candidates outrank unverified ones at similar predicted_reach).

Each candidate already includes: post_uri (its identity), cluster_thread_id (which underlying story it belongs to), predicted_reach (from the RF, at the current checkpoint), and verification_status -- this information is already complete and current; you do NOT need to call a tool to confirm it. Only call get_rf_prediction or get_verification_status for a candidate when you have a SPECIFIC reason to doubt the given number (e.g. checking a trend across checkpoints), and even then for at most one or two candidates, never as a routine re-check of everyone in the batch -- with dozens of candidates per call, spending your limited turns re-confirming information you already have means you never reach a final answer at all, which is worse than deciding from what you were given. Prefer deciding directly from the candidate list every time.

Your final answer MUST be ONLY a single JSON object (no prose before or after, no markdown fences):
{
  "decisions": [
    {"post_uri": "<id>", "action": "escalate_now" | "escalate_next_checkpoint" | "hold", "confidence": <0-1>, "reasoning": "<1-3 sentences, cite the actual predicted_reach and verification status you saw>"},
    ...
  ]
}
You MUST include exactly one entry per candidate given to you, in any order. Do not invent a post_uri that wasn't given to you. Do not select more than remaining_budget candidates for escalate_now."""


def _run_rf_tool_for(nodes_by_post: dict[str, list[dict]], ages_by_post: dict[str, float], post_uri: str, checkpoint_minutes: int) -> dict:
    nodes = nodes_by_post.get(post_uri)
    age = ages_by_post.get(post_uri)
    if nodes is None or age is None:
        return {"error": f"Unknown post_uri '{post_uri}'."}
    return _run_rf_tool(nodes, age, checkpoint_minutes)


async def run_batch_selection(
    db: Session,
    candidates: list[dict],
    checkpoint_minutes: int,
    remaining_budget: int,
) -> dict:
    """Budget-constrained selection across every eligible candidate at one
    checkpoint tier. Each candidate is ONE individual discussion thread
    (one post + its own reply cascade), keyed by post_uri -- NOT one
    clustered event. Several candidates can share the same
    cluster_thread_id if independent posts about the same story are all
    due for review in the same tick; each still gets scored and decided on
    its own propagation, since a post's own reply cascade is what actually
    determines its reach, not its cluster's.

    `candidates` items: {post_uri, cluster_thread_id, nodes, age_minutes,
    initial_rf, verification} -- nodes/age_minutes/initial_rf already
    computed by the caller (radar.py) so this function doesn't re-fetch
    Bluesky itself; verification is already-confirmed-true filtered out by
    the caller via is_confirmed_true before this is called.

    Returns {"decisions": [...one per candidate, keyed by post_uri...],
    "model", "tool_calls"}. Server-side validated: if the model selects
    more than remaining_budget for escalate_now, only the
    highest-confidence ones are kept and the rest are downgraded to
    escalate_next_checkpoint with a note -- an LLM ignoring the stated
    limit must not be able to blow the budget.
    """
    if not candidates:
        return {"decisions": [], "model": OLLAMA_MODEL, "tool_calls": []}

    nodes_by_post = {c["post_uri"]: c["nodes"] for c in candidates}
    ages_by_post = {c["post_uri"]: c["age_minutes"] for c in candidates}
    tool_call_log: list[dict] = []

    candidate_summaries = [
        {
            "post_uri": c["post_uri"],
            "cluster_thread_id": c["cluster_thread_id"],
            "predicted_reach": c["initial_rf"].get("predicted_impact"),
            "verification_status": c["verification"],
        }
        for c in candidates
    ]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_BATCH},
        {
            "role": "user",
            "content": (
                f"Checkpoint: {checkpoint_minutes} minutes. remaining_budget: {remaining_budget}.\n"
                f"Candidates:\n{json.dumps(candidate_summaries, ensure_ascii=False, indent=2)}\n\n"
                "Decide now, or use the tools first if a specific candidate needs a closer look."
            ),
        },
    ]

    async with httpx.AsyncClient() as client:
        final_content = None
        for _ in range(MAX_TOOL_ITERATIONS):
            result = await _call_ollama(client, messages, use_tools=True, tools=BATCH_TOOLS)
            message = result.get("message", {})
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                final_content = message.get("content", "")
                break
            messages.append({"role": "assistant", "content": message.get("content", ""), "tool_calls": tool_calls})

            for call in tool_calls:
                function = call.get("function", {})
                name = function.get("name", "")
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                if name == "get_rf_prediction":
                    post_uri_arg = arguments.get("post_uri")
                    checkpoint_arg = arguments.get("checkpoint_minutes")
                    if not isinstance(post_uri_arg, str) or not isinstance(checkpoint_arg, int):
                        tool_result = {"error": "post_uri (string) and checkpoint_minutes (integer) are required."}
                    else:
                        tool_result = _run_rf_tool_for(nodes_by_post, ages_by_post, post_uri_arg, checkpoint_arg)
                elif name == "get_verification_status":
                    cluster_id_arg = arguments.get("cluster_thread_id")
                    if not isinstance(cluster_id_arg, str):
                        tool_result = {"error": "cluster_thread_id (string) is required."}
                    else:
                        tool_result = _run_verification_status_tool(db, cluster_id_arg)
                else:
                    tool_result = {"error": f"Unknown tool '{name}'."}
                tool_call_log.append({"name": name, "arguments": arguments, "result": tool_result})
                messages.append({"role": "tool", "content": json.dumps(tool_result, ensure_ascii=False)})
        else:
            messages.append({"role": "user", "content": "Stop calling tools now. Output only the final JSON decisions object, using the results already gathered, respecting remaining_budget."})
            result = await _call_ollama(client, messages, use_tools=False, tools=BATCH_TOOLS)
            final_content = result.get("message", {}).get("content", "")

    candidate_uris = {c["post_uri"] for c in candidates}
    parsed = _extract_json(final_content or "") or {}
    raw_decisions = parsed.get("decisions") if isinstance(parsed.get("decisions"), list) else []

    by_uri = {}
    for entry in raw_decisions:
        if not isinstance(entry, dict) or entry.get("post_uri") not in candidate_uris:
            continue
        by_uri[entry["post_uri"]] = entry

    decisions = []
    for c in candidates:
        entry = by_uri.get(c["post_uri"]) or {}
        action = entry.get("action")
        if action not in {"escalate_now", "escalate_next_checkpoint", "hold"}:
            action = "hold"
        confidence = entry.get("confidence")
        confidence = float(confidence) if isinstance(confidence, (int, float)) else 0.0
        reasoning = entry.get("reasoning") or "Agent did not return a decision for this candidate; defaulted to hold."
        decisions.append({
            "post_uri": c["post_uri"],
            "cluster_thread_id": c["cluster_thread_id"],
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
        })

    escalated = [d for d in decisions if d["action"] == "escalate_now"]
    if len(escalated) > remaining_budget:
        escalated.sort(key=lambda d: d["confidence"], reverse=True)
        keep_uris = {d["post_uri"] for d in escalated[:remaining_budget]}
        for d in decisions:
            if d["action"] == "escalate_now" and d["post_uri"] not in keep_uris:
                d["action"] = "escalate_next_checkpoint"
                d["reasoning"] += " (Downgraded server-side: the agent selected more candidates than remaining_budget allowed.)"

    return {"decisions": decisions, "model": OLLAMA_MODEL, "tool_calls": tool_call_log}


# --- Batch evidence gathering (feeds intervention_policy.select_for_checkpoint) ---
# run_batch_selection above still exists for the on-demand/legacy path, but
# the new scheduler pipeline is: this function gets evidence only, then
# intervention_policy.select_for_checkpoint (pure Python, no LLM) decides
# who gets a slot. See intervention_policy.py's module docstring for why.

BATCH_VERIFICATION_SIZE = 6
MAX_VERIFICATION_RETRIES = 1

_CACHE_FIELDS = ("last_status", "last_checked_at_minutes", "next_check_due_minutes", "verification_attempts", "status_version", "resolved", "resolved_at_minutes")


def _load_cache_record(report_jsonb: dict | None) -> intervention_policy.CacheRecord:
    if not report_jsonb:
        return intervention_policy.CacheRecord()
    kwargs = {field: report_jsonb[field] for field in _CACHE_FIELDS if field in report_jsonb}
    return intervention_policy.CacheRecord(**kwargs)


def _dump_cache_record(record: intervention_policy.CacheRecord) -> dict:
    return {field: getattr(record, field) for field in _CACHE_FIELDS}


def persist_verification_check(
    db: Session, key: str, event_id: str, checkpoint_minutes: int, is_high_risk: bool, raw_report: dict,
) -> dict:
    """Apply intervention_policy's cooldown bookkeeping (record_check) to a
    just-completed REAL verification result and persist it -- the one place
    that turns a raw run_verification() report into what actually gets
    written to VerificationReport.report_jsonb.

    Both callers that trigger a real check (run_batch_verification's own
    inline write below, and radar.py's manual "查證這則貼文" /verify
    endpoint) must go through this, not a bare `existing.report_jsonb =
    report` overwrite -- a bare overwrite drops last_status/
    next_check_due_minutes/resolved (the fields _load_cache_record reads
    back), so the NEXT caller's needs_check() sees a record with no
    bookkeeping at all and treats it as never-checked, forcing yet another
    real (non-deterministic) LLM call even though a perfectly good, recent
    verdict already exists. That was silently turning "查證這則貼文" and
    "真實政策決策" into a ping-pong of repeated real checks that could each
    land on a different verdict -- looked like the two panels disagreeing,
    but was really one endpoint's write wiping the other's cache metadata.
    """
    existing = db.query(VerificationReport).filter_by(thread_id=key).first()
    record = _load_cache_record(existing.report_jsonb if existing else None)
    raw_credibility = raw_report.get("credibility") or "unverified"
    record = intervention_policy.record_check(record, checkpoint_minutes, raw_credibility, raw_report.get("confidence"), is_high_risk)
    report_jsonb = {**raw_report, "credibility": record.last_status, **_dump_cache_record(record)}
    _upsert_verification_report(db, key, event_id, OLLAMA_MODEL, report_jsonb)
    return report_jsonb


def _verify_fn_accepts_progress_id(verify_fn) -> bool:
    # Test/smoke stubs (see test_intervention_policy_smoke.py) are plain
    # `async def stub(db, key, event_id, claim_text, post_date)` callables
    # with no progress_id param and no **kwargs -- calling them with an
    # unexpected keyword would break every one of those. Only the real
    # verification_agent.run_verification (or a stub that opts in) gets it.
    try:
        return "progress_id" in inspect.signature(verify_fn).parameters
    except (TypeError, ValueError):
        return False


async def _call_verify_fn(verify_fn, db, key: str, event_id: str, claim_text: str, post_date: str | None, progress_id: str | None):
    if progress_id and _verify_fn_accepts_progress_id(verify_fn):
        return await verify_fn(db, key, event_id, claim_text, post_date, progress_id=progress_id)
    return await verify_fn(db, key, event_id, claim_text, post_date)


async def _verify_with_own_session(
    verify_fn,
    use_db: bool,
    key: str,
    event_id: str,
    claim_text: str,
    post_date: str | None,
    checkpoint_minutes: int,
    is_high_risk: bool,
    progress_id: str | None = None,
):
    """Real verification (verification_agent.run_verification) reads/writes
    the DB itself (search result caching -- see search_web/search_news/
    search_fact_checks). A SQLAlchemy Session is not safe to share across
    concurrently-running coroutines/tasks; each parallel verification here
    gets its OWN short-lived session, committed and closed independently,
    never the parent run_batch_verification's shared session. In DB mode the
    same child session holds the cross-process advisory lock from the final
    cache re-check through verification and the committed cache upsert. This
    closes the otherwise unavoidable call-finished/write-not-yet-visible
    race window.

    `use_db=False` (run_batch_verification's `db` argument was None --
    offline replay / smoke-test mode) skips session creation entirely and
    calls verify_fn(None, ...) -- must never open a real DB connection just
    because a caller passed a stub verify_fn and no db, or every "no DB
    needed" smoke test would silently start requiring Postgres.
    """
    if not use_db:
        return await _call_verify_fn(verify_fn, None, key, event_id, claim_text, post_date, progress_id)
    from app.database import engine
    # Pin the ORM Session to one explicitly-held Connection. The real
    # verifier commits search-cache rows during a long call; an engine-bound
    # Session may return its connection to the pool after such a commit even
    # though the Python Session object remains alive. PostgreSQL advisory
    # locks belong to the connection, so that would release/strand the lock
    # on a pooled connection and make the later unlock run elsewhere.
    connection = engine.connect()
    session = Session(bind=connection, autoflush=False)
    inflight_lock = VerificationInFlightLock(session, key)
    try:
        if not inflight_lock.try_acquire():
            return {"_inflight_deferred": True}

        # The parent checked the cache before this task acquired the lock.
        # Re-read it now: another scheduler may have completed and committed
        # while this task was waiting to run.
        existing = session.query(VerificationReport).filter_by(thread_id=key).first()
        record = _load_cache_record(existing.report_jsonb if existing else None)
        if not intervention_policy.needs_check(record, checkpoint_minutes):
            return {"_cache_filled_by_peer": True, "report_jsonb": existing.report_jsonb}

        result = await _call_verify_fn(verify_fn, session, key, event_id, claim_text, post_date, progress_id)
        raw_credibility = result.get("credibility") or "unverified"
        confidence = result.get("confidence")
        record = intervention_policy.record_check(
            record, checkpoint_minutes, raw_credibility, confidence, is_high_risk,
        )
        effective = record.last_status
        report_jsonb = {**result, "credibility": effective, **_dump_cache_record(record)}
        _upsert_verification_report(session, key, event_id, OLLAMA_MODEL, report_jsonb)
        return {
            "_persisted_under_lock": True,
            "report_jsonb": report_jsonb,
            "credibility": effective,
            "confidence": confidence,
            "summary": result.get("summary"),
            # Same trace fields as the other outcome shapes below -- without
            # these, run_batch_verification's own `outcome.get("tool_calls",
            # [])` fallback silently returns [] for every call that happens
            # to go through this locked path, even though `result` (and thus
            # report_jsonb) has them right there.
            "tool_calls": result.get("tool_calls", []),
            "queries_used": result.get("queries_used", []),
            "evidence": result.get("evidence", []),
            "model": result.get("model"),
        }
    except Exception:
        session.rollback()
        raise
    finally:
        try:
            inflight_lock.release()
        finally:
            try:
                session.close()
            finally:
                connection.close()


def _upsert_verification_report(db: Session, key: str, event_id: str, model_name: str, report_jsonb: dict) -> None:
    """Single atomic upsert (INSERT ... ON CONFLICT (thread_id) DO UPDATE)
    instead of a read-then-write query+add/update -- VerificationReport.
    thread_id is unique, so this is safe even if another process/session
    raced to insert the same key between this call's read and write (the
    read-then-write pattern this replaces was not actually atomic)."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    stmt = pg_insert(VerificationReport).values(
        thread_id=key, event_id=event_id, model_name=model_name, report_jsonb=report_jsonb,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["thread_id"],
        set_={"report_jsonb": stmt.excluded.report_jsonb, "model_name": stmt.excluded.model_name},
    )
    db.execute(stmt)
    db.commit()


async def run_batch_verification(
    db: Session,
    candidates: list[dict],
    checkpoint_minutes: int,
    verify_fn=None,
    progress_id: str | None = None,
) -> dict:
    """Ensures each DISTINCT cluster_thread_id among `candidates` has
    fresh-enough evidence for scheduling, per the intervention_policy
    verification cache -- does NOT decide who gets an intervention slot,
    only "what does the evidence say right now". See intervention_policy.
    select_for_checkpoint for the deterministic selection step this feeds.

    `candidates`: [{post_uri, cluster_thread_id, claim_text, event_id,
    post_date, is_high_risk}]. Verification is per underlying story
    (cluster_thread_id), same convention as get_verification_status -- if
    multiple candidates in this call share a cluster_thread_id (independent
    posts about the same story due at the same checkpoint), that story is
    verified exactly ONCE, not once per post; the result is looked up by
    cluster_thread_id, not duplicated into a redundant call.

    `verify_fn`: injectable async callable(db, cache_key, event_id,
    claim_text, post_date) -> report dict, defaulting to
    verification_agent.run_verification. Exists purely so smoke tests can
    inject a fast deterministic stub instead of a real search+LLM call --
    each real call does live web searches and can take well over a minute.
    Each parallel call within a batch gets its own short-lived DB session
    (see _verify_with_own_session) -- this function's own `db` argument is
    never passed into a concurrently-running verify_fn call.

    `progress_id`: forwarded to verify_fn as a `progress_id=` kwarg (only
    when verify_fn's own signature accepts one -- see
    _verify_fn_accepts_progress_id, so injected test stubs are unaffected)
    ONLY when this call verifies exactly one distinct story, so a caller
    polling verification_progress.get(progress_id) sees that one story's
    tool calls live instead of several unrelated ones interleaved.

    Runs BATCH_VERIFICATION_SIZE unique clusters at a time, concurrently
    within a batch. A cluster whose call raises is retried, alone, up to
    MAX_VERIFICATION_RETRIES times. Still failing afterward is marked
    credibility="agent_failure" and is NEVER written to the cache as if it
    had been checked -- next_check_due is left untouched so a future call
    retries it for real, instead of silently treating a failure as a
    checked-and-unverified thread (which would wrongly suppress escalation
    for something that was never actually looked at). agent_failure must
    never be scored by intervention_policy.priority_for -- see
    select_for_checkpoint's hard exclusion.

    A cluster already being verified by another scheduler is returned as
    credibility="deferred_inflight". It is neither retried immediately nor
    mislabeled agent_failure, and no cache row is written by the losing task.

    Returns {"results": {cluster_thread_id: {credibility, confidence,
    summary, from_cache, tool_calls, queries_used, evidence, model}}} --
    tool_calls/queries_used/evidence/model are the same agent-trace fields
    run_verification produces (empty/None when there was nothing to show,
    e.g. deferred_inflight), carried through untrimmed so a caller can
    display what the agent actually did, not just its verdict.
    "agent_failures": [cluster_thread_id, ...],
    "inflight_deferred": [cluster_thread_id, ...],
    "checked_now": [cluster_thread_id, ...]}. credibility in results is
    already the EFFECTIVE (confidence-gated) status -- see
    intervention_policy.effective_credibility.
    """
    verify_fn = verify_fn or _default_run_verification
    results: dict[str, dict] = {}
    checked_now: list[str] = []
    inflight_deferred: list[str] = []
    record_by_key: dict[str, intervention_policy.CacheRecord] = {}

    # Dedupe by cluster_thread_id -- one representative candidate (first
    # occurrence) stands in for the whole cluster's claim_text/event_id/
    # post_date/is_high_risk when multiple posts share a story.
    unique_by_key: dict[str, dict] = {}
    for c in candidates:
        unique_by_key.setdefault(c["cluster_thread_id"], c)

    to_check = []
    for key, c in unique_by_key.items():
        existing = db.query(VerificationReport).filter_by(thread_id=key).first() if db is not None else None
        record = _load_cache_record(existing.report_jsonb if existing else None)
        record_by_key[key] = record
        if intervention_policy.needs_check(record, checkpoint_minutes):
            to_check.append(c)
        else:
            cached = existing.report_jsonb or {} if existing else {}
            results[key] = {
                "credibility": record.last_status or "unverified",
                "confidence": cached.get("confidence"),
                "summary": cached.get("summary"),
                "from_cache": True,
                # Carried through so a caller (e.g. the manual policy MVP endpoint)
                # can show what the agent actually did even on a cache hit --
                # previously trimmed away here even though report_jsonb already
                # had it, so the frontend had no way to display it at all.
                "tool_calls": cached.get("tool_calls", []),
                "queries_used": cached.get("queries_used", []),
                "evidence": cached.get("evidence", []),
                "model": cached.get("model"),
            }

    # Only forward progress_id when this batch verifies exactly one distinct
    # story -- with more than one, concurrent verify_fn calls would all
    # write into the SAME progress feed and interleave unrelated claims'
    # tool calls into one confusing stream. The manual single-post MVP
    # endpoint (run_batch_verification's only real caller that has a
    # meaningful progress_id to give) always calls with exactly one
    # candidate, so this never actually costs it anything.
    single_candidate_progress_id = progress_id if len(to_check) == 1 else None

    pending = list(to_check)
    for _attempt in range(MAX_VERIFICATION_RETRIES + 1):
        if not pending:
            break
        next_pending = []
        for i in range(0, len(pending), BATCH_VERIFICATION_SIZE):
            chunk = pending[i:i + BATCH_VERIFICATION_SIZE]
            outcomes = await asyncio.gather(
                *(
                    _verify_with_own_session(
                        verify_fn, db is not None, c["cluster_thread_id"],
                        c.get("event_id", "radar"), c["claim_text"],
                        c.get("post_date"), checkpoint_minutes,
                        c.get("is_high_risk", False),
                        progress_id=single_candidate_progress_id,
                    )
                    for c in chunk
                ),
                return_exceptions=True,
            )
            for c, outcome in zip(chunk, outcomes):
                key = c["cluster_thread_id"]
                if isinstance(outcome, Exception):
                    next_pending.append(c)
                    continue
                if outcome.get("_inflight_deferred"):
                    inflight_deferred.append(key)
                    results[key] = {
                        "credibility": "deferred_inflight", "confidence": None,
                        "summary": "Verification is already running in another scheduler; retry at a later checkpoint.",
                        "from_cache": False,
                    }
                    continue
                if outcome.get("_cache_filled_by_peer"):
                    cached = outcome["report_jsonb"] or {}
                    results[key] = {
                        "credibility": cached.get("credibility", "unverified"),
                        "confidence": cached.get("confidence"),
                        "summary": cached.get("summary"),
                        "from_cache": True,
                        "tool_calls": cached.get("tool_calls", []),
                        "queries_used": cached.get("queries_used", []),
                        "evidence": cached.get("evidence", []),
                        "model": cached.get("model"),
                    }
                    continue
                checked_now.append(key)
                if outcome.get("_persisted_under_lock"):
                    results[key] = {
                        "credibility": outcome["credibility"],
                        "confidence": outcome.get("confidence"),
                        "summary": outcome.get("summary"),
                        "from_cache": False,
                        "tool_calls": outcome.get("tool_calls", []),
                        "queries_used": outcome.get("queries_used", []),
                        "evidence": outcome.get("evidence", []),
                        "model": outcome.get("model"),
                    }
                    continue
                raw_credibility = outcome.get("credibility") or "unverified"
                confidence = outcome.get("confidence")
                record = intervention_policy.record_check(
                    record_by_key[key], checkpoint_minutes, raw_credibility, confidence, c.get("is_high_risk", False),
                )
                record_by_key[key] = record
                effective_credibility = record.last_status  # record_check already applied effective_credibility
                if db is not None:
                    report_jsonb = {**outcome, "credibility": effective_credibility, **_dump_cache_record(record)}
                    _upsert_verification_report(db, key, c.get("event_id", "radar"), OLLAMA_MODEL, report_jsonb)
                results[key] = {
                    "credibility": effective_credibility, "confidence": confidence, "summary": outcome.get("summary"),
                    "from_cache": False,
                    # Same rationale as the cache-hit branches above: this is
                    # the one path that just ran a real agent call, so it's
                    # the richest trace available -- must not be the one
                    # branch that drops it.
                    "tool_calls": outcome.get("tool_calls", []),
                    "queries_used": outcome.get("queries_used", []),
                    "evidence": outcome.get("evidence", []),
                    "model": outcome.get("model"),
                }
        pending = next_pending

    agent_failures = [c["cluster_thread_id"] for c in pending]
    for c in pending:
        key = c["cluster_thread_id"]
        results[key] = {
            "credibility": "agent_failure", "confidence": None,
            "summary": "Verification agent failed after retry; evidence unknown, not cached as checked.",
            "from_cache": False,
        }

    return {
        "results": results,
        "agent_failures": agent_failures,
        "inflight_deferred": inflight_deferred,
        "checked_now": checked_now,
    }
