"""Time-gated replay primitives for the PHEME "即時情境模擬" live scenario
simulator -- the "指揮台" MVP: multiple threads of one PHEME event replayed
on ONE shared simulated clock, in their real arrival order, with future
nodes/edges withheld until the simulated clock actually reaches them.

Pure functions only -- no FastAPI/HTTP here (see routers/intervention.py
for the thin endpoints wrapping these) and no model training/mutation of
data/raw/pheme. Everything operates on a single scalar "simulated event
time" T, in whole seconds since the EVENT's own earliest source tweet --
never wall-clock time, never a UI-animation-speed-dependent clock -- so a
replay is fully determined by (event_id, thread arrival offsets, T) alone
and is independent of playback speed, pause/resume, or how often the
frontend polls.

Research-safety invariant this module exists to enforce (the user's own
spec, and AGENTS.md's leakage rules by extension): at simulated time T,
nothing whose real event-time is strictly after T may be returned to a
caller. This is enforced HERE, in the data layer, not left to the frontend
to merely hide after receiving it -- there is no parameter combination that
makes cascade_window or realized_blocked_nodes hand back a future node.
"""
from __future__ import annotations

from pathlib import Path

from app.services import intervention_policy
from app.services.pheme_cascade import load_source_created_at


def build_event_manifest(base_path: Path, event_id: str, thread_scores: list[dict]) -> dict:
    """Places every scored thread's source tweet on ONE shared event clock.

    `thread_scores`: rows from model_scores.ranked_threads(event_id, ...)
    (each already has thread_id/rank/percentile_score/prediction/
    preventable_impact) -- reused as-is, nothing here re-derives the score.

    Returns {"event_id", "threads": [...sorted by arrival_offset_seconds
    ascending -- i.e. REAL PHEME arrival order, not RF-rank order...],
    "excluded_missing_timestamp": [thread_id, ...]}. A thread whose source
    tweet's created_at can't be parsed is excluded from the manifest (with
    its id reported, not silently dropped) rather than crashing the whole
    event or guessing a fake arrival time for it.

    "threads" rows: {thread_id, rank, percentile_score, prediction,
    preventable_impact, arrival_offset_seconds}. arrival_offset_seconds is
    seconds since the EARLIEST source tweet in this event -- always >= 0,
    with exactly one thread at 0.
    """
    created_at_by_thread: dict[str, object] = {}
    excluded: list[str] = []
    for row in thread_scores:
        created_at = load_source_created_at(base_path, event_id, row["thread_id"])
        if created_at is None:
            excluded.append(row["thread_id"])
            continue
        created_at_by_thread[row["thread_id"]] = created_at

    if not created_at_by_thread:
        return {"event_id": event_id, "threads": [], "excluded_missing_timestamp": excluded}

    event_start = min(created_at_by_thread.values())
    threads = []
    for row in thread_scores:
        created_at = created_at_by_thread.get(row["thread_id"])
        if created_at is None:
            continue
        threads.append({
            **row,
            "arrival_offset_seconds": (created_at - event_start).total_seconds(),
        })
    threads.sort(key=lambda item: (item["arrival_offset_seconds"], item["thread_id"]))
    return {"event_id": event_id, "threads": threads, "excluded_missing_timestamp": excluded}


def auto_decision(predicted_impact: float, credibility: str | None, confidence: float | None) -> dict:
    """"Agent 自動決策" mode's rule, reusing the SAME deterministic
    thresholds the live radar pipeline already uses (intervention_policy.py)
    instead of re-deriving a second, driftable copy of "how confident is
    confident enough" for the PHEME simulator.

    Deliberately conservative, matching this project's standing principle
    (see verification_agent.py, intervention_policy.py) that an agent must
    only act decisively on strong evidence and otherwise defer to a human:
      - confirmed true (effective_credibility == likely_true) -> "observed"
        (excluded from intervention -- confirmed-accurate content).
      - confirmed false (effective_credibility == likely_false, i.e.
        recommendation_tier == "hard") -> "blocked".
      - everything else (unverified, disputed, low-confidence either way,
        agent_failure) -> "undetermined": the agent does NOT guess: this
        thread stays pending for a human (or a later re-check once more
        evidence exists), it is never silently left un-acted-on and never
        silently blocked on weak evidence.
    """
    if intervention_policy.is_confirmed_true(credibility, confidence):
        return {"action": "observed", "reasoning": "查證顯示證據支持此為真實內容，排除介入。"}
    priority = intervention_policy.priority_for(predicted_impact, credibility, confidence)
    if priority is not None and intervention_policy.recommendation_tier(credibility, confidence) == "hard":
        return {"action": "blocked", "reasoning": f"查證顯示證據反駁此內容（信心足夠），且預估影響 {predicted_impact:.2f}，建議介入。"}
    return {"action": "undetermined", "reasoning": "證據尚不足以支持自動決策（未查證、爭議中，或信心不足），保留給人工判斷。"}


def cascade_window(cascade: dict, thread_arrival_offset_seconds: float, sim_time_seconds: float) -> dict:
    """The subset of one thread's cascade visible at simulated event time T.

    A node's own absolute event-time is thread_arrival_offset_seconds plus
    its offset_sec (0 for the source node itself). A node is visible iff
    its event-time <= sim_time_seconds. If the thread itself hasn't arrived
    yet (thread_arrival_offset_seconds > sim_time_seconds), nothing is
    visible -- the thread does not exist in the simulation yet, matching
    "threads 依 source 發布時間逐步進場" instead of the old fixed-delay
    fake-arrival-order behavior.

    An edge is visible only when BOTH endpoints are visible (a reply can
    never be revealed pointing at a parent that hasn't arrived).
    """
    if sim_time_seconds < thread_arrival_offset_seconds:
        return {
            **{k: cascade[k] for k in ("thread_id", "event_id", "cutoff_seconds")},
            "nodes": [], "edges": [],
            "sim_time_seconds": sim_time_seconds,
            "thread_arrival_offset_seconds": thread_arrival_offset_seconds,
            "thread_has_arrived": False,
        }

    visible_ids: set[str] = set()
    visible_nodes = []
    for node in cascade["nodes"]:
        if node["is_source"]:
            event_time = thread_arrival_offset_seconds
        elif node["offset_sec"] is not None:
            event_time = thread_arrival_offset_seconds + node["offset_sec"]
        else:
            # No parseable timestamp for this reply at all -- never
            # visible, since we have no basis to say it "arrived" by T.
            continue
        if event_time > sim_time_seconds:
            continue
        visible_ids.add(node["id"])
        visible_nodes.append({**node, "event_time_seconds": event_time})

    visible_edges = [edge for edge in cascade["edges"] if edge["source"] in visible_ids and edge["target"] in visible_ids]

    return {
        **{k: cascade[k] for k in ("thread_id", "event_id", "cutoff_seconds")},
        "nodes": visible_nodes, "edges": visible_edges,
        "sim_time_seconds": sim_time_seconds,
        "thread_arrival_offset_seconds": thread_arrival_offset_seconds,
        "thread_has_arrived": True,
    }


def realized_blocked_nodes(cascade: dict, thread_arrival_offset_seconds: float, sim_time_seconds: float) -> dict:
    """Which of this thread's real historical post-cutoff replies have, as
    of simulated time T, reached the point in (simulated) time where they
    would have appeared -- i.e. which "future" nodes an earlier Hard
    intervention on this thread can now honestly be counted as having
    prevented, versus which are still merely scheduled to (counterfactually)
    happen later in the replay.

    Mirrors intervention.py's existing /intervene semantics exactly
    (blocked = a real historical reply with observed_by_cutoff == False,
    i.e. it occurred after the fixed 30-minute decision cutoff) -- the only
    change from that endpoint is WHEN each one becomes visible: previously
    every blocked node was returned in one response the instant the button
    was pressed; here a node only moves from "pending" to "realized" once
    sim_time_seconds actually reaches its own real event-time. Does NOT
    depend on when the user clicked intervene -- same fixed-cutoff policy
    semantics as the existing endpoint, just progressively revealed.

    Purely a function of (cascade, thread_arrival_offset_seconds,
    sim_time_seconds) -- no wall-clock time, no UI state -- so calling this
    repeatedly as sim_time_seconds monotonically increases always yields a
    monotonically growing realized set, regardless of playback speed or
    polling cadence.
    """
    realized, pending = [], []
    for node in cascade["nodes"]:
        if node["is_source"] or node["observed_by_cutoff"] or node["offset_sec"] is None:
            continue
        event_time = thread_arrival_offset_seconds + node["offset_sec"]
        target = realized if event_time <= sim_time_seconds else pending
        target.append({**node, "event_time_seconds": event_time})

    return {
        "thread_id": cascade["thread_id"],
        "event_id": cascade["event_id"],
        "sim_time_seconds": sim_time_seconds,
        "realized_blocked_node_ids": [n["id"] for n in realized],
        "realized_blocked_count": len(realized),
        "pending_blocked_count": len(pending),
        "total_blocked_node_count": len(realized) + len(pending),
        "note": (
            "回溯重播示範：realized_blocked_count 只計入模擬時間已經走到、"
            "且真實發生於 30 分鐘截止點之後的節點；尚未到達模擬時間的節點"
            "只計入 pending_blocked_count，不會提前揭露。"
        ),
    }
