"""Unit tests for pheme_replay.py -- the 即時情境模擬指揮台 MVP's time-gating
core. All pure functions, no DB/network/Bluesky/Ollama, no PHEME dataset
files touched (build_event_manifest's dataset-reading half is exercised via
a monkeypatched load_source_created_at so these never open data/raw/pheme).

Covers the user's own numbered test requirements that apply to pure,
deterministic replay logic (their #1, #2, #3, #7, #8, #9, #12 -- the
remaining ones are frontend-session / API-integration concerns out of this
file's scope, see the final report's Remaining Issues).
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.services import pheme_replay


def _dt(offset_minutes: float) -> datetime:
    return datetime(2015, 1, 7, 12, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=offset_minutes)


class EventManifestArrivalOrderTest(unittest.TestCase):
    """Requirement #12 (in spirit) + the core "real arrival order, not
    RF-rank order" claim: build_event_manifest must sort by real timestamp,
    and never invent a timestamp for a thread it can't load one for."""

    def test_sorted_by_real_arrival_not_by_rf_rank(self) -> None:
        # Deliberately: the HIGHEST-ranked (rank=1) thread arrives LAST in
        # real time -- if the old RF-rank ordering leaked through, this
        # thread would wrongly appear first.
        scores = [
            {"thread_id": "t_rank1_late", "rank": 1, "percentile_score": 100, "prediction": 5.0, "preventable_impact": 10},
            {"thread_id": "t_rank2_early", "rank": 2, "percentile_score": 90, "prediction": 4.0, "preventable_impact": 8},
        ]
        created_at_map = {"t_rank1_late": _dt(30), "t_rank2_early": _dt(0)}
        with patch.object(pheme_replay, "load_source_created_at", side_effect=lambda base, eid, tid: created_at_map[tid]):
            manifest = pheme_replay.build_event_manifest(Path("."), "testevent", scores)
        ids_in_order = [t["thread_id"] for t in manifest["threads"]]
        self.assertEqual(["t_rank2_early", "t_rank1_late"], ids_in_order)

    def test_earliest_thread_has_zero_offset(self) -> None:
        scores = [{"thread_id": "a", "rank": 1, "percentile_score": 100, "prediction": 1.0, "preventable_impact": 1}]
        with patch.object(pheme_replay, "load_source_created_at", return_value=_dt(0)):
            manifest = pheme_replay.build_event_manifest(Path("."), "testevent", scores)
        self.assertEqual(0.0, manifest["threads"][0]["arrival_offset_seconds"])

    def test_unparseable_timestamp_excludes_without_crashing(self) -> None:
        scores = [
            {"thread_id": "good", "rank": 1, "percentile_score": 100, "prediction": 1.0, "preventable_impact": 1},
            {"thread_id": "bad", "rank": 2, "percentile_score": 90, "prediction": 0.5, "preventable_impact": 0.5},
        ]
        with patch.object(pheme_replay, "load_source_created_at", side_effect=lambda base, eid, tid: (_dt(0) if tid == "good" else None)):
            manifest = pheme_replay.build_event_manifest(Path("."), "testevent", scores)
        self.assertEqual(["good"], [t["thread_id"] for t in manifest["threads"]])
        self.assertEqual(["bad"], manifest["excluded_missing_timestamp"])


def _make_cascade(nodes):
    return {"thread_id": "t1", "event_id": "e1", "cutoff_seconds": 1800, "nodes": nodes, "edges": [
        {"source": nodes[i - 1]["id"], "target": nodes[i]["id"]} for i in range(1, len(nodes))
    ]}


class CascadeWindowFutureVisibilityTest(unittest.TestCase):
    """Requirement #1/#2: a future node/edge must never appear before
    simulated time reaches it, regardless of how far in the future it is."""

    def setUp(self) -> None:
        self.cascade = _make_cascade([
            {"id": "src", "parent_id": None, "is_source": True, "offset_sec": None, "observed_by_cutoff": True, "screen_name": "a", "text": "x"},
            {"id": "r1", "parent_id": "src", "is_source": False, "offset_sec": 60.0, "observed_by_cutoff": True, "screen_name": "b", "text": "y"},
            {"id": "r2_future", "parent_id": "src", "is_source": False, "offset_sec": 5000.0, "observed_by_cutoff": False, "screen_name": "c", "text": "z"},
        ])

    def test_future_node_absent_before_its_time(self) -> None:
        window = pheme_replay.cascade_window(self.cascade, thread_arrival_offset_seconds=0.0, sim_time_seconds=100.0)
        ids = {n["id"] for n in window["nodes"]}
        self.assertIn("src", ids)
        self.assertIn("r1", ids)
        self.assertNotIn("r2_future", ids)

    def test_future_node_present_once_its_time_is_reached(self) -> None:
        window = pheme_replay.cascade_window(self.cascade, thread_arrival_offset_seconds=0.0, sim_time_seconds=5000.0)
        ids = {n["id"] for n in window["nodes"]}
        self.assertIn("r2_future", ids)

    def test_future_edge_absent_before_its_target_arrives(self) -> None:
        window = pheme_replay.cascade_window(self.cascade, thread_arrival_offset_seconds=0.0, sim_time_seconds=100.0)
        targets = {e["target"] for e in window["edges"]}
        self.assertNotIn("r2_future", targets)

    def test_thread_not_yet_arrived_shows_nothing(self) -> None:
        window = pheme_replay.cascade_window(self.cascade, thread_arrival_offset_seconds=9999.0, sim_time_seconds=100.0)
        self.assertEqual([], window["nodes"])
        self.assertEqual([], window["edges"])
        self.assertFalse(window["thread_has_arrived"])

    def test_thread_arrival_offset_shifts_absolute_event_time(self) -> None:
        # Same cascade, but this thread's source itself arrives at T=1000 on
        # the shared event clock -- r1 (offset_sec=60 relative to ITS OWN
        # source) must become visible at sim_time=1060, not sim_time=60.
        window_before = pheme_replay.cascade_window(self.cascade, thread_arrival_offset_seconds=1000.0, sim_time_seconds=1059.0)
        window_after = pheme_replay.cascade_window(self.cascade, thread_arrival_offset_seconds=1000.0, sim_time_seconds=1060.0)
        self.assertNotIn("r1", {n["id"] for n in window_before["nodes"]})
        self.assertIn("r1", {n["id"] for n in window_after["nodes"]})


class RealizedBlockedNodesProgressiveRevealTest(unittest.TestCase):
    """Requirement #9: blocked nodes unlock progressively as simulated time
    passes, never all at once regardless of when the user intervened."""

    def setUp(self) -> None:
        self.cascade = _make_cascade([
            {"id": "src", "parent_id": None, "is_source": True, "offset_sec": None, "observed_by_cutoff": True, "screen_name": "a", "text": "x"},
            {"id": "early", "parent_id": "src", "is_source": False, "offset_sec": 900.0, "observed_by_cutoff": True, "screen_name": "b", "text": "y"},
            {"id": "late1", "parent_id": "src", "is_source": False, "offset_sec": 2000.0, "observed_by_cutoff": False, "screen_name": "c", "text": "z"},
            {"id": "late2", "parent_id": "src", "is_source": False, "offset_sec": 4000.0, "observed_by_cutoff": False, "screen_name": "d", "text": "w"},
        ])

    def test_none_realized_before_any_future_node_time(self) -> None:
        result = pheme_replay.realized_blocked_nodes(self.cascade, thread_arrival_offset_seconds=0.0, sim_time_seconds=1000.0)
        self.assertEqual(0, result["realized_blocked_count"])
        self.assertEqual(2, result["pending_blocked_count"])

    def test_partial_reveal_between_the_two_future_nodes(self) -> None:
        result = pheme_replay.realized_blocked_nodes(self.cascade, thread_arrival_offset_seconds=0.0, sim_time_seconds=2500.0)
        self.assertEqual(["late1"], result["realized_blocked_node_ids"])
        self.assertEqual(1, result["pending_blocked_count"])

    def test_all_realized_once_sim_time_passes_every_future_node(self) -> None:
        result = pheme_replay.realized_blocked_nodes(self.cascade, thread_arrival_offset_seconds=0.0, sim_time_seconds=10000.0)
        self.assertEqual({"late1", "late2"}, set(result["realized_blocked_node_ids"]))
        self.assertEqual(0, result["pending_blocked_count"])

    def test_already_observed_nodes_never_counted_as_blocked(self) -> None:
        # "early" has observed_by_cutoff=True -- must never appear in either
        # realized or pending, at any sim_time.
        result = pheme_replay.realized_blocked_nodes(self.cascade, thread_arrival_offset_seconds=0.0, sim_time_seconds=999999.0)
        self.assertNotIn("early", result["realized_blocked_node_ids"])
        self.assertEqual(2, result["total_blocked_node_count"])

    def test_reveal_is_monotonically_non_decreasing_as_sim_time_advances(self) -> None:
        # Requirement #6 (speed-independence, expressed as: calling this
        # repeatedly at increasing sim_time never "loses" an already-
        # realized node) -- polling cadence/animation speed must not matter.
        counts = [
            pheme_replay.realized_blocked_nodes(self.cascade, 0.0, t)["realized_blocked_count"]
            for t in (0, 500, 1500, 2000, 2500, 4000, 4001, 999999)
        ]
        self.assertEqual(counts, sorted(counts))

    def test_result_independent_of_call_order_or_repetition(self) -> None:
        # Requirement #5 (pause/resume): calling out of order or repeatedly
        # at the same T always gives the identical answer -- no hidden
        # mutable state anywhere in this module.
        a = pheme_replay.realized_blocked_nodes(self.cascade, 0.0, 2500.0)
        b = pheme_replay.realized_blocked_nodes(self.cascade, 0.0, 1000.0)
        c = pheme_replay.realized_blocked_nodes(self.cascade, 0.0, 2500.0)
        self.assertEqual(a, c)
        self.assertNotEqual(a, b)


class AutoDecisionReusesRadarPolicyTest(unittest.TestCase):
    """"Agent 自動決策" must reuse intervention_policy's existing thresholds
    verbatim, not a second, driftable copy of the confidence cutoffs."""

    def test_confirmed_true_is_observed(self) -> None:
        result = pheme_replay.auto_decision(predicted_impact=5.0, credibility="likely_true", confidence=0.95)
        self.assertEqual("observed", result["action"])

    def test_confirmed_false_is_blocked(self) -> None:
        result = pheme_replay.auto_decision(predicted_impact=3.0, credibility="likely_false", confidence=0.9)
        self.assertEqual("blocked", result["action"])

    def test_low_confidence_false_is_undetermined_not_blocked(self) -> None:
        # Below intervention_policy.CONFIRMED_FALSE_CONFIDENCE -- must NOT
        # be auto-blocked on weak evidence.
        result = pheme_replay.auto_decision(predicted_impact=3.0, credibility="likely_false", confidence=0.4)
        self.assertEqual("undetermined", result["action"])

    def test_disputed_is_undetermined_never_auto_blocked(self) -> None:
        # disputed can reach "soft" in the radar policy but never "hard" --
        # the PHEME simulator's binary blocked/observed has no "soft", so
        # disputed must defer to a human, never silently auto-block.
        result = pheme_replay.auto_decision(predicted_impact=10.0, credibility="disputed", confidence=0.95)
        self.assertEqual("undetermined", result["action"])

    def test_unverified_is_undetermined(self) -> None:
        result = pheme_replay.auto_decision(predicted_impact=10.0, credibility="unverified", confidence=0.0)
        self.assertEqual("undetermined", result["action"])

    def test_agent_failure_is_undetermined_never_auto_blocked(self) -> None:
        result = pheme_replay.auto_decision(predicted_impact=10.0, credibility="agent_failure", confidence=None)
        self.assertEqual("undetermined", result["action"])


class UnrelatedBranchesNotBlockedTest(unittest.TestCase):
    """Requirement #8: nodes under a DIFFERENT source thread must never be
    swept into another thread's blocked count -- each cascade dict is
    already scoped to one thread_id by construction (load_cascade), and
    realized_blocked_nodes must not reach outside the cascade it's given."""

    def test_realized_blocked_nodes_only_ever_sees_its_own_cascade_argument(self) -> None:
        cascade_a = _make_cascade([
            {"id": "src_a", "parent_id": None, "is_source": True, "offset_sec": None, "observed_by_cutoff": True, "screen_name": "a", "text": "x"},
            {"id": "a_future", "parent_id": "src_a", "is_source": False, "offset_sec": 5000.0, "observed_by_cutoff": False, "screen_name": "b", "text": "y"},
        ])
        cascade_a["thread_id"] = "thread_a"
        result = pheme_replay.realized_blocked_nodes(cascade_a, 0.0, 999999.0)
        self.assertEqual("thread_a", result["thread_id"])
        self.assertEqual(["a_future"], result["realized_blocked_node_ids"])


if __name__ == "__main__":
    unittest.main()
