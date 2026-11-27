"""Small synthetic full-path checks; never use held-out research data."""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from triage_pipeline import core, full, methods, selection
from triage_pipeline.reporting import recommendation, render_report


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def data(group_count: int) -> tuple[list[core.Thread], dict]:
    threads, clean = [], {}
    for group in range(group_count):
        when = START + timedelta(days=group * 3)
        for item in range(2):
            tid = f"t{group}-{item}"
            thread = core.Thread(tid, when, f"root-{tid}", 20 + item,
                                 f"g{group:02}", when, when + timedelta(days=8))
            threads.append(thread)
            clean[tid] = [core.Reply(tid, f"{tid}-a", tid, f"other-{tid}", 30 + item),
                          core.Reply(tid, f"{tid}-b", tid, f"other-{tid}", 3600 + group + item)]
    return threads, clean


def profile(group_count: int) -> dict:
    item = {"status": "complete", "H_hours": 24, "T_minutes": 10,
            "L_seconds": 604800, "analysis_threads": group_count * 2,
            "unknown_followup": False, "possibly_truncated_target": False,
            "zero_reply_fraction": 0.0, "zero_reply_future_other_share": 0.0,
            "T_curve": {"10": {"candidate_threads": group_count * 2,
                               "candidate_coverage": 1.0, "remaining_share": .5}}}
    return item


class FullSyntheticTests(unittest.TestCase):
    def _record(self, mode: str) -> dict:
        return {"config": {"top_fraction": .5, "min_replies": 1, "split_mode": mode},
                "results": {}}

    def _check_report(self, result: dict, prof: dict, mode: str):
        report = "\n".join(render_report(result, {"negative_offset": 1}, prof, mode))
        for field in ("95% interval", "Fixed top 10", "Spearman", "Valid groups",
                      "Candidate T", "Candidate threads", "Later-reply coverage",
                      "Zero-reply fraction", "L:", "Group size min / median / max",
                      "Cleaning", "Selected method"):
            self.assertIn(field.lower(), report.lower(), field)
        self.assertIn("thresholds", str(recommendation(result, mode)))

    def test_time_model_rule_and_report_flags(self):
        threads, clean = data(15)
        flags = {thread.thread_id: {"cohort_a": thread.thread_id.endswith("-0") and
                                            int(thread.group_id[1:]) % 2 == 0}
                 for thread in threads}
        rec = self._record("time")
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(methods, "SEEDS", (1,)), \
             patch.object(methods, "PARAMS", {"n_estimators": 2, "max_depth": 2, "min_samples_leaf": 1}):
            for chosen in ("M", "U"):
                choice = {"status": "selected", "chosen": chosen, "best_rule": "U",
                          "model_minus_best_rule_point": .03 if chosen == "M" else 0.0,
                          "model_minus_best_rule_ci": {"lower": .01 if chosen == "M" else -.01, "upper": .05}}
                with patch.object(full, "select_from_scored", return_value=(choice, {})):
                    first = full.time_run(Path(directory), rec, threads, clean, profile(9), "hash", flags, draws=4)
                    flipped = {tid: {"cohort_a": not value["cohort_a"]} for tid, value in flags.items()}
                    second = full.time_run(Path(directory), rec, threads, clean, profile(9), "hash", flipped, draws=4)
                self.assertEqual(first["status"], "selected")
                self.assertNotEqual(first.pop("subset_test_metrics"), second.pop("subset_test_metrics"))
                self.assertEqual(first, second)
                first["subset_test_metrics"] = {}
                self._check_report(first, profile(9), "time")
                self.assertEqual(recommendation(first, "time")["selected_method"], chosen)

    def test_leave_group_out_model_rule_and_insufficient(self):
        threads, clean = data(4)
        prof = {"folds": {f"g{i:02}": profile(3) for i in range(4)}}
        flags = {thread.thread_id: {"cohort_a": True} for thread in threads}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(methods, "SEEDS", (1,)), \
             patch.object(methods, "PARAMS", {"n_estimators": 2, "max_depth": 2, "min_samples_leaf": 1}):
            for chosen in ("M", "R1"):
                choice = {"status": "selected", "chosen": chosen, "best_rule": "R1",
                          "model_minus_best_rule_point": .03 if chosen == "M" else 0,
                          "model_minus_best_rule_ci": {"lower": .01 if chosen == "M" else -.01, "upper": .05}}
                with patch.object(full, "select_from_scored", return_value=(choice, {})):
                    result = full.group_out_run(Path(directory), self._record("leave_group_out"),
                                                threads, clean, prof, "hash", flags, draws=4)
                    flipped = {tid: {"cohort_a": False} for tid in flags}
                    comparison = full.group_out_run(Path(directory), self._record("leave_group_out"),
                                                    threads, clean, prof, "hash", flipped, draws=4)
                for left, right in zip(result["folds"], comparison["folds"]):
                    self.assertNotEqual(left.pop("subset_test_metrics"), right.pop("subset_test_metrics"))
                self.assertEqual(result, comparison)
                self._check_report(result, prof, "leave_group_out")
                self.assertTrue(all(fold["selection"]["chosen"] == chosen for fold in result["folds"]))
            failed = {"folds": {name: {"status": "insufficient_data", "reason": "empty_analysis_set"}
                                for name in prof["folds"]}}
            result = full.group_out_run(Path(directory), self._record("leave_group_out"),
                                        threads, clean, failed, "hash", flags, draws=4)
            self.assertEqual(result["status"], "insufficient_data")
            self.assertEqual(len(result["folds"]), 4)
            self.assertIn("Insufficient data", "\n".join(render_report(result, {}, failed, "leave_group_out")))
            partial = {"folds": dict(prof["folds"])}
            partial["folds"]["g00"] = {"status": "insufficient_data", "reason": "empty_analysis_set"}
            partial_result = full.group_out_run(Path(directory), self._record("leave_group_out"),
                                                threads, clean, partial, "hash", flags, draws=4)
            self.assertEqual(partial_result["status"], "insufficient_data")
            self.assertEqual(partial_result["folds"][0]["status"], "insufficient_data")
            self.assertIn("Insufficient data", "\n".join(render_report(partial_result, {}, partial,
                                                                       "leave_group_out")))

    def test_time_profile_insufficient_is_reported(self):
        result = full.time_run(Path("unused"), self._record("time"), [], {},
                               {"status": "insufficient_data", "reason": "empty_analysis_set"}, "hash")
        self.assertEqual(result["status"], "insufficient_data")
        self.assertEqual(recommendation(result, "time")["reason"], "empty_analysis_set")
        self.assertIn("insufficient_data", "\n".join(render_report(result, {}, {}, "time")))

    def test_report_limit_markers_and_close_cutoffs(self):
        prof = profile(3)
        prof.update(unknown_followup=True, possibly_truncated_target=True,
                    no_nested_replies=True)
        prof["T_curve"]["30"] = {"candidate_threads": 2,
                                  "candidate_coverage": .9, "remaining_share": .495}
        result = {"status": "insufficient_data", "reason": "no_training_candidates",
                  "segment_sizes": {"train": {"groups": 2, "threads": 101, "candidates": 0,
                                               "group_size_min": 1, "group_size_median": 50.5,
                                               "group_size_max": 100,
                                               "groups_below_one_percent_median": 1}}}
        text = "\n".join(render_report(result, {}, prof, "time"))
        for phrase in ("T 選擇接近平手", "Follow-up length unknown", "Target may be truncated",
                       "No nested replies", "below 1%", "Insufficient data"):
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
