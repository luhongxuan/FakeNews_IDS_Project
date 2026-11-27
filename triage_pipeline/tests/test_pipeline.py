from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from triage_pipeline import core, methods, metrics, selection
from triage_pipeline.full import nearest_training_groups, outer_groups, render_report
from triage_pipeline.run_pipeline import forbidden_output_text

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def thread(name="root", created=BASE, group="g", observed=None, text=10):
    return core.Thread(name, created, "author-root", text, group, created, observed)


def reply(name, offset, parent="root", author="other", root="root"):
    return core.Reply(root, name, parent, author, offset)


class CleaningTests(unittest.TestCase):
    def test_schema_rejects_duplicate_thread_and_unknown_reply(self):
        t = thread()
        self.assertEqual(core.validate_schema([t], [reply("r", 1)]), {"threads": 1, "replies": 1})
        with self.assertRaises(ValueError):
            core.validate_schema([t, t], [])
        with self.assertRaises(ValueError):
            core.validate_schema([t], [reply("r", 1, root="unknown")])

    def test_invalid_duplicate_and_parent_time_counts(self):
        t = thread()
        cleaned, counts = core.basic_clean(t, [reply("bad", None), reply("early", -1),
            reply("dup", 20), reply("dup", 10), reply("child", 5, "dup")])
        self.assertEqual(counts, {"missing_or_nonfinite_offset": 1, "negative_offset": 1,
                                  "duplicate_reply_id": 1, "parent_later_than_child": 1})
        self.assertEqual({r.reply_id: r.offset_sec for r in cleaned}["dup"], 10)

    def test_parent_chain_missing_future_and_cycle(self):
        t = thread()
        raw = [reply("direct", 0), reply("future_parent", 601), reply("child", 100, "future_parent"),
               reply("orphan", 50, "missing"), reply("cycle_a", 20, "cycle_b"),
               reply("cycle_b", 30, "cycle_a")]
        cleaned, _ = core.basic_clean(t, raw)
        observed, counts = core.snapshot(t, cleaned, 600)
        self.assertEqual([(r.reply_id, d) for r, d in observed], [("direct", 1)])
        self.assertEqual(counts["missing_or_unobserved_parent_chain"], 2)
        self.assertEqual(counts["cycle_or_uncomputable_depth"], 2)

    def test_cutoff_features_rules_and_target_boundaries(self):
        t = thread(text=17)
        cleaned, _ = core.basic_clean(t, [reply("zero", 0), reply("on", 600), reply("after", 601),
                                         reply("horizon", 86400), reply("late", 86401),
                                         reply("self", 20, author="author-root")])
        observed, _ = core.snapshot(t, cleaned, 600)
        f = core.features(t, observed, 600)
        self.assertEqual(tuple(f), core.FEATURES)
        self.assertEqual([f[k] for k in ("n_other", "n_self", "uniq_other", "max_depth",
                                         "first_other_s", "n_other_last5", "text_chars")],
                         [2, 1, 1, 1, 0, 1, 17])
        self.assertEqual(core.rules(t, observed, 600), {"U": 1, "R1": 2, "R2": 0})
        self.assertEqual(core.target(t, cleaned, 600, 86400), 2)
        self.assertTrue(core.candidate(f, 1))

    def test_cutoff_perturbation_keeps_candidate_features_and_rules(self):
        t = thread()
        base = [reply("early", 100)]
        mutations = [[*base, reply("late", 601)], [*base, reply("late", 99999)],
                     [*base, reply("late", 601), reply("late2", 602, "late")]]
        results = []
        for raw in [base, *mutations]:
            clean, _ = core.basic_clean(t, raw)
            observed, _ = core.snapshot(t, clean, 600)
            f = core.features(t, observed, 600)
            results.append((f, core.rules(t, observed, 600), core.candidate(f, 1)))
        self.assertTrue(all(result == results[0] for result in results))

    def test_maturity(self):
        t = thread(observed=BASE + timedelta(hours=24))
        self.assertTrue(core.mature(t, 24 * 3600))
        self.assertFalse(core.mature(t, 24 * 3600 + 1))
        self.assertTrue(core.mature(thread(observed=None), 168 * 3600))


class SplitAndProfileTests(unittest.TestCase):
    def test_h_grid_threshold_and_unobservable_horizon(self):
        t = thread(observed=BASE + timedelta(hours=168))
        reply_rows = {t.thread_id: [reply("a", 60), reply("b", 24 * 3600 + 1)]}
        result = core.profile([t], reply_rows, {"T_max_min": 60, "min_replies": 1})
        self.assertEqual(result["H_hours"], 72)
        short = thread(observed=BASE + timedelta(hours=23))
        self.assertEqual(core.profile([short], {short.thread_id: reply_rows[t.thread_id]},
                                      {"T_max_min": 60, "min_replies": 1})["status"], "insufficient_data")

    def test_strict_boundary_zero_reply_start_and_group_loss(self):
        # One early, zero-reply row fixes the selection start. Boundary equality is dropped.
        rows = [thread("a", BASE, "a"),
                thread("b", BASE + timedelta(days=1), "b"),
                thread("c", BASE + timedelta(days=2), "c"),
                thread("d", BASE + timedelta(days=3), "d"),
                thread("e", BASE + timedelta(days=4), "e")]
        segments, report = core.time_segments(rows, 24 * 3600)
        self.assertEqual(report["starts"]["selection"], rows[3].created_at.isoformat())
        self.assertEqual([r.thread_id for r in segments["train"]], ["a", "b"])
        self.assertEqual(report["boundary_dropped"]["train"], {"dropped_threads": 1, "emptied_groups": 1})
        self.assertEqual(report["boundary_dropped"]["selection"], {"dropped_threads": 1, "emptied_groups": 1})
        self.assertEqual([r.thread_id for r in segments["test"]], ["e"])

    def test_h_change_does_not_change_starts_or_test(self):
        rows = [thread(str(i), BASE + timedelta(days=i), str(i)) for i in range(10)]
        a, report_a = core.time_segments(rows, 24 * 3600)
        b, report_b = core.time_segments(rows, 72 * 3600)
        self.assertEqual(report_a["starts"], report_b["starts"])
        self.assertEqual([r.thread_id for r in a["test"]], [r.thread_id for r in b["test"]])

    def test_profile_excludes_late_training_and_is_perturbation_safe(self):
        threads = [thread("old", BASE, "old", BASE + timedelta(hours=168)),
                   thread("excluded", BASE + timedelta(days=9), "excluded",
                          BASE + timedelta(days=16))]
        base = {"old": [reply("r1", 100, "old", root="old"), reply("r2", 90000, "old", root="old")],
                "excluded": [reply("x", 100, "excluded", root="excluded")]}
        cfg = {"T_max_min": 60, "min_replies": 1}
        start = BASE + timedelta(days=10)
        a = core.profile(threads, base, cfg, next_start=start)
        changed = dict(base)
        changed["excluded"] = [reply("x2", 200000, "excluded", root="excluded")]
        b = core.profile(threads, changed, cfg, next_start=start)
        self.assertEqual(a["analysis_threads"], 1)
        self.assertEqual((a["H_hours"], a["T_minutes"]), (b["H_hours"], b["T_minutes"]))
        self.assertEqual(a["L_seconds"], 168 * 3600)

    def test_training_labels_complete_before_selection_start(self):
        rows = [thread(str(i), BASE + timedelta(days=i), str(i)) for i in range(10)]
        segments, report = core.time_segments(rows, 72 * 3600)
        selection_start = core.parse_time(report["starts"]["selection"])
        self.assertTrue(all(row.created_at + timedelta(hours=72) < selection_start
                            for row in segments["train"]))

    def test_learning_curve_uses_nearest_nonempty_groups(self):
        rows = [thread("a", BASE, "a"), thread("c", BASE + timedelta(days=2), "c"),
                thread("e", BASE + timedelta(days=4), "e")]
        self.assertEqual(nearest_training_groups(rows, 1), {"e"})
        self.assertEqual(nearest_training_groups(rows, 2), {"c", "e"})
        self.assertEqual(nearest_training_groups(rows, 7), {"a", "c", "e"})

    def test_outer_group_never_in_train_or_selection(self):
        rows = [thread(str(i), BASE + timedelta(days=i), str(i)) for i in range(4)]
        replies = {row.thread_id: [reply(f"r-{row.thread_id}", 100, row.thread_id,
                                         root=row.thread_id)] for row in rows}
        cfg = {"T_max_min": 60, "min_replies": 1}
        for held, train, test in outer_groups(rows):
            self.assertNotIn(held, {row.group_id for row in train})
            self.assertEqual({row.group_id for row in test}, {held})
            training_profile = core.profile(train, replies, cfg)
            contaminated = dict(replies)
            contaminated[test[0].thread_id] = [reply("sentinel", 999999,
                                                       test[0].thread_id, root=test[0].thread_id)]
            self.assertEqual(training_profile, core.profile(train, contaminated, cfg))
            for inner_held, inner_train, inner_test in outer_groups(train):
                self.assertNotIn(held, {row.group_id for row in inner_train + inner_test})
                self.assertNotIn(inner_held, {row.group_id for row in inner_train})


class MetricAndChoiceTests(unittest.TestCase):
    def test_fixed_model_synthetic_forward_and_bootstrap_seed(self):
        t = thread()
        rows = []
        for index in range(6):
            observed = [(reply(f"r-{index}-{j}", 100 + j), 1) for j in range(index + 1)]
            rows.append({"features": core.features(t, observed, 600), "target": index,
                         "rules": core.rules(t, observed, 600), "group_id": str(index // 2),
                         "cluster_id": str(index)})
        fitted = methods.fit_models(rows)
        self.assertEqual(len(fitted), 3)
        self.assertTrue(all(model.n_estimators == 200 and model.max_depth == 6 and
                            model.min_samples_leaf == 20 for model in fitted))
        scored = [{**row, "scores": score} for row, score in zip(rows, methods.scores(rows, fitted))]
        self.assertEqual(len(scored), len(rows))
        self.assertTrue(all(math.isfinite(row["scores"]["M"]) for row in scored))
        a = metrics.paired_cluster_ci(scored, "M", "U", .1, draws=20, seed=42)
        b = metrics.paired_cluster_ci(scored, "M", "U", .1, draws=20, seed=42)
        self.assertEqual(a, b)
        intervals = metrics.method_intervals(scored, ("M", "U"), .1, draws=20, seed=42)
        self.assertEqual(intervals, metrics.method_intervals(scored, ("M", "U"), .1, draws=20, seed=42))

    def test_tie_hand_calculation(self):
        scores = np.array([1, 1, 0], dtype=float)
        targets = np.array([2, 4, 10], dtype=float)
        self.assertEqual(metrics.expected_topk_sum(scores, targets, 1), 3)
        rows = [{"group_id": "g", "cluster_id": str(i), "target": y,
                 "scores": {"U": s}} for i, (s, y) in enumerate(zip(scores, targets))]
        self.assertAlmostEqual(metrics.grouped_metric(rows, "U", .1)["mean_capture"], .3)
        self.assertEqual(metrics.paired_cluster_ci(rows, "U", "U", .1, draws=20, seed=42)["lower"], 0)

    def test_choice_four_paths(self):
        value = lambda x: {"mean_capture": x}
        m = {"M": value(.65), "U": value(.60), "R1": value(.55), "R2": value(.50)}
        self.assertEqual(selection.choose(m, 499, 3, 3, None)["status"], "insufficient_data")
        self.assertIsNotNone(selection.choose(m, 499, 1, 3, None)["temporary_rules"])
        self.assertIsNone(selection.choose(m, 499, 0, 3, None)["temporary_rules"])
        self.assertEqual(selection.choose(m, 500, 3, 3, {"lower": .01, "upper": .08})["chosen"], "M")
        self.assertEqual(selection.choose(m, 500, 3, 3, {"lower": 0, "upper": .08})["chosen"], "U")
        self.assertEqual(selection.choose(m, 500, 3, 2, {"lower": .01})["status"], "insufficient_data")

    def test_shared_core_has_no_source_names(self):
        root = Path(__file__).resolve().parents[1]
        forbidden = ("bluesky", "pheme", "rumour", "rumors", "tweet", "atproto",
                     "source-tweets", "nonrumour", "is_rumour")
        for path in [root / name for name in ("core.py", "methods.py", "metrics.py",
                                              "selection.py", "reporting.py", "full.py", "run_pipeline.py")]:
            source = path.read_text(encoding="utf-8").lower()
            for word in forbidden:
                self.assertNotIn(word, source, f"{path.name}: {word}")

    def test_output_content_scan(self):
        self.assertFalse(forbidden_output_text('{"count": 1}'))
        self.assertTrue(forbidden_output_text('{"handle": "private"}'))
        self.assertTrue(forbidden_output_text('{"text": "private"}'))

    def test_report_contains_protocol_limits(self):
        lines = render_report({"status": "insufficient_data", "candidate_counts": {}}, {},
                              {"H_hours": 24, "T_minutes": 10, "analysis_threads": 1, "L_seconds": 604800}, "time")
        report = "\n".join(lines)
        self.assertIn("insufficient_data", report)
        self.assertIn("does not assess truth", report)


if __name__ == "__main__":
    unittest.main()
