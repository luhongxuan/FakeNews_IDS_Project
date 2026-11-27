"""Reference feature gate, imported only during the consistency stage."""
from __future__ import annotations

import importlib.util
import math
from collections import Counter, defaultdict

from triage_pipeline import core


def check(threads, replies, source_counts, references, root):
    path = root / "analysis/20261110_123946_bluesky_takeoff_h2/run_experiment.py"
    spec = importlib.util.spec_from_file_location("reference_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tuple(module.FEATURES) != core.FEATURES:
        raise ValueError("reference feature columns differ")
    by_thread = defaultdict(list)
    for reply in replies:
        by_thread[reply.thread_id].append(reply)
    ref_by_id = {row["thread_id"]: row for row in references}
    compared = 0
    excluded = Counter()
    mismatched = Counter()
    for completed, thread in enumerate(threads, 1):
        raw = ref_by_id[thread.thread_id]
        clean, base_counts = core.basic_clean(thread, by_thread[thread.thread_id])
        for cutoff_min in (10, 30, 60):
            cutoff = cutoff_min * 60
            observed, snapshot_counts = core.snapshot(thread, clean, cutoff)
            if base_counts or snapshot_counts:
                excluded.update(base_counts)
                excluded.update(snapshot_counts)
                continue
            actual = core.features(thread, observed, cutoff)
            expected = module.features_at(raw["replies"], raw["root_author"], cutoff_min)
            expected["text_chars"] = raw["text_chars"]
            for name in core.FEATURES:
                if not math.isclose(actual[name], expected[name], rel_tol=0, abs_tol=1e-9):
                    mismatched[name] += 1
            compared += 1
        if completed % 10000 == 0 or completed == len(threads):
            print(f"    feature consistency {completed}/{len(threads)}", flush=True)
    if mismatched:
        raise ValueError("reference feature consistency failed")
    return {"passed": True, "compared_thread_cutoffs": compared,
            "excluded_by_cleaning_reason": dict(excluded), "mismatched_features": dict(mismatched)}
