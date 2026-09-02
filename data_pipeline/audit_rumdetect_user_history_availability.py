"""Measure time-safe user-history feature availability in RumDetect trees.

For each user action, absolute time is reconstructed only as
``source tweet Snowflake time + original tree delay``.  A user's history at an
action never includes actions at the same or a later timestamp.  This is an
availability audit: it exports aggregate counts only, not user identifiers or
features for model fitting.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from data_pipeline.audit_rumdetect_user_action_nodes import parse_tree  # noqa: E402
from data_pipeline.audit_rumdetect2017_timing import snowflake_time  # noqa: E402


CUTOFF_MINUTES = 30.0


def quantiles(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "max": int(array.max()),
        "nonzero_fraction": float(np.mean(array > 0)),
    }


def audit_dataset(dataset: Path) -> dict:
    actions: list[tuple[datetime, str, str, float]] = []
    skipped = Counter()
    for path in sorted((dataset / "tree").glob("*.txt")):
        nodes, _, roots, malformed = parse_tree(path)
        if malformed or len(roots) != 1:
            raise ValueError(f"Unexpected tree syntax/root count: {path}")
        source = roots[0]
        source_time = snowflake_time(source.tweet_id)
        if source_time is None or source.delay_minutes != 0 or any(node.delay_minutes < 0 for node in nodes):
            skipped["invalid_source_time_or_negative_delay_tree"] += 1
            continue
        for node in nodes:
            timestamp = source_time + timedelta(minutes=node.delay_minutes)
            actions.append((timestamp, node.user_id, path.stem, node.delay_minutes))
    actions.sort(key=lambda value: value[0])
    prior_actions: Counter[str] = Counter()
    prior_threads: dict[str, set[str]] = defaultdict(set)
    early_history_by_thread: dict[str, list[tuple[int, int]]] = defaultdict(list)
    index = 0
    while index < len(actions):
        end = index + 1
        while end < len(actions) and actions[end][0] == actions[index][0]:
            end += 1
        # Every action at this timestamp sees exactly the same strictly-past
        # history; same-time actions are added only after being measured.
        for _, user_id, thread_id, delay in actions[index:end]:
            if delay <= CUTOFF_MINUTES:
                early_history_by_thread[thread_id].append((prior_actions[user_id], len(prior_threads[user_id])))
        for _, user_id, thread_id, _ in actions[index:end]:
            prior_actions[user_id] += 1
            prior_threads[user_id].add(thread_id)
        index = end
    per_thread_mean_actions = []
    per_thread_max_actions = []
    per_thread_mean_threads = []
    per_thread_max_threads = []
    for values in early_history_by_thread.values():
        action_history = [value[0] for value in values]
        thread_history = [value[1] for value in values]
        per_thread_mean_actions.append(int(round(float(np.mean(action_history)))))
        per_thread_max_actions.append(max(action_history))
        per_thread_mean_threads.append(int(round(float(np.mean(thread_history)))))
        per_thread_max_threads.append(max(thread_history))
    user_total_actions = list(prior_actions.values())
    user_total_threads = [len(value) for value in prior_threads.values()]
    return {
        "dataset": dataset.name,
        "absolute_time_definition": "source tweet Snowflake time + original source-relative action delay",
        "history_rule": "strictly earlier timestamp only; same-timestamp actions are excluded from each other's history",
        "trees_skipped": dict(skipped),
        "valid_action_rows": len(actions),
        "distinct_users": len(prior_actions),
        "user_lifetime_action_multiplicity": quantiles(user_total_actions),
        "user_lifetime_thread_multiplicity": quantiles(user_total_threads),
        "30min_thread_summary": {
            "threads": len(early_history_by_thread),
            "mean_prior_actions_per_early_user": quantiles(per_thread_mean_actions),
            "max_prior_actions_among_early_users": quantiles(per_thread_max_actions),
            "mean_prior_threads_per_early_user": quantiles(per_thread_mean_threads),
            "max_prior_threads_among_early_users": quantiles(per_thread_max_threads),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit time-safe RumDetect user history availability")
    parser.add_argument("--root", type=Path, default=Path("data/raw/rumdetect2017"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {output}")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "twitter15": audit_dataset(root / "twitter15"),
        "twitter16": audit_dataset(root / "twitter16"),
        "research_safety": "Read-only aggregate audit; no user IDs, raw text, labels, splits, derived records, or model artifacts are exported or changed.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"User-history availability audit saved to: {output}")


if __name__ == "__main__":
    main()
