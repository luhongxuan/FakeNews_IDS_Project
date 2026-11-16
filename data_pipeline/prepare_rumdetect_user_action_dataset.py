"""Build time-safe RumDetect user-action snapshots without model training.

Nodes are full ``(user_id, tweet_id, delay_minutes)`` actions.  This preserves
different users propagating the same content.  Original tree delays are the
only action-time source.  A conservative strictly-forward canonical forest is
used solely for the explicit causal-reachability proxy; all ordinary snapshot
features use actions observable by the 30-minute cutoff.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from data_pipeline.audit_rumdetect_user_action_nodes import Action, parse_tree  # noqa: E402


CUTOFF_MINUTES = 30.0


def read_labels(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        label, separator, thread_id = raw.strip().partition(":")
        if not separator or not thread_id.isdigit() or thread_id in result:
            raise ValueError(f"Invalid label row at {path}:{line_number}")
        result[thread_id] = label
    return result


def canonical_forest(nodes: set[Action], edges: list[tuple[Action, Action]], source: Action) -> tuple[dict[Action, Action | None], dict[Action, list[Action]], dict[str, int]]:
    """Build an acyclic forest from strictly forward action edges only."""
    candidates: dict[Action, set[Action]] = defaultdict(set)
    stats: Counter[str] = Counter({
        "raw_edges": len(edges),
        "dropped_exact_self_action_edges": 0,
        "dropped_nonforward_or_equal_edges": 0,
        "multi_parent_actions": 0,
        "strict_forward_candidate_edges": 0,
        "reachable_actions": 0,
        "unreachable_actions": 0,
    })
    for parent, child in edges:
        if parent == child:
            stats["dropped_exact_self_action_edges"] += 1
        elif parent.delay_minutes < child.delay_minutes:
            candidates[child].add(parent)
        else:
            stats["dropped_nonforward_or_equal_edges"] += 1
    parent_map: dict[Action, Action | None] = {source: None}
    for child, parents in candidates.items():
        if child == source:
            continue
        if len(parents) > 1:
            stats["multi_parent_actions"] += 1
        parent_map[child] = max(parents, key=lambda parent: (parent.delay_minutes, parent.key))
    children: dict[Action, list[Action]] = defaultdict(list)
    for child, parent in parent_map.items():
        if parent is not None:
            children[parent].append(child)
    for values in children.values():
        values.sort(key=lambda action: action.key)
    reachable: set[Action] = set()
    stack = [source]
    while stack:
        action = stack.pop()
        if action in reachable:
            raise ValueError(f"Cycle in strictly-forward canonical forest rooted at {source.key}")
        reachable.add(action)
        stack.extend(children.get(action, []))
    parent_map = {action: parent_map[action] for action in reachable}
    children = {action: [child for child in children.get(action, []) if child in reachable] for action in reachable}
    stats["strict_forward_candidate_edges"] = sum(len(parents) for parents in candidates.values())
    stats["reachable_actions"] = len(reachable)
    stats["unreachable_actions"] = len(nodes) - len(reachable)
    return parent_map, children, dict(stats)


def feature_row(source: Action, parent_map: dict[Action, Action | None], children: dict[Action, list[Action]]) -> dict[str, float]:
    observed = {action for action in parent_map if 0 <= action.delay_minutes <= CUTOFF_MINUTES}
    if source not in observed:
        raise ValueError(f"Source missing from snapshot: {source.key}")
    reply_delays = np.sort(np.asarray([action.delay_minutes for action in observed if action != source], dtype=float))
    user_counts = Counter(action.user_id for action in observed)
    content_counts = Counter(action.tweet_id for action in observed)
    observed_children = np.asarray([
        sum(child in observed for child in children.get(action, [])) for action in observed
    ], dtype=float)
    interarrival = np.diff(reply_delays)
    latest = float(reply_delays.max()) if reply_delays.size else 0.0
    content_probability = np.asarray(list(content_counts.values()), dtype=float) / len(observed)
    root_children = sum(child in observed for child in children.get(source, []))
    return {
        "observed_action_nodes": float(len(observed)),
        "observed_unique_users": float(len(user_counts)),
        "observed_unique_content_ids": float(len(content_counts)),
        "observed_user_repeat_fraction": float(1.0 - len(user_counts) / len(observed)),
        "observed_content_repeat_fraction": float(1.0 - len(content_counts) / len(observed)),
        "observed_content_hhi": float(np.square(content_probability).sum()),
        "max_users_per_observed_content": float(max(content_counts.values())),
        "root_observed_children": float(root_children),
        "log1p_count_0_10m": float(math.log1p(np.sum(reply_delays <= 10))),
        "log1p_count_10_20m": float(math.log1p(np.sum((reply_delays > 10) & (reply_delays <= 20)))),
        "log1p_count_20_30m": float(math.log1p(np.sum((reply_delays > 20) & (reply_delays <= 30)))),
        "log1p_count_recent_5m": float(math.log1p(np.sum(reply_delays > 25))),
        "seconds_since_last_action": float((CUTOFF_MINUTES - latest) * 60.0),
        "median_interarrival_seconds": float(np.median(interarrival) * 60.0) if interarrival.size else 0.0,
        "std_interarrival_seconds": float(np.std(interarrival) * 60.0) if interarrival.size else 0.0,
        "observed_leaf_fraction": float(np.mean(observed_children == 0)),
        "mean_observed_children": float(observed_children.mean()),
        "max_observed_children": float(observed_children.max()),
    }


def build(dataset: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    labels = read_labels(dataset / "label.txt")
    records: list[dict] = []
    exclusions: list[dict] = []
    totals: Counter[str] = Counter()
    for thread_id, label in sorted(labels.items()):
        nodes, edges, roots, malformed = parse_tree(dataset / "tree" / f"{thread_id}.txt")
        if malformed or len(roots) != 1:
            raise ValueError(f"Unexpected tree syntax/root count for {thread_id}")
        source = roots[0]
        if source.delay_minutes != 0 or any(action.delay_minutes < 0 for action in nodes):
            exclusions.append({"thread_id": thread_id, "original_label": label, "reason": "negative_or_nonzero_source_delay"})
            totals["excluded_negative_delay_threads"] += 1
            continue
        parent_map, children, forest_stats = canonical_forest(nodes, edges, source)
        features = feature_row(source, parent_map, children)
        future = {action for action in parent_map if action.delay_minutes > CUTOFF_MINUTES}
        future_users = {action.user_id for action in future}
        # Under the established thread-level policy, all observed actions are
        # intervened on; every future action reachable in this canonical forest
        # is consequently counted by this explicit causal proxy.
        record = {
            "thread_id": thread_id,
            "original_label": label,
            "source_user_id": source.user_id,
            "source_tweet_id": source.tweet_id,
            **forest_stats,
            **features,
            "future_reachable_action_nodes": len(future),
            "future_reachable_unique_users": len(future_users),
            "preventable_action_proxy": len(future),
            "preventable_y": math.log1p(len(future)),
        }
        if not all(math.isfinite(float(value)) for key, value in record.items() if key not in {"thread_id", "original_label", "source_user_id", "source_tweet_id"}):
            raise ValueError(f"Non-finite output for {thread_id}")
        records.append(record)
        totals["included_threads"] += 1
        totals["action_nodes_included"] += len(nodes)
        totals["future_reachable_actions"] += len(future)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No valid user-action threads")
    return frame, pd.DataFrame(exclusions), {
        "dataset": str(dataset),
        "node_identity": "(user_id, tweet_id, source_relative_delay_minutes)",
        "time_source": "original tree delay",
        "cutoff_minutes": CUTOFF_MINUTES,
        "edge_policy": "strict parent_delay < child_delay; latest preceding parent selected for multi-parent actions",
        "target": "log1p(future user actions reachable in canonical forest after intervening on all <=30-minute actions)",
        "target_scope": "Explicit derived causal proxy, not an observed counterfactual.",
        "feature_scope": "All feature fields use only actions and canonical-forest edges at or before 30 minutes.",
        "totals": dict(totals),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create RumDetect user-action snapshots and causal proxies")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing derived dataset: {output}")
    records, exclusions, metadata = build(dataset)
    output.mkdir(parents=True, exist_ok=False)
    records.to_csv(output / "thread_level_records.csv", index=False)
    exclusions.to_csv(output / "excluded_threads.csv", index=False)
    metadata.update({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": {"label.txt": sha256(dataset / "label.txt")},
        "research_safety": (
            "New derived records only. Raw trees, labels, and previous derived datasets are unchanged. Future actions appear "
            "only in explicitly named targets, never in snapshot features. No split or model is created."
        ),
    })
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Created {len(records)} user-action snapshot records: {output}")


if __name__ == "__main__":
    main()
