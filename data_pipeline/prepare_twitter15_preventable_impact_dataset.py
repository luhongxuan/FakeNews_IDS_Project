"""Create a versioned, source-anchored Twitter15 intervention-proxy dataset.

This builder mirrors the established PHEME *thread-level* policy: at the
30-minute cutoff, an intervention applies to every node already observed in a
selected source thread.  A future node is counted only when it remains
reachable through an unbroken, time-respecting future path.

Twitter15's raw tree edges are not all temporally valid.  This script therefore
uses Twitter Snowflake timestamps, excludes threads containing pre-source
nodes, and derives a deterministic forest from strictly forward raw edges.
Future nodes/edges are used only to construct the supervised outcome; they are
never emitted as snapshot inputs.  It does not create a train/validation/test
split and never overwrites a previous derived dataset.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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
from data_pipeline.audit_rumdetect2017_timing import parse_tree, snowflake_time  # noqa: E402


CUTOFF_SECONDS = 30 * 60.0
FOREST_POLICY = "strictly_forward_parent_time_lt_child_time_latest_parent_tiebreak_parent_id"


def read_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        label, separator, source_id = raw.strip().partition(":")
        if not separator or not source_id.isdigit() or source_id in labels:
            raise ValueError(f"Invalid label row at {path}:{line_number}")
        labels[source_id] = label
    return labels


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def forward_forest(
    source_id: str,
    node_ids: set[str],
    edges: list[tuple[str, str]],
    node_times: dict[str, datetime],
) -> tuple[dict[str, str | None], dict[str, list[str]], dict[str, int]]:
    """Return a deterministic, strictly forward forest reachable from source."""
    candidates: dict[str, list[str]] = defaultdict(list)
    dropped_self_loops = 0
    dropped_parent_after_child = 0
    for parent_id, child_id in edges:
        if parent_id == child_id:
            dropped_self_loops += 1
            continue
        if node_times[parent_id] >= node_times[child_id]:
            dropped_parent_after_child += 1
            continue
        candidates[child_id].append(parent_id)

    parent_map: dict[str, str | None] = {source_id: None}
    multi_parent_nodes = 0
    duplicate_forward_edges = 0
    for child_id in sorted(node_ids - {source_id}):
        raw_parents = candidates.get(child_id, [])
        parents = sorted(set(raw_parents))
        duplicate_forward_edges += len(raw_parents) - len(parents)
        if not parents:
            continue
        if len(parents) > 1:
            multi_parent_nodes += 1
        # The latest eligible parent is the closest temporally preceding raw
        # reply.  Parent-ID tie-breaking makes repeated runs deterministic.
        parent_map[child_id] = max(parents, key=lambda parent_id: (node_times[parent_id], parent_id))

    children: dict[str, list[str]] = defaultdict(list)
    for child_id, parent_id in parent_map.items():
        if parent_id is not None:
            children[parent_id].append(child_id)
    for child_ids in children.values():
        child_ids.sort()

    reachable: set[str] = set()
    stack = [source_id]
    while stack:
        node_id = stack.pop()
        if node_id in reachable:
            raise ValueError(f"Unexpected cycle in strictly forward forest for {source_id}")
        reachable.add(node_id)
        stack.extend(children.get(node_id, []))

    reachable_parent_map = {node_id: parent_map[node_id] for node_id in reachable}
    reachable_children = {
        node_id: [child_id for child_id in children.get(node_id, []) if child_id in reachable]
        for node_id in reachable
    }
    return reachable_parent_map, reachable_children, {
        "raw_edges": len(edges),
        "raw_forward_edges": sum(len(values) for values in candidates.values()),
        "unique_forward_candidate_edges": sum(len(set(values)) for values in candidates.values()),
        "duplicate_forward_edges": duplicate_forward_edges,
        "dropped_self_loop_edges": dropped_self_loops,
        "dropped_parent_after_child_edges": dropped_parent_after_child,
        "multi_parent_nodes": multi_parent_nodes,
        "reachable_nodes": len(reachable),
        "unreachable_nodes": len(node_ids) - len(reachable),
    }


def blocked_future_nodes(
    source_id: str,
    children: dict[str, list[str]],
    offsets_seconds: dict[str, float],
    intervene_ids: set[str],
) -> int:
    """PHEME-compatible continuous-future-chain intervention accounting."""
    blocked = 0

    def visit(node_id: str, inherited_block: bool) -> None:
        nonlocal blocked
        if offsets_seconds[node_id] <= CUTOFF_SECONDS:
            child_block = node_id in intervene_ids
        else:
            if inherited_block and node_id != source_id:
                blocked += 1
            child_block = inherited_block
        for child_id in children.get(node_id, []):
            visit(child_id, child_block)

    visit(source_id, False)
    return blocked


def early_snapshot_features(
    source_id: str,
    parent_map: dict[str, str | None],
    offsets_seconds: dict[str, float],
    observed_ids: set[str],
) -> dict[str, float]:
    """Return logical 30-minute activity/structure features, never future data."""
    reply_offsets = np.sort(np.asarray([
        offsets_seconds[node_id] for node_id in observed_ids if node_id != source_id
    ], dtype=float))
    interarrival = np.diff(reply_offsets)
    observed_children: dict[str, int] = defaultdict(int)
    for node_id in observed_ids:
        parent_id = parent_map[node_id]
        if parent_id in observed_ids:
            observed_children[parent_id] += 1
    child_counts = np.asarray([observed_children[node_id] for node_id in observed_ids], dtype=float)
    latest_reply = float(reply_offsets.max()) if reply_offsets.size else 0.0
    return {
        "log1p_observed_nodes": float(math.log1p(len(observed_ids))),
        "log1p_count_0_10m": float(math.log1p(np.sum(reply_offsets <= 600))),
        "log1p_count_10_20m": float(math.log1p(np.sum((reply_offsets > 600) & (reply_offsets <= 1200)))),
        "log1p_count_20_30m": float(math.log1p(np.sum((reply_offsets > 1200) & (reply_offsets <= CUTOFF_SECONDS)))),
        "log1p_count_recent_5m": float(math.log1p(np.sum(reply_offsets > 1500))),
        "log1p_count_recent_10m": float(math.log1p(np.sum(reply_offsets > 1200))),
        "log1p_seconds_since_last_activity": float(math.log1p(CUTOFF_SECONDS - latest_reply)),
        "log1p_median_interarrival_sec": float(math.log1p(np.median(interarrival))) if interarrival.size else 0.0,
        "log1p_std_interarrival_sec": float(math.log1p(np.std(interarrival))) if interarrival.size else 0.0,
        "late_activity_frac": float(np.mean(reply_offsets > 1200)) if reply_offsets.size else 0.0,
        "observed_leaf_fraction": float(np.mean(child_counts == 0)),
        "mean_observed_children": float(child_counts.mean()),
        "max_observed_children": float(child_counts.max()),
    }


def build_records(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    labels = read_labels(dataset_dir / "label.txt")
    records: list[dict] = []
    exclusions: list[dict] = []
    aggregate = defaultdict(int)

    for source_id, source_label in sorted(labels.items()):
        source_time = snowflake_time(source_id)
        if source_time is None:
            raise ValueError(f"Invalid source Snowflake ID: {source_id}")
        node_delays, edges, malformed = parse_tree(dataset_dir / "tree" / f"{source_id}.txt")
        if malformed:
            raise ValueError(f"Malformed tree rows in {source_id}")
        node_ids = set(node_delays) | {source_id}
        node_times = {node_id: snowflake_time(node_id) for node_id in node_ids}
        invalid_ids = sorted(node_id for node_id, value in node_times.items() if value is None)
        if invalid_ids:
            raise ValueError(f"Invalid node Snowflake IDs in {source_id}: {invalid_ids[:3]}")
        valid_times = {node_id: value for node_id, value in node_times.items() if value is not None}
        pre_source_nodes = sorted(node_id for node_id, value in valid_times.items() if value < source_time)
        if pre_source_nodes:
            exclusions.append({
                "thread_id": source_id,
                "original_label": source_label,
                "reason": "pre_source_node",
                "pre_source_node_count": len(pre_source_nodes),
            })
            aggregate["pre_source_threads_excluded"] += 1
            continue

        parent_map, children, forest_stats = forward_forest(source_id, node_ids, edges, valid_times)
        offsets = {
            node_id: (valid_times[node_id] - source_time).total_seconds()
            for node_id in parent_map
        }
        observed_ids = {node_id for node_id, offset in offsets.items() if offset <= CUTOFF_SECONDS}
        future_ids = set(parent_map) - observed_ids
        if source_id not in observed_ids:
            raise ValueError(f"Source is absent from 30-minute snapshot: {source_id}")
        for node_id, parent_id in parent_map.items():
            if parent_id is not None and not offsets[parent_id] < offsets[node_id]:
                raise ValueError(f"Non-forward forest edge in {source_id}: {parent_id}->{node_id}")
        snapshot_edges = sum(
            1
            for child_id, parent_id in parent_map.items()
            if parent_id is not None and child_id in observed_ids and parent_id in observed_ids
        )
        if any(offsets[node_id] > CUTOFF_SECONDS for node_id in observed_ids):
            raise ValueError(f"Future node found in snapshot: {source_id}")

        preventable = blocked_future_nodes(source_id, children, offsets, observed_ids)
        future_reachable = len(future_ids)
        if preventable > future_reachable:
            raise ValueError(f"Preventable target exceeds future forest nodes: {source_id}")
        # This equality follows from the PHEME thread-level policy: the source
        # and every other observed forest node are all intervened on.  Keeping
        # it as an assertion documents that this target is a causal-safe
        # future-growth proxy rather than an additional graph-specific signal.
        if preventable != future_reachable:
            raise ValueError(f"Unexpected intervention-policy mismatch in {source_id}")
        early_features = early_snapshot_features(source_id, parent_map, offsets, observed_ids)
        if not all(math.isfinite(value) for value in early_features.values()):
            raise ValueError(f"Non-finite early feature in {source_id}")
        records.append({
            "thread_id": source_id,
            "original_label": source_label,
            "raw_node_count": len(node_ids),
            **forest_stats,
            "observed_node_count": len(observed_ids),
            "observed_edge_count": snapshot_edges,
            "future_reachable_node_count": future_reachable,
            "preventable_impact": preventable,
            "preventable_y": math.log1p(preventable),
            **early_features,
        })
        aggregate["included_threads"] += 1
        aggregate["raw_nodes_included_threads"] += len(node_ids)
        aggregate["reachable_nodes"] += forest_stats["reachable_nodes"]
        aggregate["future_reachable_nodes"] += future_reachable
        aggregate["preventable_impact"] += preventable

    metadata = {
        "source_trees_total": len(labels),
        **dict(aggregate),
        "cutoff_seconds": CUTOFF_SECONDS,
        "forest_policy": FOREST_POLICY,
        "target": "log1p(future nodes blocked after intervening on every source-anchored, observed forest node)",
        "target_scope": "Derived intervention proxy matching PHEME's thread-level policy; it is not an observed counterfactual.",
        "target_equivalence": (
            "Because the policy intervenes on the source and every observed forest node, preventable_impact equals "
            "future_reachable_node_count for every retained thread. The forest validates causal reachability, but this "
            "thread-level target is not an additional graph-structure signal beyond causal-safe future growth."
        ),
        "snapshot_safety": (
            "Node timestamps use Twitter Snowflakes. Included threads have no pre-source nodes; snapshot records "
            "contain only nodes at or before cutoff; snapshot edges connect only observed nodes and are strictly forward."
        ),
        "future_information_policy": (
            "Future timestamps and time-respecting forest edges are used solely to construct preventable_impact supervision. "
            "They must never be used as model inputs, normalization data, split criteria, or model-selection information."
        ),
        "early_feature_policy": (
            "Early activity, recency, interarrival, and observed-forest summaries are calculated solely from nodes and "
            "edges at or before cutoff. Source tweet is excluded from reply-count and interarrival summaries."
        ),
    }
    return pd.DataFrame(records), pd.DataFrame(exclusions), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a time-respecting Twitter15 preventable-impact proxy dataset")
    parser.add_argument("--dataset", type=Path, default=Path("data/raw/rumdetect2017/twitter15"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing derived dataset: {output}")

    records, exclusions, metadata = build_records(dataset)
    if records.empty:
        raise ValueError("No time-safe Twitter15 threads were retained")
    if not (records.preventable_y == records.preventable_impact.map(math.log1p)).all():
        raise ValueError("Inconsistent log1p preventable target")
    output.mkdir(parents=True, exist_ok=False)
    records.to_csv(output / "thread_level_records.csv", index=False)
    exclusions.to_csv(output / "excluded_threads.csv", index=False)
    metadata.update({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "input_sha256": {
            "label.txt": sha256(dataset / "label.txt"),
            "tree_file_count": len(list((dataset / "tree").glob("*.txt"))),
        },
        "outputs": {
            "thread_level_records": "thread_level_records.csv",
            "excluded_threads": "excluded_threads.csv",
        },
        "research_safety": (
            "New derived records only. Raw RumDetect2017 files and labels are untouched; no split is generated; "
            "no model is trained; this builder refuses to overwrite an existing output directory."
        ),
    })
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Created {len(records)} time-safe Twitter15 derived thread records: {output}")


if __name__ == "__main__":
    main()
