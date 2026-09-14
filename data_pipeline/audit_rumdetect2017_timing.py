"""Read-only reconciliation of RumDetect2017 tree delays and tweet timestamps.

It compares three timing sources without exporting tweet text:
* original tree delay in minutes;
* timestamp encoded in numeric Twitter Snowflake IDs;
* hydrated ``created_at`` where present.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from statistics import median


TREE_NODE = re.compile(r"\['[^']*', '([^']+)', '([^']+)'\]")
TWITTER_EPOCH_MS = 1_288_834_974_657
PLAUSIBLE_MIN = datetime(2010, 1, 1, tzinfo=timezone.utc)
PLAUSIBLE_MAX = datetime(2027, 1, 1, tzinfo=timezone.utc)


def snowflake_time(tweet_id: str) -> datetime | None:
    if not tweet_id.isdigit():
        return None
    value = int(tweet_id)
    timestamp_ms = (value >> 22) + TWITTER_EPOCH_MS
    try:
        result = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    return result if PLAUSIBLE_MIN <= result <= PLAUSIBLE_MAX else None


def parse_created_at(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            result = datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y")
        except ValueError:
            return None
    return result.astimezone(timezone.utc)


def hydrated_times(path: Path) -> tuple[dict[str, datetime], dict[str, int]]:
    values: dict[str, datetime] = {}
    parse_failures = 0
    duplicate_rows = 0
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            tweet_id = row.get("tweet_id") if isinstance(row, dict) else None
            timestamp = parse_created_at(row.get("created_at")) if isinstance(row, dict) else None
            if not isinstance(tweet_id, str) or timestamp is None:
                if isinstance(row, dict) and row.get("created_at") is not None:
                    parse_failures += 1
                continue
            if tweet_id in values:
                duplicate_rows += 1
            else:
                values[tweet_id] = timestamp
    return values, {"parsed_created_at": len(values), "created_at_parse_failures": parse_failures, "duplicate_timestamp_rows": duplicate_rows}


def parse_tree(path: Path) -> tuple[dict[str, float], list[tuple[str, str]], int]:
    """Return per-node earliest listed delay, directed edges, and malformed count."""
    node_delays: dict[str, float] = {}
    edges: list[tuple[str, str]] = []
    malformed = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        nodes = TREE_NODE.findall(raw)
        if len(nodes) != 2 or "->" not in raw:
            malformed += 1
            continue
        (parent_id, parent_delay), (child_id, child_delay) = nodes
        try:
            parent_value, child_value = float(parent_delay), float(child_delay)
        except ValueError:
            malformed += 1
            continue
        for tweet_id, delay in ((parent_id, parent_value), (child_id, child_value)):
            if tweet_id != "ROOT":
                node_delays[tweet_id] = min(node_delays.get(tweet_id, delay), delay)
        if parent_id != "ROOT" and child_id != "ROOT":
            edges.append((parent_id, child_id))
    return node_delays, edges, malformed


def dataset_report(dataset_dir: Path) -> dict:
    hydration_path = dataset_dir / "hydrated_tweets.jsonl"
    hydrated, hydration_summary = hydrated_times(hydration_path) if hydration_path.exists() else ({}, {})
    totals: Counter[str] = Counter()
    original_vs_snowflake_errors: list[float] = []
    hydrated_vs_snowflake_errors: list[float] = []
    affected_negative_trees: list[str] = []
    source_precedes_all_trees: list[str] = []
    source_has_pre_source_trees: list[str] = []
    unreconcilable_sources: list[str] = []
    reversed_edge_trees: set[str] = set()
    reversed_edge_source_valid_trees: set[str] = set()
    reversed_edge_pre_source_trees: set[str] = set()

    for path in sorted((dataset_dir / "tree").glob("*.txt")):
        source_id = path.stem
        node_delays, edges, malformed = parse_tree(path)
        totals["tree_files"] += 1
        totals["malformed_tree_lines"] += malformed
        source_time = snowflake_time(source_id)
        if source_time is None:
            unreconcilable_sources.append(source_id)
            continue
        node_times = {tweet_id: snowflake_time(tweet_id) for tweet_id in node_delays}
        valid_node_times = {tweet_id: value for tweet_id, value in node_times.items() if value is not None}
        totals["tree_nodes"] += len(node_delays)
        totals["snowflake_valid_nodes"] += len(valid_node_times)
        totals["snowflake_invalid_nodes"] += len(node_delays) - len(valid_node_times)
        negative_nodes = {tweet_id for tweet_id, delay in node_delays.items() if delay < 0}
        if negative_nodes:
            affected_negative_trees.append(source_id)
            totals["negative_delay_unique_nodes"] += len(negative_nodes)
        pre_source_nodes = {tweet_id for tweet_id, value in valid_node_times.items() if value < source_time}
        temporal_bucket = "pre_source_tree" if pre_source_nodes else "source_anchor_valid_tree"
        if pre_source_nodes:
            source_has_pre_source_trees.append(source_id)
            totals["snowflake_pre_source_nodes"] += len(pre_source_nodes)
        else:
            source_precedes_all_trees.append(source_id)
        for tweet_id, delay in node_delays.items():
            node_time = valid_node_times.get(tweet_id)
            if node_time is None:
                continue
            expected_delay = (node_time - source_time).total_seconds() / 60
            original_vs_snowflake_errors.append(delay - expected_delay)
            hydrated_time = hydrated.get(tweet_id)
            if hydrated_time is not None:
                hydrated_vs_snowflake_errors.append((hydrated_time - node_time).total_seconds())
        for parent_id, child_id in edges:
            parent_time, child_time = valid_node_times.get(parent_id), valid_node_times.get(child_id)
            if parent_time is None or child_time is None:
                continue
            totals["snowflake_checkable_edges"] += 1
            totals[f"{temporal_bucket}_checkable_edges"] += 1
            if parent_time > child_time:
                totals["snowflake_parent_after_child_edges"] += 1
                totals[f"{temporal_bucket}_parent_after_child_edges"] += 1
                reversed_edge_trees.add(source_id)
                if temporal_bucket == "source_anchor_valid_tree":
                    reversed_edge_source_valid_trees.add(source_id)
                else:
                    reversed_edge_pre_source_trees.add(source_id)

    def error_summary(values: list[float]) -> dict:
        if not values:
            return {"count": 0}
        absolute = [abs(value) for value in values]
        return {
            "count": len(values),
            "median_signed": median(values),
            "median_absolute": median(absolute),
            "p95_absolute": sorted(absolute)[int(0.95 * (len(absolute) - 1))],
            "max_absolute": max(absolute),
        }

    return {
        "dataset": dataset_dir.name,
        "hydration_created_at": hydration_summary,
        "tree_and_snowflake": {
            **dict(totals),
            "negative_delay_tree_count": len(affected_negative_trees),
            "negative_delay_tree_ids": affected_negative_trees,
            "source_anchor_valid_tree_count": len(source_precedes_all_trees),
            "source_anchor_pre_source_tree_count": len(source_has_pre_source_trees),
            "source_anchor_pre_source_tree_ids": source_has_pre_source_trees,
            "invalid_source_snowflake_count": len(unreconcilable_sources),
            "trees_with_parent_after_child_edges": len(reversed_edge_trees),
            "source_anchor_valid_trees_with_parent_after_child_edges": len(reversed_edge_source_valid_trees),
            "pre_source_trees_with_parent_after_child_edges": len(reversed_edge_pre_source_trees),
        },
        "timing_agreement": {
            "original_delay_minus_snowflake_delay_minutes": error_summary(original_vs_snowflake_errors),
            "hydrated_created_at_minus_snowflake_seconds": error_summary(hydrated_vs_snowflake_errors),
        },
        "safety_interpretation": (
            "A source-anchored snapshot is automatically safe only for trees where every usable node has "
            "a non-negative source-relative timestamp. Trees with pre-source nodes require a pre-registered "
            "pruning or exclusion rule; never clip negative times to zero."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only RumDetect2017 timing reconciliation audit")
    parser.add_argument("--root", type=Path, default=Path("data/raw/rumdetect2017"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output
    if not root.exists():
        raise FileNotFoundError(root)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite report: {output}")
    report = {
        "created_at": datetime.now().isoformat(),
        "root": str(root),
        "twitter15": dataset_report(root / "twitter15"),
        "twitter16": dataset_report(root / "twitter16"),
        "research_safety": "Read-only timing audit. No raw tree, labels, hydrated records, or derived dataset was modified. No tweet text is exported.",
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Timing reconciliation report saved to: {output}")


if __name__ == "__main__":
    main()
