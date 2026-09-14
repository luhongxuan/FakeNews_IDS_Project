"""Read-only inventory for the Twitter15/Twitter16 RumDetect2017 data.

The report deliberately records aggregate coverage only: it never emits tweet
text, user names, or raw tweet objects.  Mutable engagement counts returned by
late hydration are reported as availability metadata, not treated as early
features.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
from statistics import median
from typing import Iterable


TREE_NODE = re.compile(r"\['[^']*', '([^']+)', '([^']+)'\]")
LABEL_LINE = re.compile(r"^([^:]+):(\d+)$")


def nonempty(value: object) -> bool:
    return value is not None and value != ""


def labels(path: Path) -> tuple[dict[str, str], Counter[str]]:
    values: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = LABEL_LINE.match(raw.strip())
        if not match:
            raise ValueError(f"Malformed label at {path}:{line_number}")
        label, source_id = match.groups()
        if source_id in values:
            raise ValueError(f"Duplicate source ID in {path}: {source_id}")
        values[source_id] = label
        counts[label] += 1
    return values, counts


def source_tweet_ids(path: Path) -> set[str]:
    ids = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        source_id, separator, _ = raw.partition("\t")
        if not separator or not source_id.isdigit():
            raise ValueError(f"Malformed source tweet row at {path}:{line_number}")
        ids.add(source_id)
    return ids


def tree_inventory(tree_dir: Path) -> tuple[set[str], dict[str, object]]:
    tree_ids: set[str] = set()
    all_node_ids: set[str] = set()
    edges_per_tree: list[int] = []
    nodes_per_tree: list[int] = []
    max_delay_minutes: list[float] = []
    malformed_lines = 0
    for path in sorted(tree_dir.glob("*.txt")):
        tree_ids.add(path.stem)
        node_ids: set[str] = set()
        delays: list[float] = []
        edges = 0
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            matches = TREE_NODE.findall(raw)
            if len(matches) != 2 or "->" not in raw:
                malformed_lines += 1
                continue
            for tweet_id, delay in matches:
                if tweet_id != "ROOT":
                    node_ids.add(tweet_id)
                    try:
                        delays.append(float(delay))
                    except ValueError:
                        malformed_lines += 1
            edges += 1
        all_node_ids.update(node_ids)
        edges_per_tree.append(edges)
        nodes_per_tree.append(len(node_ids))
        max_delay_minutes.append(max(delays) if delays else 0.0)
    if not edges_per_tree:
        raise ValueError(f"No tree files found in {tree_dir}")
    return all_node_ids, {
        "tree_files": len(edges_per_tree),
        "unique_tree_sources": len(tree_ids),
        "unique_tree_node_ids": len(all_node_ids),
        "malformed_tree_lines": malformed_lines,
        "edges_per_tree": {"min": min(edges_per_tree), "median": median(edges_per_tree), "max": max(edges_per_tree)},
        "nodes_per_tree": {"min": min(nodes_per_tree), "median": median(nodes_per_tree), "max": max(nodes_per_tree)},
        "max_delay_minutes_per_tree": {"min": min(max_delay_minutes), "median": median(max_delay_minutes), "max": max(max_delay_minutes)},
    }


def jsonl_inventory(path: Path, tree_node_ids: set[str]) -> dict[str, object]:
    status_counts: Counter[str] = Counter()
    populated: Counter[str] = Counter()
    seen_ids: set[str] = set()
    duplicate_ids = 0
    malformed_lines = 0
    covered_tree_ids: set[str] = set()
    canonical_counts: Counter[str] = Counter()
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue
            if not isinstance(row, dict) or not isinstance(row.get("tweet_id"), str):
                malformed_lines += 1
                continue
            total += 1
            tweet_id = row["tweet_id"]
            if tweet_id in seen_ids:
                duplicate_ids += 1
            seen_ids.add(tweet_id)
            if tweet_id in tree_node_ids:
                covered_tree_ids.add(tweet_id)
            status_counts[str(row.get("status", "missing"))] += 1
            for field in ("text", "created_at", "username", "reply_count", "retweet_count", "like_count", "scraped_at"):
                if nonempty(row.get(field)):
                    populated[field] += 1
            if "is_canonical" in row:
                canonical_counts[str(row.get("is_canonical"))] += 1
    return {
        "file": str(path),
        "rows": total,
        "unique_tweet_ids": len(seen_ids),
        "duplicate_tweet_id_rows": duplicate_ids,
        "malformed_or_invalid_rows": malformed_lines,
        "status_counts": dict(status_counts),
        "populated_field_rows": dict(populated),
        "is_canonical_counts": dict(canonical_counts),
        "tree_node_coverage": {
            "covered_unique_tree_nodes": len(covered_tree_ids),
            "tree_nodes_total": len(tree_node_ids),
            "coverage_fraction": len(covered_tree_ids) / len(tree_node_ids) if tree_node_ids else 0.0,
        },
        "research_safety": {
            "static_candidate_inputs": ["text", "created_at"],
            "do_not_use_as_early_inputs_without_historical_snapshot_proof": ["reply_count", "retweet_count", "like_count", "scraped_at"],
            "reason": "Hydration happened after the original propagation, so mutable engagement values are post-hoc observations.",
        },
    }


def best_hydrated_rows(path: Path) -> dict[str, dict]:
    """Keep the highest-quality hydration record for each tweet ID."""
    best: dict[str, dict] = {}
    for raw in path.open(encoding="utf-8"):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        tweet_id = row.get("tweet_id") if isinstance(row, dict) else None
        if not isinstance(tweet_id, str):
            continue
        quality = int(row.get("status") == "ok") + int(nonempty(row.get("text"))) + int(nonempty(row.get("created_at")))
        previous = best.get(tweet_id)
        previous_quality = (
            int(previous.get("status") == "ok") + int(nonempty(previous.get("text"))) + int(nonempty(previous.get("created_at")))
            if previous else -1
        )
        if quality > previous_quality:
            best[tweet_id] = row
    return best


def tree_offsets(tree_dir: Path) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Return non-negative cumulative delays and flag invalid negative entries."""
    result: dict[str, dict[str, float]] = {}
    negative_entries = 0
    affected_trees: set[str] = set()
    for path in sorted(tree_dir.glob("*.txt")):
        offsets: dict[str, float] = {}
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            for tweet_id, delay in TREE_NODE.findall(raw):
                if tweet_id == "ROOT":
                    continue
                value = float(delay)
                if value < 0:
                    negative_entries += 1
                    affected_trees.add(path.stem)
                    continue
                offsets[tweet_id] = min(offsets.get(tweet_id, value), value)
        result[path.stem] = offsets
    return result, {"negative_delay_entries_excluded": negative_entries, "trees_with_negative_delays": len(affected_trees)}


def early_text_coverage(dataset_dir: Path, cutoff_minutes: float = 30.0) -> dict[str, object] | None:
    hydrated_path = dataset_dir / "hydrated_tweets.jsonl"
    if not hydrated_path.exists():
        return None
    hydrated = best_hydrated_rows(hydrated_path)
    offsets_by_tree, timing_issues = tree_offsets(dataset_dir / "tree")
    observed_nodes, observed_replies, usable_replies = [], [], []
    threads_with_usable_reply = 0
    for source_id, offsets in offsets_by_tree.items():
        observed = {tweet_id for tweet_id, delay in offsets.items() if delay <= cutoff_minutes}
        replies = observed - {source_id}
        usable = {
            tweet_id for tweet_id in replies
            if hydrated.get(tweet_id, {}).get("status") == "ok"
            and nonempty(hydrated[tweet_id].get("text"))
            and nonempty(hydrated[tweet_id].get("created_at"))
        }
        observed_nodes.append(len(observed))
        observed_replies.append(len(replies))
        usable_replies.append(len(usable))
        threads_with_usable_reply += int(bool(usable))
    total_replies = sum(observed_replies)
    total_usable = sum(usable_replies)
    return {
        "cutoff_minutes": cutoff_minutes,
        "threads": len(offsets_by_tree),
        "observed_nodes_per_thread": {"median": median(observed_nodes), "max": max(observed_nodes)},
        "observed_replies_per_thread": {"median": median(observed_replies), "max": max(observed_replies)},
        "usable_hydrated_reply_text_per_thread": {"median": median(usable_replies), "max": max(usable_replies)},
        "threads_with_at_least_one_usable_hydrated_reply": threads_with_usable_reply,
        "thread_fraction_with_at_least_one_usable_hydrated_reply": threads_with_usable_reply / len(offsets_by_tree),
        "observed_reply_text_coverage": total_usable / total_replies if total_replies else 0.0,
        "timing_integrity": timing_issues,
        "safety_note": "Source-tweet text is separately available in source_tweets.txt. This coverage measures hydrated reply text only; mutable engagement counts are excluded. Negative-delay entries are excluded from this snapshot audit and must be investigated before dataset construction.",
    }


def dataset_inventory(dataset_dir: Path) -> dict[str, object]:
    label_map, label_counts = labels(dataset_dir / "label.txt")
    source_ids = source_tweet_ids(dataset_dir / "source_tweets.txt")
    tree_node_ids, tree_summary = tree_inventory(dataset_dir / "tree")
    hydrated = {}
    for name in ("hydrated_tweets.jsonl", "hydrated_tweets_cleaned.jsonl"):
        path = dataset_dir / name
        if path.exists():
            hydrated[name] = jsonl_inventory(path, tree_node_ids)
    return {
        "dataset": dataset_dir.name,
        "labels": {"counts": dict(label_counts), "source_ids": len(label_map)},
        "source_tweets": {"source_ids": len(source_ids), "matches_label_ids": source_ids == set(label_map)},
        "tree": {**tree_summary, "tree_sources_match_label_ids": set(label_map) == {path.stem for path in (dataset_dir / "tree").glob("*.txt")}},
        "hydrated_files": hydrated,
        "early_text_coverage": early_text_coverage(dataset_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only RumDetect2017 coverage audit")
    parser.add_argument("--root", type=Path, default=Path("data/raw/rumdetect2017"))
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path outside the protected raw dataset")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite report: {args.output}")
    report = {
        "created_at": datetime.now().isoformat(),
        "root": str(root),
        "datasets": {name: dataset_inventory(root / name) for name in ("twitter15", "twitter16")},
        "research_safety": "Read-only inventory. No raw data, labels, trees, or hydrated files were modified. Aggregate metadata only; no tweet text is exported.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Audit report saved to: {args.output}")


if __name__ == "__main__":
    main()
