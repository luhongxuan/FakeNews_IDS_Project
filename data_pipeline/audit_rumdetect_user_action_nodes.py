"""Read-only semantic audit of RumDetect2017 propagation tuples.

RumDetect tree rows encode user actions as ``[user_id, tweet_id, delay]``.
This audit deliberately preserves all three fields in node identity.  It exists
because treating ``tweet_id`` alone as a node collapses distinct users who
propagate the same content and creates artificial self-loops.

No derived labels, model inputs, splits, or raw files are written or changed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re


TUPLE = re.compile(r"\['([^']+)', '([^']+)', '([^']+)'\]")
CUTOFF_MINUTES = 30.0


@dataclass(frozen=True, order=True)
class Action:
    user_id: str
    tweet_id: str
    delay_minutes: float

    @property
    def key(self) -> str:
        return f"{self.user_id}:{self.tweet_id}:{self.delay_minutes:.12g}"


def parse_action(raw: tuple[str, str, str], path: Path, line_number: int) -> Action | None:
    user_id, tweet_id, delay_text = raw
    if user_id == "ROOT" and tweet_id == "ROOT":
        return None
    if not user_id or not tweet_id:
        raise ValueError(f"Empty action identity at {path}:{line_number}")
    try:
        delay = float(delay_text)
    except ValueError as exc:
        raise ValueError(f"Invalid delay at {path}:{line_number}: {delay_text}") from exc
    return Action(user_id, tweet_id, delay)


def parse_tree(path: Path) -> tuple[set[Action], list[tuple[Action, Action]], list[Action], int]:
    nodes: set[Action] = set()
    edges: list[tuple[Action, Action]] = []
    root_children: list[Action] = []
    malformed = 0
    for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        values = TUPLE.findall(raw)
        if len(values) != 2 or "->" not in raw:
            malformed += 1
            continue
        parent, child = (parse_action(value, path, line_number) for value in values)
        if child is None:
            raise ValueError(f"ROOT cannot be a child at {path}:{line_number}")
        nodes.add(child)
        if parent is None:
            root_children.append(child)
        else:
            nodes.add(parent)
            edges.append((parent, child))
    return nodes, edges, root_children, malformed


def audit_dataset(dataset_dir: Path) -> dict:
    labels = [line for line in (dataset_dir / "label.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    totals: Counter[str] = Counter()
    root_child_count = Counter()
    source_delay_counts = Counter()
    repeated_content_action_counts: list[int] = []
    snapshot_rows: list[dict] = []
    examples: dict[str, list[dict]] = defaultdict(list)

    for tree_path in sorted((dataset_dir / "tree").glob("*.txt")):
        nodes, edges, root_children, malformed = parse_tree(tree_path)
        thread_id = tree_path.stem
        totals["trees"] += 1
        totals["malformed_lines"] += malformed
        totals["action_nodes"] += len(nodes)
        totals["action_edges"] += len(edges)
        root_child_count[len(root_children)] += 1
        if len(root_children) != 1 and len(examples["non_single_root"]) < 5:
            examples["non_single_root"].append({"thread_id": thread_id, "root_child_count": len(root_children)})
        if not root_children:
            continue
        source = root_children[0]
        source_delay_counts[source.delay_minutes] += 1
        if source.tweet_id != thread_id:
            totals["root_tweet_id_differs_from_filename"] += 1
        content_to_users: dict[str, set[str]] = defaultdict(set)
        for node in nodes:
            content_to_users[node.tweet_id].add(node.user_id)
        repeated_content_action_counts.extend(len(users) for users in content_to_users.values())
        repeated = {tweet_id: users for tweet_id, users in content_to_users.items() if len(users) > 1}
        totals["content_ids_repeated_by_multiple_users"] += len(repeated)
        totals["additional_user_actions_lost_by_tweet_id_collapse"] += sum(len(users) - 1 for users in repeated.values())
        if repeated and len(examples["content_repeated_by_users"]) < 5:
            tweet_id, users = next(iter(repeated.items()))
            examples["content_repeated_by_users"].append({"thread_id": thread_id, "tweet_id": tweet_id, "unique_users": len(users)})

        negative_nodes = [node for node in nodes if node.delay_minutes < 0]
        if negative_nodes:
            totals["trees_with_negative_delay_actions"] += 1
            totals["negative_delay_actions"] += len(negative_nodes)
            if len(examples["negative_delay_actions"]) < 5:
                examples["negative_delay_actions"].append({"thread_id": thread_id, "count": len(negative_nodes)})
        forward_edges = equal_time_edges = backward_edges = same_user_edges = same_tweet_edges = 0
        parent_sets: dict[Action, set[Action]] = defaultdict(set)
        nondecreasing_children: dict[Action, set[Action]] = defaultdict(set)
        nondecreasing_edges: set[tuple[Action, Action]] = set()
        for parent, child in edges:
            if parent.delay_minutes < child.delay_minutes:
                forward_edges += 1
            elif parent.delay_minutes == child.delay_minutes:
                equal_time_edges += 1
            else:
                backward_edges += 1
                if len(examples["backward_edges"]) < 5:
                    examples["backward_edges"].append({
                        "thread_id": thread_id,
                        "parent": parent.key,
                        "child": child.key,
                    })
            same_user_edges += parent.user_id == child.user_id
            same_tweet_edges += parent.tweet_id == child.tweet_id
            if parent == child:
                totals["exact_self_action_edges"] += 1
            if parent.delay_minutes <= child.delay_minutes:
                parent_sets[child].add(parent)
                nondecreasing_children[parent].add(child)
                nondecreasing_edges.add((parent, child))
        totals["strictly_forward_edges"] += forward_edges
        totals["equal_delay_edges"] += equal_time_edges
        totals["backward_delay_edges"] += backward_edges
        totals["same_user_edges"] += same_user_edges
        totals["same_tweet_content_edges"] += same_tweet_edges
        totals["duplicate_action_edges"] += len(edges) - len(set(edges))
        multi_parent = sum(len(parents) > 1 for parents in parent_sets.values())
        totals["actions_with_multiple_nondecreasing_parents"] += multi_parent

        indegree = {node: 0 for node in nodes}
        for parent, child in nondecreasing_edges:
            indegree[child] += 1
        queue = [node for node, degree in indegree.items() if degree == 0]
        processed = 0
        while queue:
            node = queue.pop()
            processed += 1
            for child in nondecreasing_children.get(node, set()):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if processed != len(nodes):
            totals["trees_with_nondecreasing_cycles"] += 1
            if len(examples["nondecreasing_cycles"]) < 5:
                examples["nondecreasing_cycles"].append({"thread_id": thread_id, "nodes_not_topologically_sorted": len(nodes) - processed})

        reachable: set[Action] = set()
        stack = [source]
        while stack:
            node = stack.pop()
            if node in reachable:
                continue
            reachable.add(node)
            stack.extend(nondecreasing_children.get(node, set()))
        totals["nodes_reachable_from_root_via_nondecreasing_edges"] += len(reachable)
        totals["nodes_unreachable_from_root_after_backward_edge_removal"] += len(nodes) - len(reachable)

        observable = [node for node in nodes if 0 <= node.delay_minutes <= CUTOFF_MINUTES]
        future = [node for node in nodes if node.delay_minutes > CUTOFF_MINUTES]
        snapshot_rows.append({
            "thread_id": thread_id,
            "action_nodes": len(nodes),
            "unique_users": len({node.user_id for node in nodes}),
            "unique_tweet_ids": len({node.tweet_id for node in nodes}),
            "observed_action_nodes_30m": len(observable),
            "observed_unique_users_30m": len({node.user_id for node in observable}),
            "future_action_nodes": len(future),
            "future_unique_users": len({node.user_id for node in future}),
            "strictly_forward_edges": forward_edges,
            "equal_delay_edges": equal_time_edges,
            "backward_delay_edges": backward_edges,
            "nondecreasing_reachable_action_nodes": len(reachable),
        })

    if totals["trees"] != len(labels):
        raise ValueError(f"Tree/label count mismatch in {dataset_dir}: {totals['trees']} vs {len(labels)}")
    totals["source_delay_zero_trees"] = source_delay_counts[0.0]
    return {
        "dataset": dataset_dir.name,
        "trees": totals["trees"],
        "node_identity": "(user_id, tweet_id, source_relative_delay_minutes)",
        "time_source": "original tree delay, not tweet Snowflake timestamp",
        "cutoff_minutes": CUTOFF_MINUTES,
        "totals": dict(totals),
        "root_child_count_distribution": {str(key): value for key, value in sorted(root_child_count.items())},
        "source_delay_distribution": {str(key): value for key, value in sorted(source_delay_counts.items())},
        "content_user_multiplicity": {
            "counted_content_ids": len(repeated_content_action_counts),
            "mean_unique_users_per_content_id": sum(repeated_content_action_counts) / len(repeated_content_action_counts),
            "max_unique_users_per_content_id": max(repeated_content_action_counts),
        },
        "examples": dict(examples),
        "snapshot_summary": {
            "threads_with_observed_nodes": sum(row["observed_action_nodes_30m"] > 0 for row in snapshot_rows),
            "mean_observed_action_nodes_30m": sum(row["observed_action_nodes_30m"] for row in snapshot_rows) / len(snapshot_rows),
            "mean_observed_unique_users_30m": sum(row["observed_unique_users_30m"] for row in snapshot_rows) / len(snapshot_rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RumDetect user-action tree semantics")
    parser.add_argument("--root", type=Path, default=Path("data/raw/rumdetect2017"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit report: {output}")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "twitter15": audit_dataset(root / "twitter15"),
        "twitter16": audit_dataset(root / "twitter16"),
        "research_safety": (
            "Read-only audit. It preserves user-action tuple identity and original tree delays; no raw data, labels, "
            "splits, derived datasets, or model results are modified."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"User-action audit saved to: {output}")


if __name__ == "__main__":
    main()
