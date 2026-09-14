"""Shared, leakage-safe corrected-target utilities for PHEME v5."""
from __future__ import annotations

from collections import defaultdict
import copy
import math
from pathlib import Path

import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT / "data" / "protected_research_assets" / "pheme_v5_strict30"
GRAPHS = ASSETS / "pheme_graphs_roberta_30min_replyv4_preventableimpact_semantic.pt"
REPLIES = ASSETS / "pheme_reply_level_v4.csv"
CUTOFF_SEC = 1800.0


def load_reply_maps() -> dict[str, tuple[dict[str, str | None], dict[str, float]]]:
    frame = pd.read_csv(
        REPLIES,
        usecols=["thread_id", "tweet_id", "parent_id", "offset_sec"],
        dtype={"thread_id": str, "tweet_id": str, "parent_id": str},
        low_memory=False,
    )
    if frame.duplicated(["thread_id", "tweet_id"]).any():
        raise ValueError("duplicate protected reply identity")
    if frame.offset_sec.isna().any() or (frame.offset_sec < 0).any():
        raise ValueError("invalid protected reply offset")
    maps = {}
    for thread_id, group in frame.groupby("thread_id", sort=False):
        parents = {
            str(row.tweet_id): None if pd.isna(row.parent_id) else str(row.parent_id)
            for row in group.itertuples(index=False)
        }
        offsets = {
            str(row.tweet_id): float(row.offset_sec)
            for row in group.itertuples(index=False)
        }
        maps[str(thread_id)] = parents, offsets
    return maps


def corrected_preventable_impact(
    parents: dict[str, str | None],
    offsets: dict[str, float],
    observed_ids: set[str],
    cutoff_sec: float = CUTOFF_SEC,
) -> int:
    """Count blocked future nodes across every independently traversable root."""
    children: dict[str, list[str]] = defaultdict(list)
    roots = []
    for node_id, parent_id in parents.items():
        if parent_id is None or parent_id not in parents:
            roots.append(node_id)
        else:
            children[parent_id].append(node_id)
    if not roots:
        raise ValueError("reply tree has no traversable root")
    blocked = 0
    visited: set[str] = set()

    def visit(node_id: str, inherited: bool) -> None:
        nonlocal blocked
        if node_id in visited:
            raise ValueError(f"cycle or duplicate traversal at {node_id}")
        visited.add(node_id)
        if offsets[node_id] <= cutoff_sec:
            child_block = node_id in observed_ids
        else:
            if inherited:
                blocked += 1
            child_block = inherited
        for child_id in children.get(node_id, []):
            visit(child_id, child_block)

    for root in roots:
        visit(root, False)
    if len(visited) != len(parents):
        raise ValueError("not every reply-tree node was traversed")
    return blocked


def corrected_graph(graph, reply_maps: dict) -> tuple[object, dict]:
    thread_id = str(graph.thread_id)
    if thread_id not in reply_maps:
        raise KeyError(f"missing reply tree for {thread_id}")
    parents, offsets = reply_maps[thread_id]
    observed = set(map(str, graph.node_ids))
    if not observed or any(node_id not in offsets for node_id in observed):
        raise ValueError(f"invalid observed node identity for {thread_id}")
    if any(offsets[node_id] > CUTOFF_SEC for node_id in observed):
        raise ValueError(f"post-cutoff node in protected snapshot {thread_id}")
    impact = corrected_preventable_impact(parents, offsets, observed)
    result = copy.deepcopy(graph)
    result.preventable_impact = torch.tensor([impact], dtype=torch.float)
    result.preventable_y = torch.tensor([math.log1p(impact)], dtype=torch.float)
    if list(map(str, result.node_ids)) != list(map(str, graph.node_ids)):
        raise ValueError(f"node identity/order changed for {thread_id}")
    for tensor_name in ("x", "edge_index"):
        if not torch.equal(getattr(result, tensor_name), getattr(graph, tensor_name)):
            raise ValueError(f"{tensor_name} changed for {thread_id}")
    if not torch.allclose(result.preventable_y, torch.log1p(result.preventable_impact)):
        raise ValueError(f"target transform mismatch for {thread_id}")
    return result, {
        "thread_id": thread_id,
        "event_id": str(graph.event_id),
        "legacy_preventable_impact": int(round(float(graph.preventable_impact.item()))),
        "corrected_preventable_impact": impact,
        "target_delta": impact - int(round(float(graph.preventable_impact.item()))),
        "root_count": sum(
            parent_id is None or parent_id not in parents for parent_id in parents.values()
        ),
    }


def correct_graphs(graphs: list, reply_maps: dict, progress_every: int = 400) -> tuple[list, pd.DataFrame]:
    corrected, rows = [], []
    for index, graph in enumerate(graphs, 1):
        result, row = corrected_graph(graph, reply_maps)
        corrected.append(result)
        rows.append(row)
        if index % progress_every == 0 or index == len(graphs):
            print(f"  corrected targets {index}/{len(graphs)}", flush=True)
    return corrected, pd.DataFrame(rows)
