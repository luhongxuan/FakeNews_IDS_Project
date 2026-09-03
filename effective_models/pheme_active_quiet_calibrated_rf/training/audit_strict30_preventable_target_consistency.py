"""Read-only audit of strict-30 preventable-impact root handling."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT / "data" / "protected_research_assets" / "pheme_v5_strict30"
GRAPHS = ASSETS / "pheme_graphs_roberta_30min_replyv4_preventableimpact_semantic.pt"
REPLIES = ASSETS / "pheme_reply_level_v4.csv"
OUT_ROOT = HERE.parent / "experiments"
CUTOFF_SEC = 1800.0


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def blocked_from_roots(
    parents: dict[str, str | None],
    offsets: dict[str, float],
    observed: set[str],
    roots: list[str],
) -> int:
    children: dict[str, list[str]] = defaultdict(list)
    for node_id, parent_id in parents.items():
        if parent_id is not None:
            children[parent_id].append(node_id)
    blocked = 0
    visited: set[str] = set()

    def visit(node_id: str, inherited: bool) -> None:
        nonlocal blocked
        if node_id in visited:
            raise ValueError(f"cycle or duplicate traversal at {node_id}")
        visited.add(node_id)
        if offsets[node_id] <= CUTOFF_SEC:
            child_block = node_id in observed
        else:
            if inherited:
                blocked += 1
            child_block = inherited
        for child_id in children.get(node_id, []):
            if child_id in offsets:
                visit(child_id, child_block)

    for root in roots:
        visit(root, False)
    return blocked


def main() -> None:
    output = OUT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + "_strict30_preventable_target_consistency_audit"
    )
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "started_at": now(),
        "purpose": "read-only audit of strict-30 preventable target root handling",
        "inputs": {"graphs": str(GRAPHS.resolve()), "replies": str(REPLIES.resolve())},
        "cutoff_seconds": CUTOFF_SEC,
        "research_safety": "No dataset, label, split, or model is modified. Future nodes are read only to audit existing target definitions.",
    }
    write(output, record)
    try:
        print("Starting strict-30 preventable-target consistency audit.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/4] Loading protected strict-30 graph labels...", flush=True)
        graphs = torch.load(GRAPHS, weights_only=False)
        graph_by_id = {str(graph.thread_id): graph for graph in graphs}
        if len(graph_by_id) != len(graphs):
            raise ValueError("duplicate graph thread IDs")

        print("[2/4] Loading protected reply trees...", flush=True)
        raw = pd.read_csv(
            REPLIES,
            usecols=["thread_id", "tweet_id", "parent_id", "is_source", "offset_sec", "event_id"],
            dtype={"thread_id": str, "tweet_id": str, "parent_id": str, "event_id": str},
            low_memory=False,
        )
        raw = raw.loc[raw.thread_id.isin(graph_by_id)].copy()
        if raw.offset_sec.isna().any() or (raw.offset_sec < 0).any():
            raise ValueError("invalid offsets in protected reply trees")

        print("[3/4] Comparing protected, legacy-last-root, source-root, and all-root targets...", flush=True)
        rows = []
        groups = list(raw.groupby("thread_id", sort=True))
        for index, (thread_id, group) in enumerate(groups, 1):
            graph = graph_by_id[thread_id]
            parents = {
                str(row.tweet_id): None if pd.isna(row.parent_id) else str(row.parent_id)
                for row in group.itertuples(index=False)
            }
            offsets = {str(row.tweet_id): float(row.offset_sec) for row in group.itertuples(index=False)}
            source_rows = group.loc[group.is_source.eq(1)]
            if len(source_rows) != 1:
                raise ValueError(f"{thread_id}: expected one source")
            source_id = str(source_rows.iloc[0].tweet_id)
            observed = set(map(str, graph.node_ids))
            if any(offsets[node_id] > CUTOFF_SEC for node_id in observed):
                raise ValueError(f"{thread_id}: protected graph contains post-cutoff node")
            roots = [node_id for node_id, parent_id in parents.items() if parent_id is None]
            if not roots:
                raise ValueError(f"{thread_id}: no parent-null root")
            protected = int(round(float(graph.preventable_impact.item())))
            legacy = blocked_from_roots(parents, offsets, observed, [roots[-1]])
            source_root = blocked_from_roots(parents, offsets, observed, [source_id])
            all_roots = blocked_from_roots(parents, offsets, observed, roots)
            rows.append(
                {
                    "thread_id": thread_id,
                    "event_id": str(graph.event_id),
                    "root_count": len(roots),
                    "source_is_legacy_last_root": source_id == roots[-1],
                    "protected_impact": protected,
                    "legacy_last_root_impact": legacy,
                    "source_root_impact": source_root,
                    "all_roots_impact": all_roots,
                    "protected_matches_legacy": protected == legacy,
                    "protected_matches_source": protected == source_root,
                    "protected_matches_all_roots": protected == all_roots,
                }
            )
            if index % 400 == 0 or index == len(groups):
                print(f"  target audit {index}/{len(groups)}", flush=True)

        print("[4/4] Aggregating discrepancy evidence and writing results...", flush=True)
        detail = pd.DataFrame(rows)
        summary = {
            "graphs": len(detail),
            "multi_root_threads": int((detail.root_count > 1).sum()),
            "source_not_legacy_last_root": int((~detail.source_is_legacy_last_root).sum()),
            "protected_mismatch_legacy": int((~detail.protected_matches_legacy).sum()),
            "protected_mismatch_source": int((~detail.protected_matches_source).sum()),
            "protected_mismatch_all_roots": int((~detail.protected_matches_all_roots).sum()),
            "protected_impact_sum": int(detail.protected_impact.sum()),
            "source_root_impact_sum": int(detail.source_root_impact.sum()),
            "all_roots_impact_sum": int(detail.all_roots_impact.sum()),
            "maximum_source_absolute_difference": int(
                (detail.protected_impact - detail.source_root_impact).abs().max()
            ),
        }
        detail.to_csv(output / "per_thread_target_consistency.csv", index=False)
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        record.update(
            {
                "status": "complete",
                "completed_at": now(),
                "summary": summary,
                "output_files": ["per_thread_target_consistency.csv", "summary.json", "run_record.json"],
            }
        )
        write(output, record)
        print(
            f"  multi-root={summary['multi_root_threads']}; "
            f"protected-vs-source mismatches={summary['protected_mismatch_source']}",
            flush=True,
        )
        print(f"SUCCESS: strict-30 target audit saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update(
            {
                "status": "failed",
                "failed_at": now(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "output_files": sorted(path.name for path in output.iterdir()),
            }
        )
        write(output, record)
        print(f"FAILURE: strict-30 target audit preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
