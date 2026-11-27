"""Read existing public-post collections into the unified schema."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from triage_pipeline.core import Reply, Thread, parse_time

RUN_NAMES = ("20261110_095950_full", "20261109_163551_full",
             "20261109_093759_full", "20261114_165934_full")


def digest(value: str, namespace: str) -> str:
    return hashlib.sha256((namespace + "|" + value).encode()).hexdigest()


def manifest(root: Path, source_counts: dict | None = None) -> dict:
    out = {}
    for name in RUN_NAMES:
        directory = root / "analysis/bluesky_cascade_collection/runs" / name
        for filename in ("run_record.json", "posts_raw.jsonl", "threads.jsonl"):
            path = directory / filename
            out[f"{name}/{filename}"] = {"path": str(path.resolve()),
                                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return out


def load(root: Path, progress=None) -> tuple[list[Thread], list[Reply], dict, list[dict]]:
    base = root / "analysis/bluesky_cascade_collection/runs"
    ordered = []
    for name in RUN_NAMES:
        directory = base / name
        record = json.loads((directory / "run_record.json").read_text(encoding="utf-8"))
        ordered.append((parse_time(record["started_at"]), directory))
    ordered.sort(key=lambda pair: pair[0])
    posts = {}
    trees = {}
    counts = Counter()
    for completed, (collected_at, directory) in enumerate(ordered, 1):
        run_posts = {}
        with (directory / "posts_raw.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                run_posts.setdefault(row["uri"], row)
        run_trees = {}
        with (directory / "threads.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                run_trees.setdefault(row["uri"], row)
        for uri, post in run_posts.items():
            if uri in posts:
                counts["duplicate_uris_kept_earliest"] += 1
                continue
            posts[uri] = (post, collected_at)
            trees[uri] = run_trees.get(uri)
        if progress:
            progress(completed, len(ordered))
    threads, replies, reference_rows = [], [], []
    for uri, (post, collected_at) in posts.items():
        tree = trees[uri]
        if tree is None and post["reply_count"] > 0:
            counts["positive_reply_count_without_tree_excluded"] += 1
            continue
        if tree is None:
            counts["zero_reply_without_tree"] += 1
        created = parse_time(post["created_at"])
        day = created.date().isoformat()
        tid = digest(uri, "thread")
        root_author = digest(uri.split("/")[2], "author")
        threads.append(Thread(tid, created, root_author, int(post["text_chars"]), day,
                              datetime.fromisoformat(day).replace(tzinfo=timezone.utc), collected_at))
        raw_replies = tree["replies"] if tree else []
        for index, row in enumerate(raw_replies):
            parent_index = row[2]
            parent = tid if parent_index == -1 else digest(uri + ":" + str(parent_index), "reply")
            author = root_author if row[3] == post["author_hash"] else digest(row[3], "author")
            replies.append(Reply(tid, digest(uri + ":" + str(index), "reply"), parent,
                                 author, row[0]))
        reference_rows.append({"thread_id": tid, "replies": raw_replies,
                               "root_author": post["author_hash"], "text_chars": post["text_chars"]})
    counts["threads"] = len(threads)
    counts["replies"] = len(replies)
    return threads, replies, dict(counts), reference_rows
