"""Read archived event threads and reaction JSON into the unified schema."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from statistics import median

from triage_pipeline.core import Reply, Thread


def digest(value: str, namespace: str) -> str:
    return hashlib.sha256((namespace + "|" + value).encode()).hexdigest()


def _read(path: Path, aggregate_hash) -> dict:
    raw = path.read_bytes()
    aggregate_hash.update(str(path.relative_to(path.parents[4])).encode())
    aggregate_hash.update(hashlib.sha256(raw).digest())
    return json.loads(raw)


def _source_file(thread_dir: Path) -> Path:
    for folder in ("source-tweets", "source-tweet"):
        paths = [p for p in (thread_dir / folder).glob("*.json") if not p.name.startswith("._")]
        if paths:
            if len(paths) != 1:
                raise ValueError("source count differs from one")
            return paths[0]
    raise ValueError("missing source JSON")


def manifest(root: Path, source_counts: dict) -> dict:
    base = root / "data/raw/pheme"
    return {"root_path": str(base.resolve()),
            "source_manifest_sha256": source_counts["source_manifest_sha256"]}


def load(root: Path, progress=None) -> tuple[list[Thread], list[Reply], dict, dict]:
    base = root / "data/raw/pheme"
    directories = []
    for event in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("._")):
        for cls in ("rumours", "non-rumours"):
            directories.extend((event.name.replace("-all-rnr-threads", ""), cls, path)
                               for path in sorted((event / cls).iterdir())
                               if path.is_dir() and not path.name.startswith("._"))
    checksum = hashlib.sha256()
    threads, replies = [], []
    classes = {}
    counts = Counter()
    event_times = defaultdict(list)
    for completed, (event, cls, directory) in enumerate(directories, 1):
        source = _read(_source_file(directory), checksum)
        root_raw = source["id_str"]
        root_time = parsedate_to_datetime(source["created_at"])
        root_author_raw = source["user"]["id_str"]
        tid = digest(root_raw, "thread")
        root_author = digest(root_author_raw, "author")
        source_text = source.get("full_text", source.get("text", ""))
        threads.append(Thread(tid, root_time, root_author, len(source_text), event, root_time, None))
        classes[tid] = {"rumour": cls == "rumours"}
        event_times[event].append(root_time.timestamp())
        counts["rumour_threads" if cls == "rumours" else "nonrumour_threads"] += 1
        for path in (directory / "reactions").glob("*.json"):
            if path.name.startswith("._"):
                continue
            reaction = _read(path, checksum)
            if reaction["id_str"] == root_raw:
                continue
            counts["reactions_read"] += 1
            author_raw = (reaction.get("user") or {}).get("id_str")
            when_raw = reaction.get("created_at")
            if author_raw is None or when_raw is None:
                counts["missing_author_or_time"] += 1
                continue
            offset = (parsedate_to_datetime(when_raw) - root_time).total_seconds()
            if author_raw == root_author_raw:
                counts["self_replies"] += 1
            parent_raw = reaction.get("in_reply_to_status_id_str")
            parent = tid if parent_raw == root_raw else digest(str(parent_raw), "reply")
            replies.append(Reply(tid, digest(reaction["id_str"], "reply"), parent,
                                 digest(author_raw, "author"), offset))
        if progress and (completed % 250 == 0 or completed == len(directories)):
            progress(completed, len(directories))
    medians = {event: datetime.fromtimestamp(median(values), timezone.utc)
               for event, values in event_times.items()}
    threads = [replace(thread, group_time=medians[thread.group_id]) for thread in threads]
    counts.update(threads=len(threads), replies=len(replies), events=len(event_times))
    return threads, replies, {**dict(counts), "source_manifest_sha256": checksum.hexdigest()}, classes
