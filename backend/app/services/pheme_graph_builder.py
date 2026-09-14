"""Build the backend's PHEME user graph from raw thread directories.

This runtime copy deliberately lives inside ``backend`` so the API remains a
self-contained deployable unit.  The research-oriented ``graph_analysis``
package has an equivalent builder, but the backend must not depend on a
repository-root ``sys.path`` mutation or an extra Docker bind mount.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import networkx as nx


def _parse_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
    except ValueError:
        return None


def build_graph_from_pheme(base_path: str, event: str):
    """Return ``(directed_user_graph, first_seen_date, rumour_count)``."""

    graph = nx.DiGraph()
    first_seen: datetime | None = None
    rumour_count = 0
    event_path = Path(base_path) / f"{event}-all-rnr-threads"

    for label in ("rumours", "non-rumours"):
        label_path = event_path / label
        if not label_path.exists():
            continue

        for thread_path in label_path.iterdir():
            thread_id = thread_path.name
            is_rumour = label == "rumours"
            if is_rumour:
                # Preserve the existing API's historical count semantics:
                # every entry under ``rumours`` is counted, including dataset
                # sidecar entries that are not thread directories. Changing
                # this would silently alter event-summary output during a
                # dependency-only refactor.
                rumour_count += 1
            if not thread_path.is_dir():
                continue

            tweet_to_user: dict[str, str] = {}
            for folder in (thread_path / "source-tweets", thread_path / "reactions"):
                if not folder.exists():
                    continue
                for tweet_path in folder.iterdir():
                    if tweet_path.name.startswith("._") or not tweet_path.is_file():
                        continue
                    try:
                        with tweet_path.open(encoding="utf-8", errors="ignore") as handle:
                            tweet = json.load(handle)
                    except (OSError, json.JSONDecodeError):
                        continue

                    tweet_id = tweet.get("id_str")
                    user = tweet.get("user", {}).get("screen_name")
                    if tweet_id and user:
                        tweet_to_user[str(tweet_id)] = str(user)

                    created_at = _parse_date(tweet.get("created_at"))
                    if created_at and (first_seen is None or created_at < first_seen):
                        first_seen = created_at

            structure_path = thread_path / "structure.json"
            if not structure_path.exists():
                continue
            try:
                with structure_path.open(encoding="utf-8") as handle:
                    structure = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue

            def traverse(node_dict, parent_id=None):
                if not isinstance(node_dict, dict):
                    return
                for tweet_id, children in node_dict.items():
                    if parent_id is not None:
                        source = tweet_to_user.get(str(parent_id))
                        target = tweet_to_user.get(str(tweet_id))
                        if source and target and source != target:
                            graph.add_edge(
                                source,
                                target,
                                thread_id=thread_id,
                                is_rumour=is_rumour,
                            )
                    traverse(children, tweet_id)

            traverse(structure)

    date = first_seen.strftime("%Y-%m-%d") if first_seen else None
    return graph, date, rumour_count
