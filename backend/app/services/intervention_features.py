"""Builds the Balanced Cumulative RF's 57 `cumulative__` features from a live Bluesky
reply tree, at a given checkpoint (seconds since the source post).

Reuses the exact leakage-safe feature logic from effective_models/
pheme_multicheckpoint_rf/features/cumulative_features.py instead of
reimplementing it by hand -- a hand port could silently drift from what the
RF was actually trained on and no one would notice until predictions got
quietly worse. Only the input adapter is new: PHEME training reads a flat
reply-level CSV; this instead adapts bluesky_source.build_cascade_graph's
{nodes, edges} output (root + replies, each with offset_sec and depth) into
the same per-reply frame the canonical cumulative builder expects.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from effective_models.pheme_multicheckpoint_rf import config as _model_config
from effective_models.pheme_multicheckpoint_rf.features.cumulative_features import (
    build_cumulative_features as _build_model_cumulative_features,
)

_REPLY_COLUMNS = ["tweet_id", "parent_id", "depth", "offset_sec", "reply_latency_sec", "text"]

CHECKPOINTS = _model_config.CHECKPOINTS  # (600, 1200, 1800, 2400, 3000, 3600) seconds


def nearest_usable_checkpoint(elapsed_seconds: float) -> float | None:
    """Largest trained checkpoint that has actually elapsed for this thread.

    Scoring at a checkpoint the thread hasn't reached yet would hand the RF
    a feature vector built from less real observation than the checkpoint
    implies (e.g. calling it "the 30-minute score" after only 12 minutes),
    silently understating how immature the read is. None means the thread
    is younger than the first checkpoint (10 minutes) -- not scoreable yet.
    """
    usable = [c for c in CHECKPOINTS if c <= elapsed_seconds]
    return max(usable) if usable else None


def _replies_frame(nodes: list[dict], source_id: str) -> pd.DataFrame:
    offset_by_id = {node["id"]: node["offset_sec"] for node in nodes}
    records = []
    for node in nodes:
        if node["is_source"] or node["offset_sec"] is None:
            continue  # unparsed timestamp -- can't place it on the checkpoint timeline
        parent_id = node["parent_id"] if node["parent_id"] is not None else source_id
        parent_offset = offset_by_id.get(parent_id)
        # reply_latency_sec mirrors PHEME's column: time since the immediate
        # parent, not since the root (that's offset_sec). Falls back to
        # offset_sec itself on the rare unresolved-parent case.
        if parent_offset is None:
            latency = node["offset_sec"]
        else:
            latency = max(node["offset_sec"] - parent_offset, 0.0)
        records.append({
            "tweet_id": node["id"],
            "parent_id": parent_id,
            "depth": float(node.get("depth", 1)),
            "offset_sec": float(node["offset_sec"]),
            "reply_latency_sec": float(latency),
            "text": node["text"],
        })
    return pd.DataFrame.from_records(records, columns=_REPLY_COLUMNS)


def build_cumulative_features(nodes: list[dict], checkpoint_sec: float) -> dict[str, float]:
    """`nodes` is bluesky_source.build_cascade_graph(...)["nodes"]. Returns a
    dict keyed exactly like intervention_model.feature_columns() (the
    `cumulative__...` names), using only replies observed by checkpoint_sec
    -- the same leakage boundary the RF was trained under.
    """
    root = next((node for node in nodes if node["is_source"]), None)
    if root is None:
        raise ValueError("thread has no source node")
    source_id = root["id"]
    source_text = pd.Series([root["text"]])

    replies = _replies_frame(nodes, source_id)
    observed = replies.loc[replies.offset_sec.le(checkpoint_sec)] if len(replies) else replies

    return _build_model_cumulative_features(observed, source_text, source_id, checkpoint_sec)
