"""Platform-neutral schema, snapshots, profiling, splitting and evaluation."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Iterable

import numpy as np

FEATURES = ("n_other", "n_self", "uniq_other", "max_depth", "mean_depth",
            "first_other_s", "n_other_last5", "text_chars")
T_GRID_MIN = (10, 30, 60, 180)
H_GRID_H = (24, 72, 168)


@dataclass(frozen=True)
class Thread:
    thread_id: str
    created_at: datetime
    root_author_id: str
    root_text_chars: int
    group_id: str
    group_time: datetime
    observed_until: datetime | None


@dataclass(frozen=True)
class Reply:
    thread_id: str
    reply_id: str
    parent_id: str
    author_id: str
    offset_sec: float | None


def parse_time(value: str | datetime | None) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timestamp must have a timezone")
    return result


def validate_schema(threads: list[Thread], replies: list[Reply]) -> dict:
    ids = set()
    for thread in threads:
        if not thread.thread_id or not thread.root_author_id or thread.root_text_chars < 0:
            raise ValueError("invalid thread schema")
        if thread.created_at.tzinfo is None or thread.group_time.tzinfo is None:
            raise ValueError("thread time must be UTC aware")
        if thread.observed_until is not None and thread.observed_until.tzinfo is None:
            raise ValueError("observed time must be UTC aware")
        if thread.thread_id in ids:
            raise ValueError("duplicate thread id")
        ids.add(thread.thread_id)
    for reply in replies:
        if reply.thread_id not in ids or not reply.reply_id or not reply.parent_id or not reply.author_id:
            raise ValueError("invalid reply schema")
    return {"threads": len(threads), "replies": len(replies)}


def basic_clean(thread: Thread, replies: list[Reply]) -> tuple[list[Reply], dict[str, int]]:
    counts = Counter()
    kept: dict[str, Reply] = {}
    for reply in replies:
        value = reply.offset_sec
        if value is None or not math.isfinite(value):
            counts["missing_or_nonfinite_offset"] += 1
            continue
        if value < 0:
            counts["negative_offset"] += 1
            continue
        if reply.reply_id in kept:
            counts["duplicate_reply_id"] += 1
            if value < kept[reply.reply_id].offset_sec:
                kept[reply.reply_id] = reply
        else:
            kept[reply.reply_id] = reply
    by_id = kept
    for reply in kept.values():
        parent = by_id.get(reply.parent_id)
        if parent is not None and parent.offset_sec > reply.offset_sec:
            counts["parent_later_than_child"] += 1
    return list(kept.values()), dict(counts)


def snapshot(thread: Thread, clean: list[Reply], cutoff_sec: float) -> tuple[list[tuple[Reply, int]], dict[str, int]]:
    """Keep only observable replies with a complete acyclic parent chain."""
    eligible = {reply.reply_id: reply for reply in clean if reply.offset_sec <= cutoff_sec}
    cache: dict[str, int | None] = {}
    reason: dict[str, str] = {}

    def depth(reply_id: str, visiting: set[str]) -> int | None:
        if reply_id in cache:
            return cache[reply_id]
        if reply_id in visiting:
            reason[reply_id] = "cycle_or_uncomputable_depth"
            cache[reply_id] = None
            return None
        reply = eligible[reply_id]
        if reply.parent_id == thread.thread_id:
            cache[reply_id] = 1
            return 1
        if reply.parent_id not in eligible:
            reason[reply_id] = "missing_or_unobserved_parent_chain"
            cache[reply_id] = None
            return None
        parent_depth = depth(reply.parent_id, visiting | {reply_id})
        if parent_depth is None:
            reason[reply_id] = reason.get(reply.parent_id, "missing_or_unobserved_parent_chain")
            cache[reply_id] = None
            return None
        cache[reply_id] = parent_depth + 1
        return cache[reply_id]

    result = []
    for reply in eligible.values():
        value = depth(reply.reply_id, set())
        if value is not None:
            result.append((reply, value))
    return result, dict(Counter(reason.values()))


def features(thread: Thread, observed: list[tuple[Reply, int]], cutoff_sec: float) -> dict[str, float]:
    other = [(reply, depth) for reply, depth in observed if reply.author_id != thread.root_author_id]
    offsets = [reply.offset_sec for reply, _ in other]
    depths = [depth for _, depth in other]
    result = {
        "n_other": float(len(other)),
        "n_self": float(len(observed) - len(other)),
        "uniq_other": float(len({reply.author_id for reply, _ in other})),
        "max_depth": float(max(depths)) if depths else 0.0,
        "mean_depth": float(np.mean(depths)) if depths else 0.0,
        "first_other_s": float(min(offsets)) if offsets else float(cutoff_sec),
        "n_other_last5": float(sum(offset > cutoff_sec - 300 for offset in offsets)),
        "text_chars": float(thread.root_text_chars),
    }
    if tuple(result) != FEATURES or not np.isfinite(list(result.values())).all():
        raise ValueError("invalid feature vector")
    return result


def rules(thread: Thread, observed: list[tuple[Reply, int]], cutoff_sec: float) -> dict[str, float]:
    other = [reply for reply, _ in observed if reply.author_id != thread.root_author_id]
    return {"U": float(len({reply.author_id for reply in other})),
            "R1": float(len(other)),
            "R2": float(-(cutoff_sec - max(reply.offset_sec for reply in other))) if other else -float(cutoff_sec)}


def target(thread: Thread, clean: list[Reply], cutoff_sec: float, horizon_sec: float) -> int:
    return sum(cutoff_sec < reply.offset_sec <= horizon_sec and reply.author_id != thread.root_author_id
               for reply in clean)


def mature(thread: Thread, horizon_sec: float) -> bool:
    return thread.observed_until is None or thread.observed_until >= thread.created_at + timedelta(seconds=horizon_sec)


def candidate(feature_row: dict[str, float], min_replies: int) -> bool:
    return feature_row["n_other"] >= min_replies


def raw_time_groups(threads: list[Thread]) -> tuple[dict[str, list[Thread]], dict[str, datetime | None]]:
    groups = defaultdict(list)
    group_times = {}
    for thread in threads:
        groups[thread.group_id].append(thread)
        group_times[thread.group_id] = thread.group_time
    ordered = sorted(groups, key=lambda key: (group_times[key], key))
    n = len(ordered)
    n_train, n_select, n_test = math.floor(.6 * n), math.floor(.2 * n), math.floor(.2 * n)
    n_train += n - n_train - n_select - n_test
    keys = {"train": ordered[:n_train], "selection": ordered[n_train:n_train + n_select],
            "test": ordered[n_train + n_select:]}
    segments = {name: [thread for key in group_ids for thread in groups[key]] for name, group_ids in keys.items()}
    starts = {name: min((thread.created_at for thread in rows), default=None) for name, rows in segments.items()}
    return segments, starts


def drop_previous_boundary(threads: list[Thread], next_start: datetime, horizon_sec: float) -> tuple[list[Thread], dict]:
    kept = [thread for thread in threads if thread.created_at + timedelta(seconds=horizon_sec) < next_start]
    return kept, {"dropped_threads": len(threads) - len(kept),
                  "emptied_groups": len({t.group_id for t in threads} - {t.group_id for t in kept})}


def time_segments(threads: list[Thread], horizon_sec: float) -> tuple[dict[str, list[Thread]], dict]:
    raw, starts = raw_time_groups(threads)
    if starts["selection"] is None or starts["test"] is None:
        return raw, {"starts": {k: v.isoformat() if v else None for k, v in starts.items()},
                     "insufficient_data": True}
    selection, selection_drop = drop_previous_boundary(raw["selection"], starts["test"], horizon_sec)
    train, train_drop = drop_previous_boundary(raw["train"], starts["selection"], horizon_sec)
    return {"train": train, "selection": selection, "test": raw["test"]}, {
        "starts": {k: v.isoformat() if v else None for k, v in starts.items()},
        "boundary_dropped": {"train": train_drop, "selection": selection_drop, "test": {"dropped_threads": 0, "emptied_groups": 0}},
    }


def longest_observable(threads: list[Thread]) -> tuple[float, bool]:
    known = [(thread.observed_until - thread.created_at).total_seconds()
             for thread in threads if thread.observed_until is not None]
    return min(168 * 3600.0, min(known)) if known else 168 * 3600.0, not bool(known)


def profile(threads: list[Thread], clean_by_thread: dict[str, list[Reply]], config: dict,
            *, next_start: datetime | None = None) -> dict:
    """Use only the supplied training portion; never inspect method performance."""
    L, unknown = longest_observable(threads)
    h_grid = [h for h in H_GRID_H if h * 3600 <= L]
    if not h_grid:
        return {"status": "insufficient_data", "reason": "no_observable_horizon", "L_seconds": L}
    analysis = [thread for thread in threads if
                (next_start is None or thread.created_at + timedelta(seconds=L) < next_start) and mature(thread, L)]
    if not analysis:
        return {"status": "insufficient_data", "reason": "empty_analysis_set", "L_seconds": L}
    all_replies = [reply for thread in analysis for reply in clean_by_thread[thread.thread_id]]
    denominator = sum(reply.offset_sec <= L for reply in all_replies)
    if denominator == 0:
        return {"status": "insufficient_data", "reason": "zero_horizon_denominator", "L_seconds": L}
    h_curve = {str(h): sum(reply.offset_sec <= h * 3600 for reply in all_replies) / denominator for h in h_grid}
    chosen_h = next((h for h in h_grid if h_curve[str(h)] >= .95), h_grid[-1])
    h_sec = chosen_h * 3600
    t_grid = [t for t in T_GRID_MIN if t <= config["T_max_min"]]
    if not t_grid:
        return {"status": "insufficient_data", "reason": "no_allowed_cutoff", "L_seconds": L}
    total_other = sum(reply.offset_sec <= h_sec and reply.offset_sec > 0 and
                      reply.author_id != thread.root_author_id
                      for thread in analysis for reply in clean_by_thread[thread.thread_id])
    t_curve = {}
    arrival = {str(h): h_curve[str(h)] for h in h_grid}
    for t in t_grid:
        seconds = t * 60
        selected = []
        remaining = 0
        covered = 0
        for thread in analysis:
            observed, _ = snapshot(thread, clean_by_thread[thread.thread_id], seconds)
            row = features(thread, observed, seconds)
            if candidate(row, config["min_replies"]):
                selected.append(thread)
                remaining += target(thread, clean_by_thread[thread.thread_id], seconds, h_sec)
                covered += sum(0 < reply.offset_sec <= h_sec and reply.author_id != thread.root_author_id
                               for reply in clean_by_thread[thread.thread_id])
        t_curve[str(t)] = {"candidate_threads": len(selected),
                           "candidate_coverage": covered / total_other if total_other else None,
                           "remaining_other_replies": remaining,
                           "remaining_share": remaining / total_other if total_other else None}
    if total_other == 0:
        return {"status": "insufficient_data", "reason": "zero_other_reply_denominator",
                "L_seconds": L, "analysis_threads": len(analysis), "H_hours": chosen_h, "arrival_curve": arrival}
    chosen_t = max(t_grid, key=lambda t: (t_curve[str(t)]["remaining_share"], -t))
    snapshots = [(thread, *snapshot(thread, clean_by_thread[thread.thread_id], chosen_t * 60))
                 for thread in analysis]
    zero_at_t = [thread for thread, observed, _ in snapshots if not observed]
    snapshot_cleaning = Counter()
    for _, _, counts in snapshots:
        snapshot_cleaning.update(counts)
    return {"status": "complete", "L_seconds": L, "unknown_followup": unknown,
            "analysis_threads": len(analysis), "H_hours": chosen_h, "T_minutes": chosen_t,
            "possibly_truncated_target": h_curve[str(chosen_h)] < .95,
            "arrival_curve": arrival, "T_curve": t_curve,
            "zero_reply_fraction": len(zero_at_t) / len(analysis),
            "zero_reply_future_other_share": sum(
                chosen_t * 60 < reply.offset_sec <= h_sec and reply.author_id != thread.root_author_id
                for thread in zero_at_t for reply in clean_by_thread[thread.thread_id]) / total_other,
            "total_other_within_H": total_other,
            "snapshot_cleaning_counts_at_T": dict(snapshot_cleaning)}
