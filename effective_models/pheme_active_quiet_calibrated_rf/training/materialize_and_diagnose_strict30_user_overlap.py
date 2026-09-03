"""Build and diagnose strict-past cross-thread PHEME user-overlap features.

Raw user identifiers are used transiently while scanning the local PHEME JSON
files.  Only thread-level aggregates are written.  Every history lookup is
strictly earlier than the target thread's source time + 30 minutes, and actions
from the target thread itself are excluded from cross-thread history.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from itertools import combinations
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
from data_pipeline.audit_rumdetect2017_timing import parse_created_at  # noqa: E402
from pheme_account_age_v5_common import DATASET, load_graphs  # noqa: E402
from pheme_quiet_expert_common import QUIET_QUANTILE, recency_values  # noqa: E402


RAW_ROOT = REPOSITORY_ROOT / "data" / "raw" / "pheme"
EXPERIMENT_ROOT = HERE.parent / "experiments"
HYBRID_RUN = EXPERIMENT_ROOT / "20260902_132700_413954_quiet_tier_hybrid_policy_full"
HYBRID_SCORES = HYBRID_RUN / "outer_scores.csv"

CUTOFF = timedelta(minutes=30)
RECENT_WINDOWS = {
    "30m": timedelta(minutes=30),
    "2h": timedelta(hours=2),
    "24h": timedelta(hours=24),
}
ORACLE_KS = (5, 10)
ADMISSION_CAPS = (1, 2, 3, 5, 10)
PRIMARY_K = 5
PRIMARY_CAP = 2
MIN_PRIMARY_HITS = 4
MIN_NONWORSE_EVENTS = 6
SEED = 42
HISTORY_SCOPE_MODE = "protected_strict30"
RUN_SUFFIX = "_strict30_cross_thread_user_overlap_diagnostic"
GLOBAL_SCAN_WORKERS = 16


@dataclass(frozen=True)
class Action:
    when: datetime
    user_id: str
    thread_id: str
    event_id: str
    is_source: bool
    is_retweet: bool


@dataclass
class RawThread:
    thread_id: str
    event_id: str
    category: str
    source_time: datetime
    actions: list[Action]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON is not an object: {path}")
    return value


def structure_nodes(tree: dict) -> tuple[list[str], str]:
    nodes: list[str] = []
    roots: list[str] = []

    def walk(branch: dict, parent: str | None = None) -> None:
        for raw_id, children in branch.items():
            node_id = str(raw_id)
            nodes.append(node_id)
            if parent is None:
                roots.append(node_id)
            if isinstance(children, dict):
                walk(children, node_id)

    walk(tree)
    if len(nodes) != len(set(nodes)) or len(roots) != 1:
        raise ValueError("invalid structure tree")
    return nodes, roots[0]


def tweet_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.glob("*.json") if not path.name.startswith("._"))


def scan_protected_snapshot_actions(
    graphs: list, status_path: Path
) -> tuple[dict[str, RawThread], list[Action], list[dict], dict]:
    """Read raw identity/time only for node IDs in protected strict-30 graphs."""
    targets: dict[str, RawThread] = {}
    all_actions: list[Action] = []
    exclusions: list[dict] = []
    audit = Counter()
    for index, graph in enumerate(graphs, 1):
        event_id = str(graph.event_id)
        category = "rumours"
        thread_id = str(graph.thread_id)
        thread_dir = RAW_ROOT / f"{event_id}-all-rnr-threads" / category / thread_id
        try:
            node_ids = [str(value) for value in graph.node_ids]
            source_ids = [
                node_id
                for node_id, include in zip(node_ids, graph.is_source_mask.tolist())
                if bool(include)
            ]
            if len(node_ids) != len(set(node_ids)) or len(source_ids) != 1:
                raise ValueError("invalid protected snapshot node/source identities")
            root_id = source_ids[0]
            source_path = thread_dir / "source-tweets" / f"{root_id}.json"
            source_tweet = load_json(source_path)
            if str(source_tweet.get("id")) != root_id:
                raise ValueError("raw source ID differs from protected source node")
            source_time = parse_created_at(source_tweet.get("created_at"))
            if source_time is None:
                raise ValueError("missing source timestamp")
            actions = []
            for node_id in node_ids:
                tweet_path = (
                    source_path
                    if node_id == root_id
                    else thread_dir / "reactions" / f"{node_id}.json"
                )
                if not tweet_path.exists():
                    raise ValueError("tree action missing raw tweet")
                tweet = load_json(tweet_path)
                if str(tweet.get("id")) != node_id:
                    raise ValueError("raw tweet ID differs from structure node")
                user = tweet.get("user")
                user_id = str((user or {}).get("id_str") or (user or {}).get("id") or "")
                when = parse_created_at(tweet.get("created_at"))
                if not user_id or when is None:
                    raise ValueError("tree action missing user or timestamp")
                if when < source_time or when > source_time + CUTOFF:
                    raise ValueError("protected snapshot action lies outside strict-30 interval")
                is_retweet = isinstance(tweet.get("retweeted_status"), dict)
                action = Action(when, user_id, thread_id, event_id, node_id == root_id, is_retweet)
                actions.append(action)
                audit["tree_actions"] += 1
                audit["retweet_actions"] += int(is_retweet)
                audit["quote_actions"] += int(bool(tweet.get("is_quote_status")))
            all_actions.extend(actions)
            audit["valid_threads"] += 1
            targets[thread_id] = RawThread(thread_id, event_id, category, source_time, actions)
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
            exclusions.append(
                {"thread_id": thread_id, "event_id": event_id, "category": category, "reason": str(error)}
            )
        if index % 100 == 0 or index == len(graphs):
            print(
                f"  protected snapshot identity scan {index}/{len(graphs)} "
                f"(valid={audit['valid_threads']}, official={len(targets)})",
                flush=True,
            )
            status_path.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "stage": "raw_scan",
                        "completed_threads": index,
                        "total_threads": len(graphs),
                        "valid_threads": audit["valid_threads"],
                        "official_threads_found": len(targets),
                        "updated_at": utc_now(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    audit["protected_snapshot_threads"] = len(graphs)
    audit["history_scope"] = "only actions whose node IDs occur in the 2402 protected strict-30 rumour snapshots"
    audit["excluded_threads"] = len(exclusions)
    return targets, sorted(all_actions, key=lambda item: item.when), exclusions, dict(audit)


def scan_global_raw_actions(
    graphs: list,
    status_path: Path,
    max_workers: int = GLOBAL_SCAN_WORKERS,
    directory_limit: int | None = None,
) -> tuple[dict[str, RawThread], list[Action], list[dict], dict]:
    """Read every valid raw tree action; retain protected nodes for targets."""
    official = {}
    for graph in graphs:
        node_ids = [str(value) for value in graph.node_ids]
        source_ids = [
            node_id
            for node_id, include in zip(node_ids, graph.is_source_mask.tolist())
            if bool(include)
        ]
        if len(source_ids) != 1:
            raise ValueError(f"{graph.thread_id}: invalid protected source identity")
        official[str(graph.thread_id)] = {
            "event_id": str(graph.event_id),
            "node_ids": set(node_ids),
            "source_id": source_ids[0],
        }
    directories = sorted(
        path
        for event_dir in RAW_ROOT.glob("*-all-rnr-threads")
        for category in ("rumours", "non-rumours")
        for path in (event_dir / category).glob("*")
        if path.is_dir()
    )
    if directory_limit is not None:
        if directory_limit <= 0:
            raise ValueError("directory_limit must be positive")
        directories = directories[:directory_limit]

    def scan_one(thread_dir: Path) -> tuple[RawThread | None, list[Action], Counter]:
        event_id = thread_dir.parent.parent.name.removesuffix("-all-rnr-threads")
        category = thread_dir.parent.name
        thread_id = thread_dir.name
        node_ids, root_id = structure_nodes(load_json(thread_dir / "structure.json"))
        source_path = thread_dir / "source-tweets" / f"{root_id}.json"
        source_tweet = load_json(source_path)
        if str(source_tweet.get("id")) != root_id:
            raise ValueError("raw source ID differs from structure root")
        source_time = parse_created_at(source_tweet.get("created_at"))
        if source_time is None:
            raise ValueError("missing source timestamp")
        actions = []
        audit = Counter()
        for node_id in node_ids:
            tweet_path = source_path if node_id == root_id else thread_dir / "reactions" / f"{node_id}.json"
            if not tweet_path.exists():
                raise ValueError("tree action missing raw tweet")
            tweet = load_json(tweet_path)
            if str(tweet.get("id")) != node_id:
                raise ValueError("raw tweet ID differs from structure node")
            user = tweet.get("user") or {}
            user_id = str(user.get("id_str") or user.get("id") or "")
            when = parse_created_at(tweet.get("created_at"))
            if not user_id or when is None or when < source_time:
                raise ValueError("tree action missing user/time or precedes source")
            is_retweet = isinstance(tweet.get("retweeted_status"), dict)
            actions.append(Action(when, user_id, thread_id, event_id, node_id == root_id, is_retweet))
            audit["tree_actions"] += 1
            audit["retweet_actions"] += int(is_retweet)
            audit["quote_actions"] += int(bool(tweet.get("is_quote_status")))
        target = None
        if thread_id in official:
            expected = official[thread_id]
            if category != "rumours" or event_id != expected["event_id"] or root_id != expected["source_id"]:
                raise ValueError("official thread raw metadata differs from protected graph")
            action_by_id = {node_id: action for node_id, action in zip(node_ids, actions)}
            if not expected["node_ids"].issubset(action_by_id):
                raise ValueError("protected node missing from full raw tree")
            protected_actions = [action_by_id[node_id] for node_id in expected["node_ids"]]
            if any(action.when > source_time + CUTOFF for action in protected_actions):
                raise ValueError("protected target node lies after strict-30 cutoff")
            target = RawThread(thread_id, event_id, category, source_time, protected_actions)
        return target, actions, audit

    targets: dict[str, RawThread] = {}
    all_actions: list[Action] = []
    exclusions: list[dict] = []
    audit = Counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(scan_one, path): path for path in directories}
        for index, future in enumerate(as_completed(futures), 1):
            thread_dir = futures[future]
            event_id = thread_dir.parent.parent.name.removesuffix("-all-rnr-threads")
            category = thread_dir.parent.name
            try:
                target, actions, local_audit = future.result()
                all_actions.extend(actions)
                audit.update(local_audit)
                audit["valid_threads"] += 1
                if target is not None:
                    targets[target.thread_id] = target
            except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
                exclusions.append(
                    {
                        "thread_id": thread_dir.name,
                        "event_id": event_id,
                        "category": category,
                        "reason": str(error),
                    }
                )
            if index % 100 == 0 or index == len(directories):
                print(
                    f"  global raw history scan {index}/{len(directories)} "
                    f"(valid={audit['valid_threads']}, official={len(targets)})",
                    flush=True,
                )
                status_path.write_text(
                    json.dumps(
                        {
                            "status": "running",
                            "stage": "global_raw_scan",
                            "completed_threads": index,
                            "total_threads": len(directories),
                            "valid_threads": audit["valid_threads"],
                            "official_threads_found": len(targets),
                            "workers": max_workers,
                            "updated_at": utc_now(),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
    audit["raw_thread_directories"] = len(directories)
    audit["excluded_threads"] = len(exclusions)
    audit["history_scope"] = (
        "all valid tree-referenced actions from rumour and non-rumour raw threads; "
        "category is not retained as an action feature"
    )
    return targets, sorted(all_actions, key=lambda item: item.when), exclusions, dict(audit)


def numeric_summary(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {f"{prefix}_{suffix}": 0.0 for suffix in ("mean", "max", "p90")}
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_max": float(array.max()),
        f"{prefix}_p90": float(np.quantile(array, 0.9)),
    }


def materialize_features(
    targets: dict[str, RawThread], all_actions: list[Action], status_path: Path
) -> pd.DataFrame:
    by_user: dict[str, list[Action]] = defaultdict(list)
    for action in all_actions:
        by_user[action.user_id].append(action)

    rows = []
    ordered_targets = sorted(targets.values(), key=lambda item: (item.source_time, item.thread_id))
    for index, thread in enumerate(ordered_targets, 1):
        decision_time = thread.source_time + CUTOFF
        observed = [action for action in thread.actions if action.when <= decision_time]
        if not observed or sum(action.is_source for action in observed) != 1:
            raise ValueError(f"{thread.thread_id}: invalid observed actions")
        users = sorted({action.user_id for action in observed})
        source_user = next(action.user_id for action in observed if action.is_source)
        reply_users = sorted({action.user_id for action in observed if not action.is_source})
        retweet_users = sorted({action.user_id for action in observed if action.is_retweet})

        histories: dict[str, list[Action]] = {}
        for user_id in users:
            histories[user_id] = [
                action
                for action in by_user[user_id]
                if action.when < decision_time and action.thread_id != thread.thread_id
            ]
        prior_actions = [len(histories[user_id]) for user_id in users]
        prior_threads = [{action.thread_id for action in histories[user_id]} for user_id in users]
        same_event_threads = [
            {action.thread_id for action in histories[user_id] if action.event_id == thread.event_id}
            for user_id in users
        ]

        prior_thread_users: dict[str, set[str]] = defaultdict(set)
        recent24_thread_users: dict[str, set[str]] = defaultdict(set)
        for user_id in users:
            for action in histories[user_id]:
                prior_thread_users[action.thread_id].add(user_id)
                if action.when >= decision_time - RECENT_WINDOWS["24h"]:
                    recent24_thread_users[action.thread_id].add(user_id)
        shared_pairs = set()
        for shared_users in prior_thread_users.values():
            if len(shared_users) >= 2:
                shared_pairs.update(combinations(sorted(shared_users), 2))
        possible_pairs = len(users) * (len(users) - 1) // 2

        repeat_counts = Counter(action.user_id for action in observed)
        feature = {
            "thread_id": thread.thread_id,
            "event_id": thread.event_id,
            "decision_time": decision_time.isoformat(),
            "observed_actions_raw": len(observed),
            "observed_unique_users_raw": len(users),
            "observed_repeat_author_fraction_raw": (
                sum(count - 1 for count in repeat_counts.values()) / len(observed)
            ),
            "user_overlap_any_prior_thread_fraction": float(np.mean([bool(value) for value in prior_threads])),
            "user_overlap_same_event_fraction": float(np.mean([bool(value) for value in same_event_threads])),
            "source_prior_other_actions": float(len(histories[source_user])),
            "source_prior_other_threads": float(len(prior_threads[users.index(source_user)])),
            "source_prior_same_event_threads": float(len(same_event_threads[users.index(source_user)])),
            "reply_user_overlap_any_fraction": float(
                np.mean([bool(histories[user_id]) for user_id in reply_users]) if reply_users else 0.0
            ),
            "coordination_prior_threads_shared_by_2plus_users": float(
                sum(len(value) >= 2 for value in prior_thread_users.values())
            ),
            "coordination_prior_thread_max_shared_users": float(
                max((len(value) for value in prior_thread_users.values()), default=0)
            ),
            "coordination_recent24h_thread_max_shared_users": float(
                max((len(value) for value in recent24_thread_users.values()), default=0)
            ),
            "coordination_prior_coparticipant_pair_fraction": (
                len(shared_pairs) / possible_pairs if possible_pairs else 0.0
            ),
            "observed_retweet_actions_raw": float(sum(action.is_retweet for action in observed)),
            "observed_unique_retweeters_raw": float(len(retweet_users)),
            "retweeter_prior_other_thread_fraction": float(
                np.mean([bool(histories[user_id]) for user_id in retweet_users]) if retweet_users else 0.0
            ),
        }
        feature.update(numeric_summary(prior_actions, "user_overlap_prior_actions"))
        feature.update(numeric_summary([len(value) for value in prior_threads], "user_overlap_prior_threads"))
        feature.update(
            numeric_summary([len(value) for value in same_event_threads], "user_overlap_same_event_threads")
        )
        for name, window in RECENT_WINDOWS.items():
            counts = [
                sum(action.when >= decision_time - window for action in histories[user_id])
                for user_id in users
            ]
            feature[f"user_overlap_recent_{name}_fraction"] = float(np.mean(np.asarray(counts) > 0))
            feature.update(numeric_summary(counts, f"user_overlap_recent_{name}_actions"))

        feature["user_overlap_strength"] = float(
            feature["user_overlap_any_prior_thread_fraction"]
            * np.log1p(sum(len(value) for value in prior_threads))
        )
        feature["same_event_overlap_strength"] = float(
            feature["user_overlap_same_event_fraction"]
            * np.log1p(sum(len(value) for value in same_event_threads))
        )
        feature["coordination_strength"] = float(
            np.log1p(feature["coordination_prior_threads_shared_by_2plus_users"])
            * (1.0 + feature["coordination_prior_coparticipant_pair_fraction"])
        )
        feature["recent_coordination_strength"] = float(
            feature["user_overlap_recent_2h_fraction"]
            * np.log1p(feature["coordination_recent24h_thread_max_shared_users"])
        )
        feature["retweeter_coordination_strength"] = float(
            feature["retweeter_prior_other_thread_fraction"]
            * np.log1p(feature["observed_unique_retweeters_raw"])
        )
        rows.append(feature)
        if index % 250 == 0 or index == len(ordered_targets):
            print(f"  user-overlap materialization {index}/{len(ordered_targets)}", flush=True)
            status_path.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "stage": "feature_materialization",
                        "completed_threads": index,
                        "total_threads": len(ordered_targets),
                        "updated_at": utc_now(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    frame = pd.DataFrame(rows)
    numeric = frame.drop(columns=["thread_id", "event_id", "decision_time"])
    if frame.thread_id.duplicated().any() or not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("invalid user-overlap feature artifact")
    return frame


def add_oracle_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for k in ORACLE_KS:
        result[f"oracle_top{k}"] = False
        for _, event in result.groupby("event_id", sort=True):
            chosen = event.sort_values(
                ["preventable_impact", "thread_id"], ascending=[False, True], kind="stable"
            ).head(k).index
            result.loc[chosen, f"oracle_top{k}"] = True
    return result


def selected_hits(frame: pd.DataFrame, score: str, cap: int, k: int) -> int:
    selected = frame.sort_values(
        [score, "thread_id"], ascending=[False, True], kind="stable"
    ).head(min(cap, len(frame)))
    return int(selected[f"oracle_top{k}"].sum())


def metric_rows(frame: pd.DataFrame, score: str, protocol: str) -> list[dict]:
    rows = []
    for k in ORACLE_KS:
        labels = frame[f"oracle_top{k}"].to_numpy(bool)
        positives = int(labels.sum())
        auc = float(roc_auc_score(labels, frame[score])) if 0 < positives < len(frame) else np.nan
        ap = float(average_precision_score(labels, frame[score])) if positives else np.nan
        for cap in ADMISSION_CAPS:
            hits = selected_hits(frame, score, cap, k)
            rows.append(
                {
                    "protocol": protocol,
                    "event_id": str(frame.event_id.iloc[0]),
                    "score_name": score,
                    "oracle_k": k,
                    "admission_cap": cap,
                    "quiet_candidates": len(frame),
                    "quiet_oracle_positives": positives,
                    "hits": hits,
                    "precision": hits / min(cap, len(frame)) if len(frame) else 0.0,
                    "recall": hits / positives if positives else np.nan,
                    "roc_auc": auc,
                    "average_precision": ap,
                }
            )
    return rows


def event_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    event_count = frame.event_id.nunique()
    counts = frame.groupby("event_id").thread_id.transform("count")
    return (len(frame) / (event_count * counts)).to_numpy(float)


def main() -> None:
    output = EXPERIMENT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + RUN_SUFFIX
    )
    output.mkdir(parents=True, exist_ok=False)
    status_path = output / "build_status.json"
    record = {
        "status": "running",
        "started_at": utc_now(),
        "purpose": "materialize and diagnose strict-past cross-thread user overlap and coordination",
        "raw_input": str(RAW_ROOT.resolve()),
        "protected_graphs": str(DATASET.resolve()),
        "baseline_scores": str(HYBRID_SCORES.resolve()),
        "split": "seven eligible outer LOEO events; fixed logistic combination plus inner-LOEO single-score selection",
        "cutoff_seconds": 1800,
        "seed": SEED,
        "history_scope_mode": HISTORY_SCOPE_MODE,
        "admission_caps": list(ADMISSION_CAPS),
        "oracle_ks": list(ORACLE_KS),
        "privacy": "Raw user IDs are used only in memory; outputs contain thread-level aggregates and no user IDs.",
        "research_safety": (
            "For target decision D=source+30m, cross-thread history includes only actions with created_at<D and excludes the target thread. "
            "No future action, retweet_count, favorite_count, outcome, label, or outer-test label constructs a feature."
        ),
    }
    write_record(output, record)
    status_path.write_text(json.dumps({"status": "running", "stage": "initializing", "started_at": record["started_at"]}, indent=2), encoding="utf-8")
    try:
        print("Starting strict-30 cross-thread user-overlap materialization and diagnostic.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/7] Loading official strict-30 identities and saved hybrid scores...", flush=True)
        graphs, _, eligible = load_graphs()
        graph_ids = [str(graph.thread_id) for graph in graphs]
        if len(graph_ids) != len(set(graph_ids)):
            raise ValueError("duplicate protected graph thread IDs")
        saved = pd.read_csv(HYBRID_SCORES, dtype={"thread_id": str, "event_id": str})
        required = {"thread_id", "event_id", "preventable_impact", "hybrid_calibrated_score", "quiet"}
        if not required.issubset(saved) or sorted(saved.event_id.unique()) != sorted(eligible):
            raise ValueError("invalid saved hybrid score artifact")

        print("[2/7] Scanning raw PHEME user identities and absolute action times...", flush=True)
        if HISTORY_SCOPE_MODE == "protected_strict30":
            targets, all_actions, exclusions, raw_audit = scan_protected_snapshot_actions(graphs, status_path)
        elif HISTORY_SCOPE_MODE == "global_raw_strict_past":
            targets, all_actions, exclusions, raw_audit = scan_global_raw_actions(graphs, status_path)
            missing_ids = set(graph_ids) - set(targets)
            if missing_ids:
                print(
                    f"  recovering {len(missing_ids)} official targets from their validated protected strict-30 nodes...",
                    flush=True,
                )
                missing_graphs = [graph for graph in graphs if str(graph.thread_id) in missing_ids]
                fallback_targets, fallback_actions, fallback_exclusions, _ = scan_protected_snapshot_actions(
                    missing_graphs, status_path
                )
                if fallback_exclusions or set(fallback_targets) != missing_ids:
                    raise ValueError("protected fallback did not recover every incomplete full-raw target")
                targets.update(fallback_targets)
                all_actions = sorted([*all_actions, *fallback_actions], key=lambda item: item.when)
                raw_audit["official_targets_recovered_from_protected_nodes"] = len(fallback_targets)
        else:
            raise ValueError(f"unknown history scope mode: {HISTORY_SCOPE_MODE}")
        pd.DataFrame(exclusions).to_csv(output / "raw_thread_exclusions.csv", index=False)
        missing = sorted(set(graph_ids) - set(targets))
        if missing:
            raise ValueError(f"raw user data missing for {len(missing)} protected threads; first={missing[:5]}")

        print("[3/7] Building strict-past cross-thread overlap and coordination aggregates...", flush=True)
        features = materialize_features(targets, all_actions, status_path)
        features.to_csv(output / "strict30_user_overlap_features.csv", index=False)
        pd.DataFrame(exclusions).to_csv(output / "raw_thread_exclusions.csv", index=False)

        print("[4/7] Joining protected outcomes and auditing feature coverage...", flush=True)
        graph_frame = pd.DataFrame(
            {
                "thread_id": graph_ids,
                "event_id_graph": [str(graph.event_id) for graph in graphs],
                "preventable_impact": [float(graph.preventable_impact.item()) for graph in graphs],
                "recency": recency_values(graphs),
            }
        )
        frame = graph_frame.merge(features, on="thread_id", how="inner", validate="one_to_one")
        if len(frame) != len(graphs) or not frame.event_id.eq(frame.event_id_graph).all():
            raise ValueError("user-overlap join changed protected identities or events")
        frame.drop(columns="event_id_graph", inplace=True)
        frame = add_oracle_labels(frame)
        frame = frame.merge(
            saved[["thread_id", "quiet", "hybrid_calibrated_score"]],
            on="thread_id", how="left", validate="one_to_one",
        )
        if frame.loc[frame.event_id.isin(eligible), ["quiet", "hybrid_calibrated_score"]].isna().any().any():
            raise ValueError("eligible saved-score alignment is incomplete")

        metadata = {"thread_id", "event_id", "decision_time", "preventable_impact", "recency", "quiet", "hybrid_calibrated_score", "oracle_top5", "oracle_top10"}
        feature_columns = [column for column in frame.columns if column not in metadata]
        model_feature_columns = [
            column
            for column in feature_columns
            if column.startswith(
                (
                    "user_overlap_",
                    "source_prior_",
                    "reply_user_overlap_",
                    "coordination_",
                    "retweeter_",
                    "same_event_overlap_",
                    "recent_coordination_",
                    "observed_unique_retweeters_",
                )
            )
        ]
        score_columns = [
            "user_overlap_strength",
            "same_event_overlap_strength",
            "coordination_strength",
            "recent_coordination_strength",
            "retweeter_coordination_strength",
            "user_overlap_any_prior_thread_fraction",
            "user_overlap_recent_2h_fraction",
            "coordination_prior_coparticipant_pair_fraction",
        ]
        coverage = {
            "protected_threads": len(frame),
            "threads_with_any_prior_user_overlap": int((frame.user_overlap_any_prior_thread_fraction > 0).sum()),
            "threads_with_same_event_user_overlap": int((frame.user_overlap_same_event_fraction > 0).sum()),
            "threads_with_prior_coparticipant_pair": int((frame.coordination_prior_coparticipant_pair_fraction > 0).sum()),
            "threads_with_observed_retweet_identity": int((frame.observed_unique_retweeters_raw > 0).sum()),
            "raw_retweet_tree_actions": int(raw_audit.get("retweet_actions", 0)),
        }

        print("[5/7] Running fixed outer-LOEO feature-combination diagnostic...", flush=True)
        predictions = []
        selection_rows = []
        for outer_number, outer_event in enumerate(sorted(eligible), 1):
            print(f"  outer fold {outer_number}/{len(eligible)}: {outer_event}", flush=True)
            outer_train = frame.loc[~frame.event_id.eq(outer_event)].copy()
            threshold = float(np.quantile(outer_train.recency, QUIET_QUANTILE))
            train = outer_train.loc[outer_train.recency >= threshold].copy()
            test = frame.loc[frame.event_id.eq(outer_event) & (frame.recency >= threshold)].copy()
            official = frame.loc[frame.event_id.eq(outer_event), "quiet"].astype(bool).to_numpy()
            recomputed = (frame.loc[frame.event_id.eq(outer_event), "recency"] >= threshold).to_numpy()
            if not np.array_equal(official, recomputed):
                raise ValueError(f"{outer_event}: quiet gate differs from official hybrid")

            usable = [column for column in model_feature_columns if float(train[column].std()) > 0]
            scaler = StandardScaler().fit(train[usable])
            classifier = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED
            )
            classifier.fit(
                scaler.transform(train[usable]),
                train.oracle_top5.astype(int),
                sample_weight=event_balanced_weights(train),
            )
            test["user_overlap_logistic_score"] = classifier.predict_proba(scaler.transform(test[usable]))[:, 1]

            inner_summary = []
            for score in score_columns:
                row = {"outer_event": outer_event, "score_name": score}
                for k in ORACLE_KS:
                    for cap in ADMISSION_CAPS:
                        hits = 0
                        for inner_event in sorted(event for event in eligible if event != outer_event):
                            inner_reference = outer_train.loc[~outer_train.event_id.eq(inner_event)]
                            inner_threshold = float(np.quantile(inner_reference.recency, QUIET_QUANTILE))
                            validation = outer_train.loc[
                                outer_train.event_id.eq(inner_event)
                                & (outer_train.recency >= inner_threshold)
                            ]
                            hits += selected_hits(validation, score, cap, k)
                        row[f"top{k}_hits_at_cap{cap}"] = int(hits)
                inner_summary.append(row)
            selection = pd.DataFrame(inner_summary).sort_values(
                ["top5_hits_at_cap2", "top5_hits_at_cap1", "top10_hits_at_cap2", "score_name"],
                ascending=[False, False, False, True], kind="stable",
            )
            selected_score = str(selection.iloc[0].score_name)
            selection["selected"] = selection.score_name.eq(selected_score)
            selection_rows.append(selection)
            test["selected_direct_score_name"] = selected_score
            test["selected_direct_score"] = test[selected_score]
            predictions.append(test)

        prediction_frame = pd.concat(predictions, ignore_index=True)
        print("[6/7] Evaluating minimal-admission signal curves...", flush=True)
        rows = []
        for _, event in prediction_frame.groupby("event_id", sort=True):
            rows.extend(metric_rows(event, "user_overlap_logistic_score", "fixed_user_overlap_logistic"))
            rows.extend(metric_rows(event, "selected_direct_score", "nested_direct_user_overlap"))
            rows.extend(metric_rows(event, "hybrid_calibrated_score", "saved_hybrid_quiet_score"))
        metrics = pd.DataFrame(rows)
        pooled = metrics.groupby(["protocol", "oracle_k", "admission_cap"], as_index=False).agg(
            eligible_events=("event_id", "nunique"),
            total_quiet_candidates=("quiet_candidates", "sum"),
            total_quiet_oracle_positives=("quiet_oracle_positives", "sum"),
            total_hits=("hits", "sum"),
            mean_event_precision=("precision", "mean"),
            mean_event_recall=("recall", "mean"),
            mean_event_auc=("roc_auc", "mean"),
            mean_event_ap=("average_precision", "mean"),
        )
        primary_protocol = "fixed_user_overlap_logistic"
        primary = metrics.loc[
            metrics.protocol.eq(primary_protocol)
            & metrics.oracle_k.eq(PRIMARY_K)
            & metrics.admission_cap.eq(PRIMARY_CAP)
        ].set_index("event_id")
        comparator = metrics.loc[
            metrics.protocol.eq("saved_hybrid_quiet_score")
            & metrics.oracle_k.eq(PRIMARY_K)
            & metrics.admission_cap.eq(PRIMARY_CAP)
        ].set_index("event_id")
        hits = int(primary.hits.sum())
        nonworse = int((primary.hits >= comparator.hits).sum())
        decision = {
            "passed": bool(hits >= MIN_PRIMARY_HITS and nonworse >= MIN_NONWORSE_EVENTS),
            "primary_protocol": primary_protocol,
            "primary_top5_hits_at_cap2": hits,
            "events_nonworse_than_saved_hybrid_quiet_score": nonworse,
            "required_hits": MIN_PRIMARY_HITS,
            "required_nonworse_events": MIN_NONWORSE_EVENTS,
        }
        decision["interpretation"] = (
            "Proceed to a separately validated hybrid admission experiment."
            if decision["passed"]
            else "Do not integrate the user-overlap channel into the hybrid policy."
        )

        print("[7/7] Writing saved evidence and final decision...", flush=True)
        prediction_frame.to_csv(output / "outer_quiet_user_overlap_predictions.csv", index=False)
        pd.concat(selection_rows, ignore_index=True).to_csv(output / "inner_direct_score_selection.csv", index=False)
        metrics.to_csv(output / "per_event_admission_metrics.csv", index=False)
        pooled.to_csv(output / "pooled_admission_metrics.csv", index=False)
        (output / "feature_schema.json").write_text(
            json.dumps(
                {
                    "metadata_columns": ["thread_id", "event_id", "decision_time"],
                    "feature_columns": feature_columns,
                    "model_feature_columns": model_feature_columns,
                    "target_columns_not_model_inputs": ["preventable_impact", "oracle_top5", "oracle_top10"],
                    "raw_user_ids_exported": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        status_path.write_text(json.dumps({"status": "complete", "completed_at": utc_now()}, indent=2), encoding="utf-8")
        record.update(
            {
                "status": "complete",
                "completed_at": utc_now(),
                "eligible_events": sorted(eligible),
                "raw_audit": raw_audit,
                "coverage": coverage,
                "decision": decision,
                "metrics": pooled.to_dict(orient="records"),
                "output_files": [
                    "strict30_user_overlap_features.csv",
                    "raw_thread_exclusions.csv",
                    "outer_quiet_user_overlap_predictions.csv",
                    "inner_direct_score_selection.csv",
                    "per_event_admission_metrics.csv",
                    "pooled_admission_metrics.csv",
                    "feature_schema.json",
                    "decision.json",
                    "build_status.json",
                    "run_record.json",
                ],
            }
        )
        write_record(output, record)
        print(
            f"  coverage: overlap={coverage['threads_with_any_prior_user_overlap']}/{len(frame)}, "
            f"co-participant={coverage['threads_with_prior_coparticipant_pair']}/{len(frame)}, "
            f"retweet-identity={coverage['threads_with_observed_retweet_identity']}/{len(frame)}",
            flush=True,
        )
        print(
            f"  decision: {'PASS' if decision['passed'] else 'FAIL'} "
            f"({hits} quiet Oracle Top-5 hits at cap=2; {nonworse}/7 events non-worse)",
            flush=True,
        )
        print(f"SUCCESS: strict-30 cross-thread user-overlap diagnostic saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        status_path.write_text(
            json.dumps({"status": "failed", "failed_at": utc_now(), "error": f"{type(error).__name__}: {error}"}, indent=2),
            encoding="utf-8",
        )
        record.update(
            {
                "status": "failed",
                "failed_at": utc_now(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "output_files": sorted(path.name for path in output.iterdir()),
            }
        )
        write_record(output, record)
        print(f"FAILURE: strict-30 cross-thread user-overlap diagnostic preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
