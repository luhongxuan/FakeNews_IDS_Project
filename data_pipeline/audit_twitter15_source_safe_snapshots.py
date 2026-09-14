"""Build a read-only source-anchored 30-minute Twitter15 snapshot audit.

This is not an intervention dataset builder: RumDetect tree edges fail
time-order validation, so this audit uses only timestamp-safe node sets and a
future-node-growth outcome.  It tests whether early activity/recency signals
observed in PHEME also occur in an independent Twitter corpus.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from data_pipeline.audit_rumdetect2017_timing import parse_tree, snowflake_time  # noqa: E402


CUTOFF_MINUTES = 30.0
TOP_K = 10
MATCH_FEATURES = ("log1p_observed_nodes", "late_activity_frac")
COMPARE_FEATURES = (
    "seconds_since_last_activity", "log1p_count_0_10m", "log1p_count_10_20m",
    "log1p_count_20_30m", "observed_reply_text_coverage",
)
SEED = 42
BOOTSTRAP_DRAWS = 10_000


def read_labels(path: Path) -> dict[str, str]:
    result = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        label, separator, source_id = raw.strip().partition(":")
        if not separator or not source_id.isdigit() or source_id in result:
            raise ValueError(f"Invalid label row at {path}:{line_number}")
        result[source_id] = label
    return result


def best_hydration_text(path: Path) -> set[str]:
    """Return IDs with usable static text, never using mutable engagement data."""
    usable: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(row, dict)
                and row.get("status") == "ok"
                and isinstance(row.get("tweet_id"), str)
                and isinstance(row.get("text"), str)
                and row["text"].strip()
            ):
                usable.add(row["tweet_id"])
    return usable


def snapshot_rows(dataset_dir: Path) -> tuple[pd.DataFrame, dict]:
    labels = read_labels(dataset_dir / "label.txt")
    text_ids = best_hydration_text(dataset_dir / "hydrated_tweets.jsonl")
    rows, excluded_pre_source = [], []
    for source_id, label in sorted(labels.items()):
        source_time = snowflake_time(source_id)
        if source_time is None:
            raise ValueError(f"Invalid source Snowflake ID: {source_id}")
        delays, _, malformed = parse_tree(dataset_dir / "tree" / f"{source_id}.txt")
        if malformed:
            raise ValueError(f"Malformed tree rows in {source_id}")
        node_offsets = {}
        for node_id in delays:
            node_time = snowflake_time(node_id)
            if node_time is None:
                raise ValueError(f"Invalid node Snowflake ID in {source_id}: {node_id}")
            node_offsets[node_id] = (node_time - source_time).total_seconds() / 60.0
        if any(value < 0 for value in node_offsets.values()):
            excluded_pre_source.append(source_id)
            continue
        observed_ids = [node_id for node_id, offset in node_offsets.items() if offset <= CUTOFF_MINUTES]
        future_ids = [node_id for node_id, offset in node_offsets.items() if offset > CUTOFF_MINUTES]
        if source_id not in observed_ids:
            raise ValueError(f"Source missing from its own snapshot: {source_id}")
        reply_offsets = np.asarray([node_offsets[node_id] for node_id in observed_ids if node_id != source_id], dtype=float)
        latest = float(max(node_offsets[node_id] for node_id in observed_ids))
        observed_reply_text = sum(node_id in text_ids for node_id in observed_ids if node_id != source_id)
        rows.append({
            "thread_id": source_id,
            "label": label,
            "observed_nodes": len(observed_ids),
            "observed_replies": len(reply_offsets),
            "future_nodes": len(future_ids),
            "target_log1p_future_nodes": float(np.log1p(len(future_ids))),
            "log1p_observed_nodes": float(np.log1p(len(observed_ids))),
            "log1p_count_0_10m": float(np.log1p(np.sum(reply_offsets <= 10))),
            "log1p_count_10_20m": float(np.log1p(np.sum((reply_offsets > 10) & (reply_offsets <= 20)))),
            "log1p_count_20_30m": float(np.log1p(np.sum((reply_offsets > 20) & (reply_offsets <= 30)))),
            "late_activity_frac": float(np.mean(reply_offsets > 20)) if len(reply_offsets) else 0.0,
            "seconds_since_last_activity": float((CUTOFF_MINUTES - latest) * 60.0),
            "usable_hydrated_reply_texts": observed_reply_text,
            "observed_reply_text_coverage": float(observed_reply_text / len(reply_offsets)) if len(reply_offsets) else 0.0,
        })
    frame = pd.DataFrame(rows)
    return frame, {
        "source_trees_total": len(labels),
        "source_anchor_temporal_valid_trees": len(frame),
        "pre_source_tree_ids_excluded": excluded_pre_source,
        "pre_source_trees_excluded": len(excluded_pre_source),
    }


def spearman_table(frame: pd.DataFrame) -> pd.DataFrame:
    features = ["log1p_observed_nodes", "late_activity_frac", "seconds_since_last_activity", *COMPARE_FEATURES[1:]]
    rows = []
    for group_name, group in [("all_labels", frame), *frame.groupby("label", sort=True)]:
        for feature in dict.fromkeys(features):
            value = group[[feature, "target_log1p_future_nodes"]].corr(method="spearman").iloc[0, 1]
            rows.append({"group": group_name, "n_threads": len(group), "feature": feature, "spearman_with_log1p_future_nodes": float(value)})
    return pd.DataFrame(rows)


def matched_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Within label, match future-growth top-k threads to early-signal controls."""
    pairs = []
    for label, group in frame.groupby("label", sort=True):
        oracle = group.nlargest(TOP_K, "future_nodes").copy()
        available = group.loc[~group.thread_id.isin(oracle.thread_id)].copy()
        scale = group.loc[:, MATCH_FEATURES].std(ddof=0).replace(0, 1.0)
        for oracle_row in oracle.itertuples(index=False):
            oracle_values = pd.Series(oracle_row, index=oracle.columns)
            normalized = (
                available.loc[:, MATCH_FEATURES].to_numpy(dtype=float)
                - oracle_values.loc[list(MATCH_FEATURES)].to_numpy(dtype=float)
            ) / scale.to_numpy(dtype=float)
            distance = pd.Series(np.sqrt(np.square(normalized).sum(axis=1)), index=available.index)
            control = available.loc[distance.idxmin()]
            row = {
                "label": label, "match_distance": float(distance.loc[control.name]),
                "oracle_thread_id": oracle_values.thread_id, "control_thread_id": control.thread_id,
                "oracle_future_nodes": int(oracle_values.future_nodes), "control_future_nodes": int(control.future_nodes),
            }
            for feature in COMPARE_FEATURES:
                event_std = float(group[feature].std(ddof=0))
                delta = float(oracle_values[feature] - control[feature])
                row[f"standardized_delta_{feature}"] = delta / event_std if event_std else 0.0
            pairs.append(row)
            available = available.drop(index=control.name)
    return pd.DataFrame(pairs)


def bootstrap_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for feature in COMPARE_FEATURES:
        by_label = pairs.groupby("label", sort=True)[f"standardized_delta_{feature}"].mean().to_numpy(dtype=float)
        draws = rng.choice(by_label, size=(BOOTSTRAP_DRAWS, len(by_label)), replace=True).mean(axis=1)
        rows.append({
            "feature": feature,
            "mean_label_standardized_oracle_minus_control": float(by_label.mean()),
            "bootstrap_ci_low": float(np.quantile(draws, 0.025)),
            "bootstrap_ci_high": float(np.quantile(draws, 0.975)),
            "labels": len(by_label),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Twitter15 source-safe 30-minute snapshot and proxy validation audit")
    parser.add_argument("--dataset", type=Path, default=Path("data/raw/rumdetect2017/twitter15"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite report: {args.output}")
    frame, safety = snapshot_rows(args.dataset.resolve())
    correlations = spearman_table(frame)
    pairs = matched_pairs(frame)
    pair_summary = bootstrap_summary(pairs)
    args.output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output / "source_safe_snapshot_records.csv", index=False)
    correlations.to_csv(args.output / "future_growth_spearman.csv", index=False)
    pairs.to_csv(args.output / "matched_future_growth_pairs.csv", index=False)
    pair_summary.to_csv(args.output / "matched_future_growth_summary.csv", index=False)
    metadata = {
        "dataset": str(args.dataset.resolve()),
        "cutoff_minutes": CUTOFF_MINUTES,
        "outcome": "log1p(number of source-anchored, timestamp-safe nodes after 30 minutes)",
        "scope": "External proxy validation of early activity/recency signals; not a preventable-impact or causal-intervention evaluation because original tree edges are not time-respecting.",
        "matching": {"within_label": True, "features": MATCH_FEATURES, "top_k_per_label": TOP_K},
        "safety": {
            **safety,
            "node_time_source": "Twitter Snowflake timestamp",
            "edge_policy": "No tree edge is used for features or labels.",
            "hydration_policy": "Only static reply text availability is summarized; no mutable engagement count is read.",
        },
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Created source-safe audit for {len(frame)} temporal-valid Twitter15 trees: {args.output}")


if __name__ == "__main__":
    main()
