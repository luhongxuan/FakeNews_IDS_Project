"""Strict-60 quiet-thread tier diagnostic.

This is deliberately a separate experiment from the strict-30 PHEME policy.
It asks whether 60-minute snapshot-only inputs can distinguish coarse future
impact tiers among threads that are quiet at 60 minutes.  It never uses profile
or engagement fields from ``graph.x``; only temporal summaries, observed node
count, and observed depth are read.

Set RUN_MODE to ``full`` only after the smoke output and data audit have been
reviewed.  Full mode remains a foreground, user-run experiment.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import torch


RUN_MODE = "smoke"  # "smoke" is safe for Codex; "full" is user-run only.
SMOKE_OUTER_EVENT = "charliehebdo"
SMOKE_MAX_TRAIN_PER_EVENT = 80
SMOKE_MAX_TEST_THREADS = 80
CUTOFF_SECONDS = 3600.0
QUIET_QUANTILE = 0.60
HIGH_FRACTION = 0.20
MIDDLE_CUMULATIVE_FRACTION = 0.50
SEED = 42
SMOKE_TREES = 25
FULL_TREES = 300

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
LEGACY60 = ROOT / "research_scratch" / "legacy_full" / "graphsage_intervention_6"
DATASET = LEGACY60 / "pheme_graphs_roberta_60min_replyv4_preventableimpact_raw.pt"
REPLY_OFFSETS = ROOT / "research_scratch" / "legacy_full" / "graphsage_intervention_4" / "pheme_reply_level_v4.csv"
OUTPUT_ROOT = BASE / "experiments"

TEMPORAL_NAMES = (
    "log1p_count_0_10m", "log1p_count_10_20m", "log1p_count_20_30m",
    "log1p_count_30_40m", "log1p_count_40_50m", "log1p_count_50_60m",
    "log1p_count_recent_5m", "log1p_count_recent_10m",
    "log1p_seconds_since_last_activity", "log1p_median_interarrival_sec",
    "log1p_std_interarrival_sec", "observed_leaf_fraction", "max_observed_children",
)
FEATURE_NAMES = TEMPORAL_NAMES + ("log1p_observed_nodes", "mean_observed_depth", "max_observed_depth")
RECENCY_INDEX = TEMPORAL_NAMES.index("log1p_seconds_since_last_activity")
TIER_NAMES = ("high", "middle", "low")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def load_offsets() -> dict[tuple[str, str], float]:
    frame = pd.read_csv(
        REPLY_OFFSETS,
        usecols=["thread_id", "tweet_id", "offset_sec"],
        dtype={"thread_id": str, "tweet_id": str},
        low_memory=False,
    )
    if frame.duplicated(["thread_id", "tweet_id"]).any():
        raise ValueError("Reply offsets are not unique by (thread_id, tweet_id)")
    values = frame.offset_sec.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Reply offsets are missing, non-finite, or negative")
    return {(row.thread_id, row.tweet_id): float(row.offset_sec) for row in frame.itertuples(index=False)}


def audit_graphs(graphs: list, offsets: dict[tuple[str, str], float]) -> None:
    if len(graphs) != 2402:
        raise ValueError(f"Expected 2,402 rumour graphs, got {len(graphs)}")
    if len({str(graph.thread_id) for graph in graphs}) != len(graphs):
        raise ValueError("Graph artifact contains duplicate thread IDs")
    events = {str(graph.event_id) for graph in graphs}
    if len(events) != 9:
        raise ValueError(f"Expected 9 events, got {len(events)}")
    for index, graph in enumerate(graphs, 1):
        if graph.temporal_features.numel() != len(TEMPORAL_NAMES):
            raise ValueError(f"Unexpected temporal feature length for {graph.thread_id}")
        if not torch.isfinite(graph.temporal_features).all() or not torch.isfinite(graph.x[:, 0]).all():
            raise ValueError(f"Non-finite permitted input for {graph.thread_id}")
        if graph.edge_index.numel() and (int(graph.edge_index.min()) < 0 or int(graph.edge_index.max()) >= graph.num_nodes):
            raise ValueError(f"Snapshot edge outside observable node IDs for {graph.thread_id}")
        if not torch.allclose(graph.preventable_y, torch.log1p(graph.preventable_impact)):
            raise ValueError(f"Target mismatch for {graph.thread_id}")
        for tweet_id in graph.node_ids:
            key = (str(graph.thread_id), str(tweet_id))
            if key not in offsets:
                raise KeyError(f"Missing offset for snapshot node {key}")
            if offsets[key] > CUTOFF_SECONDS + 1e-9:
                raise ValueError(f"Future node found in 60-minute snapshot {key}: {offsets[key]}")
        if index % 500 == 0 or index == len(graphs):
            print(f"  audit snapshot graphs: {index}/{len(graphs)}", flush=True)


def frame_from_graphs(graphs: list) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for graph in graphs:
        temporal = graph.temporal_features.detach().cpu().numpy().reshape(-1).astype(float)
        depths = graph.x[:, 0].detach().cpu().numpy().reshape(-1).astype(float)
        row: dict[str, object] = {
            "event_id": str(graph.event_id),
            "thread_id": str(graph.thread_id),
            "preventable_impact": float(graph.preventable_impact.item()),
            "recency": float(temporal[RECENCY_INDEX]),
        }
        row.update(dict(zip(TEMPORAL_NAMES, temporal)))
        row.update({
            "log1p_observed_nodes": float(math.log1p(graph.num_nodes)),
            "mean_observed_depth": float(depths.mean()),
            "max_observed_depth": float(depths.max()),
        })
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not np.isfinite(frame[list(FEATURE_NAMES)].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite diagnostic feature")
    return frame


def assign_tiers(quiet: pd.DataFrame) -> pd.DataFrame:
    """Outcome-only labels, assigned separately inside each train/test event."""
    parts: list[pd.DataFrame] = []
    for _, pool in quiet.groupby("event_id", sort=True):
        ranked = pool.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable").copy()
        high_end = max(1, math.ceil(len(ranked) * HIGH_FRACTION))
        middle_end = max(high_end + 1, math.ceil(len(ranked) * MIDDLE_CUMULATIVE_FRACTION))
        ranked["tier"] = "low"
        ranked.iloc[:high_end, ranked.columns.get_loc("tier")] = "high"
        ranked.iloc[high_end:middle_end, ranked.columns.get_loc("tier")] = "middle"
        parts.append(ranked)
    return pd.concat(parts, ignore_index=True)


def deterministic_smoke_sample(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    return frame.sort_values("thread_id", kind="stable").head(maximum).copy()


def prepare_fold(frame: pd.DataFrame, outer_event: str, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    raw_train = frame[frame.event_id != outer_event].copy()
    raw_test = frame[frame.event_id == outer_event].copy()
    threshold = float(np.quantile(raw_train.recency.to_numpy(dtype=float), QUIET_QUANTILE))
    train_quiet = raw_train[raw_train.recency >= threshold].copy()
    test_quiet = raw_test[raw_test.recency >= threshold].copy()
    if smoke:
        train_quiet = pd.concat(
            [deterministic_smoke_sample(part, SMOKE_MAX_TRAIN_PER_EVENT) for _, part in train_quiet.groupby("event_id", sort=True)],
            ignore_index=True,
        )
        # Keep the complete held-out event: reducing it would redefine the
        # event-relative high/middle/low outcome tiers being diagnosed.
    if len(train_quiet) == 0 or len(test_quiet) == 0:
        raise ValueError("Quiet gate produced an empty train or held-out cohort")
    return train_quiet, test_quiet, threshold


def evaluate_fold(train_quiet: pd.DataFrame, test_quiet: pd.DataFrame, threshold: float, trees: int) -> tuple[dict[str, object], pd.DataFrame]:
    # The held-out outcome is intentionally not read until after probabilities are produced.
    train_labeled = assign_tiers(train_quiet)
    if set(train_labeled.tier) != set(TIER_NAMES):
        raise ValueError("Training smoke sample does not contain all three tiers")
    model = RandomForestClassifier(
        n_estimators=trees, max_depth=8, min_samples_leaf=3, max_features=0.8,
        class_weight="balanced", n_jobs=1, random_state=SEED,
    )
    model.fit(train_labeled[list(FEATURE_NAMES)], train_labeled.tier)
    probabilities = model.predict_proba(test_quiet[list(FEATURE_NAMES)])
    class_index = {str(name): index for index, name in enumerate(model.classes_)}
    if set(class_index) != set(TIER_NAMES):
        raise ValueError("Classifier unexpectedly omitted a tier")

    evaluated = assign_tiers(test_quiet)
    predicted = model.classes_[np.argmax(probabilities, axis=1)]
    evaluated["predicted_tier"] = predicted
    evaluated["high_probability"] = probabilities[:, class_index["high"]]
    truth_high = evaluated.tier.eq("high")
    pred_high = evaluated.predicted_tier.eq("high")
    tp = int((truth_high & pred_high).sum())
    precision = tp / int(pred_high.sum()) if pred_high.any() else 0.0
    recall = tp / int(truth_high.sum()) if truth_high.any() else 0.0
    baseline_impact = float(evaluated.preventable_impact.mean())
    selected_impact = float(evaluated.loc[pred_high, "preventable_impact"].mean()) if pred_high.any() else 0.0
    return {
        "outer_event": str(evaluated.event_id.iloc[0]),
        "quiet_threshold": threshold,
        "train_quiet_threads": int(len(train_labeled)),
        "test_quiet_threads": int(len(evaluated)),
        "test_high_threads": int(truth_high.sum()),
        "predicted_high_threads": int(pred_high.sum()),
        "high_precision": precision,
        "high_recall": recall,
        "selected_mean_impact": selected_impact,
        "quiet_pool_mean_impact": baseline_impact,
        "selected_to_pool_impact_ratio": selected_impact / baseline_impact if baseline_impact else 0.0,
    }, evaluated


def main() -> None:
    if RUN_MODE not in {"smoke", "full"}:
        raise ValueError("RUN_MODE must be 'smoke' or 'full'")
    smoke = RUN_MODE == "smoke"
    output = OUTPUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_quiet_60min_tier_{RUN_MODE}")
    output.mkdir(parents=True, exist_ok=False)
    record: dict[str, object] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": RUN_MODE,
        "dataset": str(DATASET),
        "reply_offsets": str(REPLY_OFFSETS),
        "split": "LOEO by PHEME event",
        "cutoff_seconds": CUTOFF_SECONDS,
        "candidate_pool": "PHEME v4 rumour-only 60-minute snapshots",
        "quiet_gate": f"train-fold {QUIET_QUANTILE:.0%} quantile of 60-minute time since last activity",
        "tiers": {"high": f"top {HIGH_FRACTION:.0%} quiet impact rank within event", "middle": f"next {(MIDDLE_CUMULATIVE_FRACTION - HIGH_FRACTION):.0%}", "low": f"remaining {(1 - MIDDLE_CUMULATIVE_FRACTION):.0%}"},
        "features": list(FEATURE_NAMES),
        "excluded_inputs": "all profile/engagement fields, embeddings, future nodes, labels, and target values",
        "hypothesis": "At 60 minutes, coarse high/middle/low future-impact tiers may be more identifiable among quiet threads than strict-30 exact ranking.",
        "research_safety": "Every snapshot node is audited against raw offset <= 3600; only temporal summaries, observed count, and observed depth are model inputs. Quiet threshold and all fitting use outer-train events only. Held-out impact is read only after probability prediction for evaluation.",
    }
    write_json(output / "run_record.json", record)
    try:
        print(f"Starting strict-60 quiet tier {RUN_MODE} diagnostic.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/4] Loading archived strict-60 artifact and raw offsets...", flush=True)
        graphs = torch.load(DATASET, weights_only=False)
        offsets = load_offsets()
        print(f"  graphs={len(graphs)}, offset rows={len(offsets)}", flush=True)
        print("[2/4] Auditing every snapshot node, edge, and target boundary...", flush=True)
        audit_graphs(graphs, offsets)
        frame = frame_from_graphs(graphs)
        events = sorted(frame.event_id.unique())
        outer_events = [SMOKE_OUTER_EVENT] if smoke else [event for event in events if int((frame.event_id == event).sum()) >= 100]
        if not outer_events:
            raise ValueError("No outer events selected")
        print(f"[3/4] Fitting fixed three-tier classifier for {len(outer_events)} outer fold(s)...", flush=True)
        rows: list[dict[str, object]] = []
        predictions: list[pd.DataFrame] = []
        for index, event in enumerate(outer_events, 1):
            print(f"  fold {index}/{len(outer_events)}: held-out={event}", flush=True)
            train_quiet, test_quiet, threshold = prepare_fold(frame, event, smoke)
            metric, predicted = evaluate_fold(train_quiet, test_quiet, threshold, SMOKE_TREES if smoke else FULL_TREES)
            rows.append(metric)
            predictions.append(predicted)
            print(f"    quiet={metric['test_quiet_threads']}; high precision={metric['high_precision']:.3f}; high recall={metric['high_recall']:.3f}", flush=True)
        metrics = pd.DataFrame(rows)
        print("[4/4] Writing diagnostic evidence and final run record...", flush=True)
        metrics.to_csv(output / "outer_fold_tier_metrics.csv", index=False)
        pd.concat(predictions, ignore_index=True).to_csv(output / "outer_fold_tier_predictions.csv", index=False)
        record.update({
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "outer_events": outer_events,
            "metric_macro_mean": metrics.drop(columns=["outer_event"]).mean(numeric_only=True).to_dict(),
            "output_files": ["outer_fold_tier_metrics.csv", "outer_fold_tier_predictions.csv"],
        })
        write_json(output / "run_record.json", record)
        print(f"SUCCESS: strict-60 quiet tier diagnostic saved to: {output.resolve()}", flush=True)
    except Exception as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}"})
        write_json(output / "run_record.json", record)
        print(f"FAILURE: strict-60 quiet tier diagnostic preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
