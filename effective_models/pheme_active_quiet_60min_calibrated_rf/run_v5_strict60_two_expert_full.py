"""Five-event LOEO evaluation of the protected-v5 strict-60 two-expert policy."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATASET = (
    HERE
    / "experiments"
    / "20260902_231543_841935_v5_strict60_materialization_full"
    / "pheme_graphs_roberta_v5_strict60_preventableimpact.pt"
)
MATERIALIZATION_RECORD = DATASET.parent / "run_record.json"
REPLIES = (
    ROOT
    / "data"
    / "protected_research_assets"
    / "pheme_v5_strict30"
    / "pheme_reply_level_v4.csv"
)
OUT_ROOT = HERE / "experiments"
EVENTS = (
    "charliehebdo",
    "ferguson",
    "germanwings-crash",
    "ottawashooting",
    "prince-toronto",
    "putinmissing",
    "sydneysiege",
)
EXPECTED_ARTIFACT_EVENTS = EVENTS + ("ebola-essien", "gurlitt")
SEED = 42
TREES = 300
TOP_K = 50

sys.path.insert(
    0, str(ROOT / "effective_models" / "pheme_active_quiet_calibrated_rf" / "training")
)
from pheme_quiet_expert_calibration_common import fit_calibrators


def save_record(output_dir: Path, record: dict) -> None:
    (output_dir / "run_record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


def recency(graphs: list) -> np.ndarray:
    return np.asarray([float(g.temporal_features[8]) for g in graphs], np.float32)


def load_interval_features() -> tuple[
    dict[str, tuple[float, float, float]], dict[tuple[str, str], float]
]:
    frame = pd.read_csv(
        REPLIES,
        dtype={"thread_id": str, "tweet_id": str, "parent_id": str},
        usecols=["thread_id", "tweet_id", "parent_id", "is_source", "offset_sec"],
        low_memory=False,
    )
    sources = frame.loc[frame.is_source.eq(1), ["thread_id", "tweet_id"]].rename(
        columns={"tweet_id": "source"}
    )
    replies = frame.loc[
        frame.is_source.ne(1) & frame.offset_sec.gt(1800) & frame.offset_sec.le(3600)
    ].merge(sources, on="thread_id", how="left")
    replies["reply_to_reply"] = replies.parent_id.ne(replies.source)
    max_branch = (
        replies.groupby(["thread_id", "parent_id"])
        .size()
        .groupby(level=0)
        .max()
    )
    aggregate = replies.groupby("thread_id").agg(
        count=("tweet_id", "size"), reply_to_reply=("reply_to_reply", "sum")
    )
    aggregate["max_branch"] = max_branch
    intervals = {
        str(thread_id): (
            float(np.log1p(row["count"])),
            float(row["reply_to_reply"] / max(row["count"], 1)),
            float(np.log1p(row["max_branch"])),
        )
        for thread_id, row in aggregate.iterrows()
    }
    offsets = {
        (str(row.thread_id), str(row.tweet_id)): float(row.offset_sec)
        for row in frame.itertuples(index=False)
    }
    return intervals, offsets


def scalar_features(graph, intervals: dict) -> np.ndarray:
    temporal = graph.temporal_features.numpy().astype(np.float32)
    depth = graph.x[:, 0].numpy()
    interval = intervals.get(str(graph.thread_id), (0.0, 0.0, 0.0))
    acceleration = float(temporal[3:6].sum() - temporal[:3].sum())
    return np.r_[
        temporal,
        np.log1p(graph.num_nodes),
        depth.mean(),
        depth.max(),
        interval,
        acceleration,
    ].astype(np.float32)


def raw_text_features(graphs: list) -> np.ndarray:
    rows = []
    for graph in graphs:
        embeddings = graph.x[:, 13:].numpy().astype(np.float32)
        source_index = np.flatnonzero(graph.x[:, 3].numpy() > 0.5)
        if len(source_index) != 1 or embeddings.shape[1] != 768:
            raise ValueError(f"Invalid source/text layout for {graph.thread_id}")
        replies = np.delete(embeddings, source_index[0], axis=0)
        reply_mean = replies.mean(0) if len(replies) else np.zeros(768, np.float32)
        rows.append(np.r_[embeddings[source_index[0]], reply_mean])
    return np.vstack(rows)


def fit_predict(train: list, test: list, intervals: dict, seed: int) -> np.ndarray:
    train_scalar = np.vstack([scalar_features(g, intervals) for g in train])
    test_scalar = np.vstack([scalar_features(g, intervals) for g in test])
    train_text = raw_text_features(train)
    test_text = raw_text_features(test)
    text_scaler = StandardScaler().fit(train_text)
    pca = PCA(n_components=min(32, len(train) - 1), random_state=seed).fit(
        text_scaler.transform(train_text)
    )
    train_x = np.c_[train_scalar, pca.transform(text_scaler.transform(train_text))]
    test_x = np.c_[test_scalar, pca.transform(text_scaler.transform(test_text))]
    train_age = np.asarray([g.x[:, 12].mean().item() for g in train])[:, None]
    test_age = np.asarray([g.x[:, 12].mean().item() for g in test])[:, None]
    train_x = np.c_[train_x, train_age]
    test_x = np.c_[test_x, test_age]
    scaler = StandardScaler().fit(train_x)
    model = RandomForestRegressor(
        n_estimators=TREES,
        max_depth=8,
        min_samples_leaf=3,
        max_features=0.8,
        n_jobs=-1,
        random_state=seed,
    ).fit(scaler.transform(train_x), [float(g.preventable_y) for g in train])
    return model.predict(scaler.transform(test_x)).astype(np.float32)


def gated_predict(
    train: list, test: list, intervals: dict, seed: int
) -> tuple[np.ndarray, np.ndarray, float]:
    threshold = float(np.quantile(recency(train), 0.6))
    train_quiet = recency(train) >= threshold
    test_quiet = recency(test) >= threshold
    predictions = np.empty(len(test), np.float32)
    for quiet in (False, True):
        expert_train = [g for g, flag in zip(train, train_quiet) if flag == quiet]
        expert_test = [g for g, flag in zip(test, test_quiet) if flag == quiet]
        if not expert_train or not expert_test:
            raise ValueError(f"Empty {'quiet' if quiet else 'active'} expert split")
        predictions[test_quiet == quiet] = fit_predict(
            expert_train, expert_test, intervals, seed + int(quiet)
        )
    return predictions, test_quiet, threshold


def calibration_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    event_count = frame.validation_event.nunique()
    frame["event_weight"] = frame.groupby("validation_event").thread_id.transform(
        lambda group: len(frame) / (event_count * len(group))
    )
    return frame


def apply_calibration(raw: np.ndarray, quiet: np.ndarray, calibrators: dict) -> np.ndarray:
    calibrated = np.empty(len(raw), np.float32)
    for expert, mask in (("quiet", quiet), ("active", ~quiet)):
        calibrated[mask] = (
            calibrators[expert]["coefficient"] * raw[mask]
            + calibrators[expert]["intercept"]
        )
    return calibrated


def evaluate_event(frame: pd.DataFrame) -> dict:
    ranked = frame.sort_values(["score", "thread_id"], ascending=[False, True])
    oracle = frame.sort_values(
        ["preventable_impact_60m", "thread_id"], ascending=[False, True]
    )
    selected = ranked.head(TOP_K)
    oracle_top = oracle.head(TOP_K)
    selected_ids = set(selected.thread_id)
    oracle_ids = set(oracle_top.thread_id)
    blocked = float(selected.preventable_impact_60m.sum())
    optimal = float(oracle_top.preventable_impact_60m.sum())
    total = float(frame.preventable_impact_60m.sum())
    quiet_selected = selected.loc[selected.quiet]
    active_selected = selected.loc[~selected.quiet]
    return {
        "threads": int(len(frame)),
        "blocked_future_nodes_at_50": blocked,
        "total_future_nodes": total,
        "oracle_blocked_future_nodes_at_50": optimal,
        "reduction_at_50": blocked / total if total else 0.0,
        "oracle_efficiency_at_50": blocked / optimal if optimal else 0.0,
        "oracle_top50_hits": len(selected_ids & oracle_ids),
        "oracle_top50_missed": len(oracle_ids - selected_ids),
        "active_selected": int(len(active_selected)),
        "active_oracle_top50_hits": len(set(active_selected.thread_id) & oracle_ids),
        "quiet_selected": int(len(quiet_selected)),
        "quiet_oracle_top50_hits": len(set(quiet_selected.thread_id) & oracle_ids),
    }


def main() -> None:
    output_dir = OUT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + "_v5_strict60_two_expert_full"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET),
        "materialization_record": str(MATERIALIZATION_RECORD),
        "reply_table": str(REPLIES),
        "events": list(EVENTS),
        "split": "seven eligible events (>=100 threads) leave-one-event-out; nested outer-train event OOF calibration",
        "cutoff_seconds": 3600,
        "seed": SEED,
        "trees": TREES,
        "top_k": TOP_K,
        "features": (
            "strict-60 temporal/depth/source+reply PCA/account-age plus explicit "
            "30-60 reply count, reply-to-reply rate, branching, and acceleration"
        ),
        "research_safety": (
            "All model inputs are <=60 minutes; normalization/PCA/gating are fitted "
            "on training data; calibration uses outer-train event OOF predictions only."
        ),
    }
    save_record(output_dir, record)
    try:
        print("Starting protected-v5 strict-60 calibrated two-expert full LOEO.", flush=True)
        print(f"Output directory: {output_dir.resolve()}", flush=True)
        print("[1/6] Loading and validating strict-60 inputs...", flush=True)
        materialization = json.loads(MATERIALIZATION_RECORD.read_text(encoding="utf-8"))
        if materialization.get("status") != "complete":
            raise ValueError("Strict-60 materialization is not complete")
        if materialization.get("cutoff_seconds") != 3600:
            raise ValueError("Strict-60 materialization cutoff mismatch")
        if Path(materialization.get("output_dataset", "")).resolve() != DATASET.resolve():
            raise ValueError("Strict-60 materialization output identity mismatch")
        graphs = torch.load(DATASET, weights_only=False)
        intervals, raw_offsets = load_interval_features()
        graph_events = {str(g.event_id) for g in graphs}
        if graph_events != set(EXPECTED_ARTIFACT_EVENTS):
            raise ValueError(f"Unexpected events: {sorted(graph_events)}")
        for index, graph in enumerate(graphs, 1):
            node_offsets = []
            for node_id in graph.node_ids:
                key = (str(graph.thread_id), str(node_id))
                if key not in raw_offsets:
                    raise ValueError(f"Missing raw timestamp for {key}")
                node_offsets.append(raw_offsets[key])
            if node_offsets and max(node_offsets) > 3600.0 + 1e-6:
                raise ValueError(f"Post-cutoff node for {graph.thread_id}")
            if graph.x.shape[1] != 781 or not torch.allclose(
                graph.x[:, [8, 9, 10, 11]], torch.zeros_like(graph.x[:, [8, 9, 10, 11]])
            ):
                raise ValueError(f"Protected feature layout failed for {graph.thread_id}")
            if index % 600 == 0 or index == len(graphs):
                print(f"  graph audit {index}/{len(graphs)}", flush=True)
        graphs = [g for g in graphs if str(g.event_id) in EVENTS]
        all_scores = []
        fold_metrics = []
        fold_details = []
        print("[2/6] Running seven outer LOEO folds with nested calibration...", flush=True)
        for fold_index, outer_event in enumerate(EVENTS, 1):
            print(f"  outer fold {fold_index}/{len(EVENTS)}: {outer_event}", flush=True)
            train = [g for g in graphs if str(g.event_id) != outer_event]
            test = [g for g in graphs if str(g.event_id) == outer_event]
            calibration_rows = []
            inner_events = [event for event in EVENTS if event != outer_event]
            for inner_index, validation_event in enumerate(inner_events, 1):
                print(
                    f"    calibration fold {inner_index}/{len(inner_events)}: "
                    f"{validation_event}",
                    flush=True,
                )
                validation = [g for g in train if str(g.event_id) == validation_event]
                inner_train = [g for g in train if str(g.event_id) != validation_event]
                raw, quiet, _ = gated_predict(
                    inner_train,
                    validation,
                    intervals,
                    SEED + fold_index * 100 + inner_index,
                )
                calibration_rows.extend(
                    {
                        "validation_event": validation_event,
                        "thread_id": str(graph.thread_id),
                        "expert": "quiet" if is_quiet else "active",
                        "raw_score": float(score),
                        "target": float(graph.preventable_y),
                    }
                    for graph, score, is_quiet in zip(validation, raw, quiet)
                )
            calibrators, calibration_details = fit_calibrators(
                calibration_frame(calibration_rows)
            )
            raw, quiet, threshold = gated_predict(
                train, test, intervals, SEED + fold_index * 1000
            )
            scores = apply_calibration(raw, quiet, calibrators)
            fold_frame = pd.DataFrame(
                {
                    "thread_id": [str(g.thread_id) for g in test],
                    "event_id": [str(g.event_id) for g in test],
                    "preventable_impact_60m": [float(g.preventable_impact) for g in test],
                    "preventable_y_60m": [float(g.preventable_y) for g in test],
                    "raw_score": raw,
                    "score": scores,
                    "quiet": quiet,
                }
            )
            metrics = evaluate_event(fold_frame)
            metrics["event_id"] = outer_event
            fold_metrics.append(metrics)
            all_scores.append(fold_frame)
            fold_details.append(
                {
                    "outer_event": outer_event,
                    "gate_threshold": threshold,
                    "calibration": calibration_details,
                }
            )
            print(
                f"    Top-50 hits={metrics['oracle_top50_hits']}/50, "
                f"oracle efficiency={metrics['oracle_efficiency_at_50']:.4f}",
                flush=True,
            )
        print("[3/6] Computing aggregate Top-50 policy metrics...", flush=True)
        score_frame = pd.concat(all_scores, ignore_index=True)
        metrics_frame = pd.DataFrame(fold_metrics)
        aggregate = {
            "events": len(EVENTS),
            "selected_threads": TOP_K * len(EVENTS),
            "blocked_future_nodes_at_50": float(
                metrics_frame.blocked_future_nodes_at_50.sum()
            ),
            "total_future_nodes": float(metrics_frame.total_future_nodes.sum()),
            "oracle_blocked_future_nodes_at_50": float(
                metrics_frame.oracle_blocked_future_nodes_at_50.sum()
            ),
            "reduction_at_50": float(
                metrics_frame.blocked_future_nodes_at_50.sum()
                / metrics_frame.total_future_nodes.sum()
            ),
            "oracle_efficiency_at_50": float(
                metrics_frame.blocked_future_nodes_at_50.sum()
                / metrics_frame.oracle_blocked_future_nodes_at_50.sum()
            ),
            "oracle_top50_hits": int(metrics_frame.oracle_top50_hits.sum()),
            "oracle_top50_missed": int(metrics_frame.oracle_top50_missed.sum()),
            "active_selected": int(metrics_frame.active_selected.sum()),
            "active_oracle_top50_hits": int(
                metrics_frame.active_oracle_top50_hits.sum()
            ),
            "quiet_selected": int(metrics_frame.quiet_selected.sum()),
            "quiet_oracle_top50_hits": int(
                metrics_frame.quiet_oracle_top50_hits.sum()
            ),
        }
        print("[4/6] Writing per-thread and per-event results...", flush=True)
        score_frame.to_csv(output_dir / "loeo_scores.csv", index=False)
        metrics_frame.to_csv(output_dir / "event_metrics.csv", index=False)
        print("[5/6] Writing reproducibility and safety evidence...", flush=True)
        record.update(
            {
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "graphs": len(graphs),
                "fold_details": fold_details,
                "aggregate_metrics": aggregate,
                "event_metrics": fold_metrics,
                "output_files": ["loeo_scores.csv", "event_metrics.csv"],
            }
        )
        save_record(output_dir, record)
        print("[6/6] Complete.", flush=True)
        print(
            "SUCCESS: protected-v5 strict-60 calibrated two-expert full saved to: "
            f"{output_dir.resolve()}",
            flush=True,
        )
    except BaseException as error:
        record.update(
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "output_files": sorted(
                    path.name for path in output_dir.iterdir() if path.is_file()
                ),
            }
        )
        save_record(output_dir, record)
        print(
            "FAILURE: protected-v5 strict-60 calibrated two-expert full preserved at: "
            f"{output_dir.resolve()}",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
