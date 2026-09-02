"""Strict-30 quiet classifier for membership in the event-global Oracle Top-50.

Unlike the previous quiet-relative tier model, the binary label is whether a
quiet thread belongs to the top 50 preventable-impact threads of its complete
event.  Labels are outcomes only and never become model inputs.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch

from strict30_protected_feature_registry import STRICT30_ROOT, feature_group


RUN_MODE = "smoke"
N_ESTIMATORS = 50
OUTER_EVENT = "charliehebdo"
SMOKE_INNER_EVENTS = 2
SOURCE_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
QUIET_QUANTILE = 0.60
ORACLE_K = 50
SEED = 42

BASE = Path(__file__).resolve().parent
MATERIALIZATION = BASE / "experiments" / "20260902_030441_849111_strict30_protected_feature_materialization"
FEATURE_TABLE = MATERIALIZATION / "strict30_protected_snapshot_features.csv"
MATERIALIZATION_RECORD = MATERIALIZATION / "run_record.json"
DATASET = STRICT30_ROOT / "pheme_graphs_roberta_30min_replyv4_preventableimpact_semantic.pt"
OUT_ROOT = BASE / "experiments"


def _record(path: Path, value: dict) -> None:
    (path / "run_record.json").write_text(json.dumps(value, indent=2), encoding="utf-8")


def _global_top50(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(); result["oracle_global_top50"] = False
    for _, event in result.groupby("event_id", sort=True):
        top = event.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable").head(ORACLE_K).index
        result.loc[top, "oracle_global_top50"] = True
    return result


def _embeddings(graphs: list) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    recency, values = {}, {}
    for index, graph in enumerate(graphs, 1):
        temporal = graph.temporal_features.detach().cpu().numpy().reshape(-1)
        source = torch.nonzero(graph.is_source_mask, as_tuple=False).flatten()
        if len(temporal) != 11 or not np.isfinite(temporal).all() or source.numel() != 1 or graph.x.shape[1] <= 13:
            raise ValueError(f"{graph.thread_id}: invalid protected snapshot input")
        root = int(source.item()); vectors = graph.x[:, 13:].detach().cpu().numpy().astype(np.float32); replies = np.delete(vectors, root, axis=0)
        recency[str(graph.thread_id)] = float(temporal[5])
        values[str(graph.thread_id)] = np.concatenate([vectors[root], replies.mean(axis=0) if len(replies) else np.zeros_like(vectors[root])])
        if index % 500 == 0 or index == len(graphs): print(f"  audited {index}/{len(graphs)} snapshots", flush=True)
    return recency, values


def _quiet_split(frame: pd.DataFrame, recency: dict[str, float], train_events: list[str], test_event: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    raw_train = frame.loc[frame.event_id.isin(train_events)].copy(); raw_test = frame.loc[frame.event_id.eq(test_event)].copy()
    threshold = float(np.quantile(raw_train.thread_id.map(recency), QUIET_QUANTILE))
    train = raw_train.loc[raw_train.thread_id.map(recency) >= threshold].copy(); test = raw_test.loc[raw_test.thread_id.map(recency) >= threshold].copy()
    if train.oracle_global_top50.nunique() != 2 or test.oracle_global_top50.sum() == 0:
        raise ValueError(f"{test_event}: degenerate quiet global-Top50 label")
    return train, test, threshold


def _scores(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], embeddings: dict[str, np.ndarray], weight: float, seed: int) -> np.ndarray:
    train_embed = np.vstack([embeddings[value] for value in train.thread_id]); test_embed = np.vstack([embeddings[value] for value in test.thread_id])
    scaler = StandardScaler().fit(train_embed); pca = PCA(n_components=16, random_state=seed).fit(scaler.transform(train_embed))
    source = ExtraTreesClassifier(n_estimators=N_ESTIMATORS, min_samples_leaf=3, max_features=0.7, class_weight="balanced", n_jobs=1, random_state=seed)
    temporal = ExtraTreesClassifier(n_estimators=N_ESTIMATORS, min_samples_leaf=3, max_features=0.7, class_weight="balanced", n_jobs=1, random_state=seed + 1)
    target = train.oracle_global_top50.to_numpy(bool)
    source.fit(pca.transform(scaler.transform(train_embed)), target)
    temporal.fit(train[columns].to_numpy(np.float32), target)
    source_score = source.predict_proba(pca.transform(scaler.transform(test_embed)))[:, 1]
    temporal_score = temporal.predict_proba(test[columns].to_numpy(np.float32))[:, 1]
    return weight * source_score + (1.0 - weight) * temporal_score


def _metrics(test: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    truth = test.oracle_global_top50.to_numpy(bool); ordered = test.assign(score=score).sort_values(["score", "thread_id"], ascending=[False, True], kind="stable")
    k = int(truth.sum()); hits = int(ordered.head(k).oracle_global_top50.sum())
    return {"quiet_global_top50_ap": float(average_precision_score(truth, score)), "quiet_global_top50_auc": float(roc_auc_score(truth, score)), "quiet_global_top50_recall_at_oracle_quiet_k": hits / k, "oracle_quiet_top50_count": k}


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_strict30_quiet_global_top50_smoke")
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "mode": RUN_MODE, "cutoff_seconds": 1800, "outer_event": OUTER_EVENT, "label": "event-global Oracle Top-50 membership; outcome-only", "quiet_gate": "outer/inner-train 60th recency percentile", "source_weights": list(SOURCE_WEIGHTS), "seed": SEED}
    _record(output, record)
    try:
        print("Starting strict-30 quiet global-Top50 classifier smoke.", flush=True); print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/5] Loading protected materialization and snapshots...", flush=True)
        if json.loads(MATERIALIZATION_RECORD.read_text(encoding="utf-8")).get("status") != "complete": raise ValueError("materialization incomplete")
        frame = pd.read_csv(FEATURE_TABLE, dtype={"thread_id": str, "event_id": str})
        if len(frame) != 2402 or frame.thread_id.duplicated().any(): raise ValueError("invalid feature table")
        columns = [column for column in frame if feature_group(column) in {"activity_level", "temporal_dynamics"}]
        frame = _global_top50(frame); sizes = frame.groupby("event_id").size(); eligible = sorted(sizes[sizes >= 100].index.tolist())
        graphs = torch.load(DATASET, weights_only=False); recency, embeddings = _embeddings(graphs)
        frame = frame.loc[frame.event_id.isin(eligible)].copy()
        print("[2/5] Running inner LOEO source/temporal fusion selection...", flush=True)
        inner_events = [event for event in eligible if event != OUTER_EVENT][:SMOKE_INNER_EVENTS]; selection = []
        for weight in SOURCE_WEIGHTS:
            fold = []
            for index, event in enumerate(inner_events, 1):
                train, test, _ = _quiet_split(frame, recency, [value for value in eligible if value not in {OUTER_EVENT, event}], event)
                fold.append(_metrics(test, _scores(train, test, columns, embeddings, weight, SEED + index * 10)))
            selection.append({"source_weight": weight, "inner_folds": len(fold), "mean_inner_ap": float(np.mean([item["quiet_global_top50_ap"] for item in fold])), "mean_inner_recall": float(np.mean([item["quiet_global_top50_recall_at_oracle_quiet_k"] for item in fold]))})
        selected = pd.DataFrame(selection).sort_values(["mean_inner_ap", "mean_inner_recall", "source_weight"], ascending=[False, False, True], kind="stable")
        weight = float(selected.iloc[0].source_weight); print(f"  selected source_weight={weight:.2f}", flush=True)
        print("[3/5] Fitting outer-train quiet classifier and scoring held-out quiet pool...", flush=True)
        train, test, threshold = _quiet_split(frame, recency, [event for event in eligible if event != OUTER_EVENT], OUTER_EVENT)
        score = _scores(train, test, columns, embeddings, weight, SEED + 1000); metrics = _metrics(test, score)
        print("[4/5] Writing held-out quiet predictions...", flush=True)
        prediction = test[["thread_id", "event_id", "preventable_impact", "oracle_global_top50"]].copy(); prediction["global_top50_score"] = score; prediction.sort_values(["global_top50_score", "thread_id"], ascending=[False, True], inplace=True)
        prediction.to_csv(output / "held_out_quiet_predictions.csv", index=False); selected.to_csv(output / "inner_source_weight_selection.csv", index=False)
        print("[5/5] Finalizing smoke run record...", flush=True)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "quiet_threshold": threshold, "train_quiet_threads": len(train), "test_quiet_threads": len(test), "metrics": metrics, "output_files": ["held_out_quiet_predictions.csv", "inner_source_weight_selection.csv"]}); _record(output, record)
        print(f"SUCCESS: strict-30 quiet global-Top50 smoke saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}); _record(output, record); print(f"FAILURE: output preserved at: {output.resolve()}", flush=True); raise


if __name__ == "__main__": main()
