"""Bounded corrected-target v5 feature/model smoke."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

import corrected_target_common as target


HERE = Path(__file__).resolve().parent
MODEL_ROOT = HERE.parent
ROOT = HERE.parents[2]
OUT_ROOT = MODEL_ROOT / "experiments"
V5_TRAINING = ROOT / "effective_models" / "pheme_v5_text_rf" / "training"
sys.path.insert(0, str(V5_TRAINING))
import run_nested_text_impact_ranker_rf as v5  # noqa: E402

TRAIN_EVENTS = ("charliehebdo", "ferguson")
TEST_EVENT = "germanwings-crash"
PER_EVENT = 80
KNOWN_MISMATCH = "552808071387701248"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def main() -> None:
    output = OUT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_corrected_v5_smoke"
    )
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "started_at": now(),
        "mode": "smoke",
        "dataset": str(target.GRAPHS.resolve()),
        "reply_source": str(target.REPLIES.resolve()),
        "split": {"train_events": list(TRAIN_EVENTS), "test_event": TEST_EVENT},
        "cutoff_seconds": target.CUTOFF_SEC,
        "model": "original v5 RandomForestRegressor pipeline",
        "feature_sets": list(v5.FEATURE_SETS),
        "seed": v5.SEED,
        "research_safety": "Only target fields are changed on deep-copied graphs; event split and strict-30 inputs are unchanged.",
    }
    write(output, record)
    try:
        print("Starting corrected-target v5 smoke.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/6] Loading protected graphs and reply maps...", flush=True)
        graphs = torch.load(target.GRAPHS, weights_only=False)
        maps = target.load_reply_maps()
        print("[2/6] Selecting bounded event-disjoint smoke graphs...", flush=True)
        selected = []
        for event in (*TRAIN_EVENTS, TEST_EVENT):
            event_graphs = [graph for graph in graphs if str(graph.event_id) == event]
            selected.extend(event_graphs[:PER_EVENT])
        known = next(graph for graph in graphs if str(graph.thread_id) == KNOWN_MISMATCH)
        if all(str(graph.thread_id) != KNOWN_MISMATCH for graph in selected):
            selected.append(known)
        print("[3/6] Recomputing corrected all-roots targets on deep copies...", flush=True)
        corrected, summary = target.correct_graphs(selected, maps, progress_every=80)
        known_row = summary.loc[summary.thread_id.eq(KNOWN_MISMATCH)].iloc[0]
        if int(known_row.legacy_preventable_impact) != 0 or int(known_row.corrected_preventable_impact) != 93:
            raise ValueError("known multi-root correction did not reproduce 0 -> 93")
        print("[4/6] Loading original raw-node feature source...", flush=True)
        raw_nodes = v5.load_raw_node_features()
        train = [graph for graph in corrected if str(graph.event_id) in TRAIN_EVENTS]
        test = [graph for graph in corrected if str(graph.event_id) == TEST_EVENT]
        if {str(g.event_id) for g in train} & {str(g.event_id) for g in test}:
            raise ValueError("event overlap in smoke split")
        print("[5/6] Fitting and predicting both original v5 feature paths...", flush=True)
        metrics = {}
        for feature_set in v5.FEATURE_SETS:
            print(f"  feature set: {feature_set}", flush=True)
            prediction = v5.fit_predict(train, test, raw_nodes, feature_set)
            if prediction.shape != (len(test),) or not np.isfinite(prediction).all():
                raise ValueError(f"invalid {feature_set} prediction")
            metrics[feature_set] = {
                "train_graphs": len(train),
                "test_graphs": len(test),
                "prediction_min": float(prediction.min()),
                "prediction_max": float(prediction.max()),
            }
        print("[6/6] Writing smoke evidence...", flush=True)
        summary.to_csv(output / "corrected_target_summary.csv", index=False)
        record.update(
            {
                "status": "complete",
                "completed_at": now(),
                "graphs": len(corrected),
                "changed_targets": int((summary.target_delta != 0).sum()),
                "known_correction": known_row.to_dict(),
                "metrics": metrics,
                "output_files": ["corrected_target_summary.csv", "run_record.json"],
            }
        )
        write(output, record)
        print(f"SUCCESS: corrected-target v5 smoke saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": now(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(), "output_files": sorted(p.name for p in output.iterdir())})
        write(output, record)
        print(f"FAILURE: corrected-target v5 smoke preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
