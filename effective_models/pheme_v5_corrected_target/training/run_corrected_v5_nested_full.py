"""Foreground full corrected-target replay of the original v5 nested RF."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import torch

import corrected_target_common as target


HERE = Path(__file__).resolve().parent
MODEL_ROOT = HERE.parent
ROOT = HERE.parents[2]
OUT_ROOT = MODEL_ROOT / "experiments"
V5_TRAINING = ROOT / "effective_models" / "pheme_v5_text_rf" / "training"
LEGACY_RESULT = (
    ROOT
    / "effective_models"
    / "pheme_v5_text_rf"
    / "reference_result"
    / "20260831_223827_909201_v5_best_text_oof_min_interventions"
    / "result.json"
)
sys.path.insert(0, str(V5_TRAINING))
import run_nested_text_impact_ranker_rf as v5  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def legacy_reference_crr(reference: dict) -> float:
    """Read CRR@10 from either the trainer report or preserved policy replay."""
    if "mean_eligible_events" in reference:
        return float(reference["mean_eligible_events"]["crr_model"])
    rows = [
        row["outer"]["crr_at_10"]
        for row in reference["outer_events"].values()
        if int(row["outer"]["n_candidate_rumours"]) >= v5.ELIGIBLE_MIN_THREADS
    ]
    if not rows:
        raise ValueError("legacy reference contains no eligible outer events")
    return float(np.mean(rows))


def main() -> None:
    output = OUT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_corrected_v5_nested_full"
    )
    output.mkdir(parents=True, exist_ok=False)
    corrected_dataset = output / "pheme_v5_strict30_corrected_all_roots_semantic.pt"
    record = {
        "status": "running",
        "started_at": now(),
        "mode": "full",
        "experiment": "v5_corrected_all_roots_target_v1",
        "inputs": {
            "protected_graphs": str(target.GRAPHS.resolve()),
            "protected_replies": str(target.REPLIES.resolve()),
            "legacy_result": str(LEGACY_RESULT.resolve()),
        },
        "output_dataset": str(corrected_dataset.resolve()),
        "split": "original nested LOEO by event",
        "cutoff_seconds": target.CUTOFF_SEC,
        "candidate_pool": "original protected v5 rumour-only graphs",
        "target_change_only": "legacy-last-root to corrected-all-traversable-roots",
        "model": "RandomForestRegressor",
        "model_params": v5.MODEL_PARAMS,
        "feature_sets": list(v5.FEATURE_SETS),
        "pca_components": v5.PCA_COMPONENTS,
        "inner_selection": "mean CRR@10 over eligible inner validation events",
        "eligible_min_threads": v5.ELIGIBLE_MIN_THREADS,
        "seed": v5.SEED,
        "research_safety": (
            "Protected inputs are read-only. Corrected graphs are deep copies with only target "
            "fields replaced. Snapshot nodes/edges/features, event membership, cutoff, RF, PCA, "
            "and nested feature-set selection are unchanged from v5."
        ),
    }
    write_record(output, record)
    try:
        print("Starting full corrected-target v5 nested LOEO (foreground job).", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/7] Loading protected strict-30 graphs and reply trees...", flush=True)
        protected = torch.load(target.GRAPHS, weights_only=False)
        maps = target.load_reply_maps()

        print("[2/7] Recomputing all-roots targets on deep-copied graphs...", flush=True)
        graphs, target_summary = target.correct_graphs(protected, maps)
        if [str(g.thread_id) for g in graphs] != [str(g.thread_id) for g in protected]:
            raise ValueError("graph identity/order changed during target correction")
        changed = target_summary.loc[target_summary.target_delta.ne(0)]
        print(
            f"  changed targets={len(changed)}/{len(target_summary)}; "
            f"impact {int(target_summary.legacy_preventable_impact.sum())} -> "
            f"{int(target_summary.corrected_preventable_impact.sum())}",
            flush=True,
        )

        print("[3/7] Saving new versioned corrected graph artifact...", flush=True)
        torch.save(graphs, corrected_dataset)
        target_summary.to_csv(output / "corrected_target_summary.csv", index=False)

        print("[4/7] Loading original fold-safe raw-node feature source...", flush=True)
        raw_nodes = v5.load_raw_node_features()
        events = sorted({str(graph.event_id) for graph in graphs})
        eligible = [
            event
            for event in events
            if sum(str(graph.event_id) == event for graph in graphs) >= v5.ELIGIBLE_MIN_THREADS
        ]

        print("[5/7] Running unchanged nested LOEO feature-set selection and RF fitting...", flush=True)
        results: dict[str, dict] = {}
        selected_sets: dict[str, str] = {}
        inner_rows: list[dict] = []
        oof_rows: list[dict] = []
        for outer_index, test_event in enumerate(events, 1):
            print(f"  outer fold {outer_index}/{len(events)}: {test_event}", flush=True)
            inner_events = [event for event in eligible if event != test_event]
            validation = {feature_set: [] for feature_set in v5.FEATURE_SETS}
            for inner_index, validation_event in enumerate(inner_events, 1):
                print(
                    f"    inner event {inner_index}/{len(inner_events)}: {validation_event}",
                    flush=True,
                )
                train = [
                    graph
                    for graph in graphs
                    if str(graph.event_id) not in (test_event, validation_event)
                ]
                valid = [graph for graph in graphs if str(graph.event_id) == validation_event]
                for feature_set in v5.FEATURE_SETS:
                    prediction = v5.fit_predict(train, valid, raw_nodes, feature_set)
                    score = v5.selection_metrics(valid, prediction)[validation_event]["crr_model"]
                    validation[feature_set].append(score)
                    inner_rows.append(
                        {
                            "outer_event": test_event,
                            "validation_event": validation_event,
                            "feature_set": feature_set,
                            "crr_at_10": score,
                        }
                    )
            means = {
                feature_set: float(np.mean(values))
                for feature_set, values in validation.items()
            }
            chosen = max(means, key=means.get)
            train = [graph for graph in graphs if str(graph.event_id) != test_event]
            test = [graph for graph in graphs if str(graph.event_id) == test_event]
            prediction = v5.fit_predict(train, test, raw_nodes, chosen)
            row = v5.selection_metrics(test, prediction)[test_event]
            row.update(
                {
                    "selected_feature_set": chosen,
                    "inner_validation_mean_crr": means,
                    "inner_validation_events": inner_events,
                }
            )
            results[test_event] = row
            selected_sets[test_event] = chosen
            legacy_by_id = target_summary.set_index("thread_id").legacy_preventable_impact
            for graph, score in zip(test, prediction):
                oof_rows.append(
                    {
                        "thread_id": str(graph.thread_id),
                        "event_id": test_event,
                        "prediction": float(score),
                        "corrected_preventable_impact": int(graph.preventable_impact.item()),
                        "legacy_preventable_impact": int(legacy_by_id[str(graph.thread_id)]),
                        "selected_feature_set": chosen,
                    }
                )
            print(
                f"    selected={chosen}; outer CRR@10={row['crr_model']:.6f}", flush=True
            )
            pd.DataFrame(oof_rows).to_csv(output / "oof_thread_scores_partial.csv", index=False)
            (output / "partial_result.json").write_text(
                json.dumps(
                    {"completed_outer_events": list(results), "event_summary": results}, indent=2
                ),
                encoding="utf-8",
            )

        print("[6/7] Comparing corrected result with the preserved legacy-v5 reference...", flush=True)
        included = [results[event] for event in eligible]
        corrected_mean = {
            key: float(np.mean([row[key] for row in included]))
            for key in ("crr_model", "crr_size", "crr_random", "preventable_recall_model")
        }
        legacy = json.loads(LEGACY_RESULT.read_text(encoding="utf-8"))
        legacy_crr = legacy_reference_crr(legacy)
        comparison = {
            "legacy_v5_crr_at_10": legacy_crr,
            "corrected_v5_crr_at_10": corrected_mean["crr_model"],
            "corrected_minus_legacy": corrected_mean["crr_model"] - legacy_crr,
            "comparability_note": (
                "Model/split/cutoff/features are held fixed, but outcome labels differ; this is a "
                "target-correction sensitivity, not a same-label model improvement."
            ),
        }

        print("[7/7] Writing complete results and run record...", flush=True)
        pd.DataFrame(inner_rows).to_csv(output / "inner_validation_scores.csv", index=False)
        pd.DataFrame(oof_rows).to_csv(output / "oof_thread_scores.csv", index=False)
        record.update(
            {
                "status": "complete",
                "completed_at": now(),
                "graphs": len(graphs),
                "eligible_events": eligible,
                "changed_targets": len(changed),
                "legacy_impact_sum": int(target_summary.legacy_preventable_impact.sum()),
                "corrected_impact_sum": int(target_summary.corrected_preventable_impact.sum()),
                "metrics": corrected_mean,
                "legacy_comparison": comparison,
                "output_files": [
                    corrected_dataset.name,
                    "corrected_target_summary.csv",
                    "inner_validation_scores.csv",
                    "oof_thread_scores_partial.csv",
                    "oof_thread_scores.csv",
                    "partial_result.json",
                    "result.json",
                    "run_record.json",
                ],
            }
        )
        report = {
            "config": dict(record),
            "eligible_events": eligible,
            "event_summary": results,
            "selected_feature_sets": selected_sets,
            "mean_eligible_events": corrected_mean,
            "legacy_comparison": comparison,
        }
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_record(output, record)
        print(
            f"SUCCESS: full corrected-target v5 saved to: {output.resolve()}", flush=True
        )
    except BaseException as error:
        record.update(
            {
                "status": "failed",
                "failed_at": now(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "output_files": sorted(path.name for path in output.iterdir()),
            }
        )
        write_record(output, record)
        print(f"FAILURE: corrected-target v5 run preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
