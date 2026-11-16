"""Shared smoke/full workflow for calibrated Active/Quiet RF experts."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from effective_models.pheme_active_quiet_calibrated_rf import config
from effective_models.pheme_active_quiet_calibrated_rf.evaluation.metrics import curve_rows
from effective_models.pheme_active_quiet_calibrated_rf.features.account_age_text_temporal_features import (
    load_graphs,
    load_node_offsets,
    load_safe_nodes,
)
from effective_models.pheme_active_quiet_calibrated_rf.features.expert_score_calibration import (
    calibrated_scores,
    calibration_safety_statement,
    fit_calibrators,
    inner_oof_predictions,
)
from effective_models.pheme_active_quiet_calibrated_rf.features.quiet_active_gate_features import (
    gated_predict, safety_statement,
)


MODELS = ("uncalibrated_two_expert_rf", "train_only_calibrated_two_expert_rf")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def observable_keys(graphs: list[object]) -> set[tuple[str, str]]:
    return {
        (str(graph.thread_id), str(tweet_id))
        for graph in graphs
        for tweet_id in graph.node_ids
    }


def run_experiment(mode: str) -> Path:
    if mode not in {"smoke", "full"}:
        raise ValueError(f"Unsupported mode: {mode}")
    is_smoke = mode == "smoke"
    started_at = datetime.now(timezone.utc)
    suffix = "active_quiet_calibrated_smoke" if is_smoke else "active_quiet_calibrated_full"
    output = config.EXPERIMENTS_DIR / f"{started_at:%Y%m%d_%H%M%S_%f}_{suffix}"
    output.mkdir(parents=True, exist_ok=False)
    record_path = output / "run_record.json"
    parameters = config.SMOKE_PARAMETERS if is_smoke else config.FULL_PARAMETERS
    record: dict[str, object] = {
        "status": "running",
        "started_at": started_at.isoformat(),
        "mode": mode,
        "model": "PHEME train-only calibrated Active/Quiet RF",
        "configuration": {
            "cutoff_seconds": config.CUTOFF_SECONDS,
            "quiet_quantile": config.QUIET_QUANTILE,
            "calibration_alpha": config.CALIBRATION_ALPHA,
            "budgets": list(config.BUDGETS),
            "parameters": parameters,
            "smoke_difference": (
                "four-event subset and five trees; same cutoff validation, gate, expert-specific "
                "fold-local PCA/scalers/RF, inner-event OOF calibration, and ranking metrics"
            ),
        },
        "input_artifacts": {
            "graph_dataset": str(config.DATASET_PATH),
            "raw_nodes": str(config.RAW_NODES_PATH),
            "reply_offsets": str(config.REPLIES_PATH),
        },
        "split": "outer LOEO with inner event-OOF expert calibration",
        "cutoff_seconds": config.CUTOFF_SECONDS,
        "seed": config.SEED,
        "metrics_results": None,
        "output_file_inventory": [],
        "failure_details": None,
    }
    _write_json(record_path, record)
    print(f"Starting calibrated Active/Quiet RF {mode} run.", flush=True)
    print(f"Output directory: {output.resolve()}", flush=True)

    try:
        print("[1/5] Loading protected strict-30 graphs...", flush=True)
        all_graphs, all_events, all_eligible = load_graphs()
        if is_smoke:
            events = [
                event for event in ("charliehebdo", "ferguson", "germanwings-crash", "ottawashooting")
                if event in all_events
            ]
            graphs = [graph for graph in all_graphs if str(graph.event_id) in events]
            outer_events = events[:1]
            eligible = events
        else:
            graphs, events, outer_events, eligible = all_graphs, all_events, all_events, all_eligible

        print("[2/5] Loading immutable safe-node fields and validating cutoff offsets...", flush=True)
        keys = observable_keys(graphs)
        depth_nodes, account_age = load_safe_nodes(keys)
        node_offsets = load_node_offsets(keys)

        print(f"[3/5] Running {len(outer_events)} outer fold(s) with train-only calibration...", flush=True)
        scores_all, curves_all, calibration_all = [], [], []
        details_all, gates, outer_results = [], [], {}
        for index, event in enumerate(outer_events, start=1):
            print(f"  outer {index}/{len(outer_events)}: held_out={event}", flush=True)
            outer_train = [graph for graph in graphs if str(graph.event_id) != event]
            held_out = [graph for graph in graphs if str(graph.event_id) == event]
            oof = inner_oof_predictions(
                outer_train,
                depth_nodes,
                account_age,
                node_offsets,
                parameters,
                progress_prefix="    ",
            )
            calibrators, details = fit_calibrators(oof)
            raw_score, gate, _ = gated_predict(
                outer_train, held_out, depth_nodes, account_age, node_offsets, parameters
            )
            calibrated = calibrated_scores(raw_score, held_out, gate, calibrators)
            score_frame = pd.DataFrame(
                {
                    "thread_id": [str(graph.thread_id) for graph in held_out],
                    "event_id": [str(graph.event_id) for graph in held_out],
                    "uncalibrated_two_expert_score": raw_score,
                    "calibrated_two_expert_score": calibrated,
                    "preventable_impact": [float(graph.preventable_impact.item()) for graph in held_out],
                }
            )
            raw_curve = pd.DataFrame(
                curve_rows(score_frame, "uncalibrated_two_expert_score", MODELS[0])
            )
            calibrated_curve = pd.DataFrame(
                curve_rows(score_frame, "calibrated_two_expert_score", MODELS[1])
            )
            curves_all.extend([raw_curve, calibrated_curve])
            scores_all.append(score_frame)
            calibration_all.append(oof.assign(outer_test_event=event))
            details_all.extend({"outer_test_event": event, **row} for row in details)
            gates.append({"outer_test_event": event, **gate})
            budget10_raw = float(raw_curve.loc[raw_curve["budget"].eq(10), "model_reduction"].iloc[0])
            budget10_cal = float(calibrated_curve.loc[calibrated_curve["budget"].eq(10), "model_reduction"].iloc[0])
            outer_results[event] = {
                "uncalibrated_reduction_at_10": budget10_raw,
                "calibrated_reduction_at_10": budget10_cal,
                "gate": gate,
                "calibrators": details,
            }
            print(
                f"    reduction@10: uncalibrated={budget10_raw:.4f}; calibrated={budget10_cal:.4f}",
                flush=True,
            )

        print("[4/5] Writing paired scores, calibration evidence, and curves...", flush=True)
        scores = pd.concat(scores_all, ignore_index=True)
        curves = pd.concat(curves_all, ignore_index=True)
        if scores["thread_id"].duplicated().any() or not np.isfinite(
            scores[["uncalibrated_two_expert_score", "calibrated_two_expert_score"]].to_numpy()
        ).all():
            raise ValueError("Paired OOF integrity failure")
        summary = (
            curves.loc[curves["event_id"].isin(eligible)]
            .groupby(["model", "budget"], as_index=False)
            .agg(
                eligible_events=("event_id", "nunique"),
                mean_oracle_efficiency=("oracle_efficiency", "mean"),
                mean_model_reduction=("model_reduction", "mean"),
            )
        )
        scores.to_csv(output / "paired_oof_scores.csv", index=False)
        curves.to_csv(output / "per_event_paired_budget_curve.csv", index=False)
        summary.to_csv(output / "eligible_event_paired_budget_summary.csv", index=False)
        pd.concat(calibration_all, ignore_index=True).to_csv(
            output / "inner_oof_calibration_predictions.csv", index=False
        )
        pd.DataFrame(details_all).to_csv(output / "outer_fold_calibration_details.csv", index=False)
        pd.DataFrame(gates).to_csv(output / "outer_fold_gate_details.csv", index=False)

        print("[5/5] Finalizing complete run record...", flush=True)
        result = {
            "outer_events": outer_results,
            "eligible_event_budget_summary": summary.to_dict(orient="records"),
        }
        inventory = [
            "paired_oof_scores.csv",
            "per_event_paired_budget_curve.csv",
            "eligible_event_paired_budget_summary.csv",
            "inner_oof_calibration_predictions.csv",
            "outer_fold_calibration_details.csv",
            "outer_fold_gate_details.csv",
            "run_record.json",
        ]
        record.update(
            {
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "metrics_results": result,
                "research_safety": {
                    "dataset_and_labels_unchanged": True,
                    "outer_event_excluded_from_gate_experts_and_calibration": True,
                    "snapshot_node_offsets_at_or_before_cutoff_verified": True,
                    "mutable_profile_fields_excluded": True,
                    "smoke_is_not_a_research_result": is_smoke,
                    "details": safety_statement() + " " + calibration_safety_statement(),
                },
                "output_file_inventory": inventory,
            }
        )
        _write_json(record_path, record)
        print(f"SUCCESS: calibrated Active/Quiet RF {mode} saved to: {output.resolve()}", flush=True)
        return output
    except Exception:
        record.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "failure_details": traceback.format_exc(),
                "output_file_inventory": [path.name for path in output.iterdir() if path.is_file()],
            }
        )
        _write_json(record_path, record)
        print(f"FAILURE: calibrated Active/Quiet RF {mode} preserved at: {output.resolve()}", flush=True)
        raise
