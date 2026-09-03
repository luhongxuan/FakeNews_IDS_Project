"""Corrected-target active/quiet plus quiet-tier hybrid experiment runner."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
MODEL_ROOT = HERE.parent
ROOT = HERE.parents[2]
OUT_ROOT = MODEL_ROOT / "experiments"
LEGACY_TRAINING = ROOT / "effective_models" / "pheme_active_quiet_calibrated_rf" / "training"
sys.path.insert(0, str(LEGACY_TRAINING))
import run_quiet_tier_hybrid_smoke as legacy  # noqa: E402
import pheme_account_age_v5_common as v5_common  # noqa: E402


CORRECTED_RUN = (
    ROOT / "effective_models" / "pheme_v5_corrected_target" / "experiments"
    / "20260903_143643_168807_corrected_v5_nested_full"
)
CORRECTED_DATASET = CORRECTED_RUN / "pheme_v5_strict30_corrected_all_roots_semantic.pt"
PROTECTED_DATASET = legacy.DATASET
TARGET_FIELDS = {"preventable_impact", "preventable_y"}
EXPECTED_CHANGED_TARGETS = 23
EXPECTED_LEGACY_IMPACT_SUM = 11730
EXPECTED_CORRECTED_IMPACT_SUM = 12077


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def values_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    return left == right


def load_and_audit_graphs() -> tuple[list, list[str], list[str], dict]:
    """Prove that the versioned corrected artifact differs only in target fields."""
    protected = torch.load(PROTECTED_DATASET, weights_only=False)
    corrected = torch.load(CORRECTED_DATASET, weights_only=False)
    if len(protected) != 2402 or len(corrected) != len(protected):
        raise ValueError("unexpected corrected/protected graph count")
    changed = 0
    for legacy_graph, corrected_graph in zip(protected, corrected):
        if set(legacy_graph.keys()) != set(corrected_graph.keys()):
            raise ValueError("graph field set changed during target correction")
        for field in legacy_graph.keys():
            if field in TARGET_FIELDS:
                continue
            if not values_equal(legacy_graph[field], corrected_graph[field]):
                raise ValueError(
                    f"non-target graph field changed for {corrected_graph.thread_id}: {field}"
                )
        if (
            float(legacy_graph.preventable_impact.item())
            != float(corrected_graph.preventable_impact.item())
        ):
            changed += 1
    legacy_sum = int(sum(float(graph.preventable_impact.item()) for graph in protected))
    corrected_sum = int(sum(float(graph.preventable_impact.item()) for graph in corrected))
    if (
        changed != EXPECTED_CHANGED_TARGETS
        or legacy_sum != EXPECTED_LEGACY_IMPACT_SUM
        or corrected_sum != EXPECTED_CORRECTED_IMPACT_SUM
    ):
        raise ValueError(
            f"corrected target audit mismatch: changed={changed}, sums={legacy_sum}->{corrected_sum}"
        )
    events = sorted({str(graph.event_id) for graph in corrected})
    eligible = [
        event for event in events
        if sum(str(graph.event_id) == event for graph in corrected) >= legacy.ELIGIBLE_MIN_THREADS
    ]
    return corrected, events, eligible, {
        "graphs": len(corrected), "changed_targets": changed,
        "legacy_impact_sum": legacy_sum, "corrected_impact_sum": corrected_sum,
    }


def overlay_corrected_targets(frame: pd.DataFrame, graphs: list) -> tuple[pd.DataFrame, int]:
    """Create a derived in-memory tier frame; never mutate the source materialization."""
    if frame.thread_id.duplicated().any():
        raise ValueError("duplicate materialized thread identity")
    target = pd.Series(
        {str(graph.thread_id): float(graph.preventable_impact.item()) for graph in graphs},
        name="corrected_preventable_impact",
    )
    if set(frame.thread_id) != set(target.index):
        raise ValueError("corrected graph/materialization identity mismatch")
    result = frame.copy()
    result["legacy_preventable_impact"] = result.preventable_impact.astype(float)
    result["preventable_impact"] = result.thread_id.map(target).astype(float)
    changed = int(result.legacy_preventable_impact.ne(result.preventable_impact).sum())
    if changed != EXPECTED_CHANGED_TARGETS:
        raise ValueError(f"tier frame target overlay changed {changed}, expected 23")
    return result, changed


def raw_components(
    train: list, test: list, depth_nodes, account_age, offsets,
    frame: pd.DataFrame, temporal_columns: list[str],
    embeddings: dict[str, np.ndarray], seed: int, model_params: dict,
) -> tuple[np.ndarray, dict, dict[float, np.ndarray]]:
    utility, gate, _ = legacy.gated_predict(
        train, test, depth_nodes, account_age, offsets, model_params
    )
    train_quiet = legacy.recency_values(train) >= gate["threshold"]
    test_quiet = legacy.recency_values(test) >= gate["threshold"]
    quiet_train = [graph for graph, include in zip(train, train_quiet) if include]
    quiet_test = [graph for graph, include in zip(test, test_quiet) if include]
    probability = (
        legacy.tier_probabilities(
            quiet_train, quiet_test, frame, temporal_columns, embeddings, seed
        )
        if quiet_test else np.empty((0, 3))
    )
    candidates = {}
    for weight in legacy.UTILITY_WEIGHTS:
        score = np.asarray(utility, dtype=np.float32).copy()
        if quiet_test:
            ids = [str(graph.thread_id) for graph in quiet_test]
            score[test_quiet] = legacy.hybrid_quiet_score(
                score[test_quiet], probability, ids, weight
            )
        candidates[weight] = score
    return utility, gate, candidates


def append_oof_rows(
    graphs: list, event: str, baseline_raw: np.ndarray, gate: dict,
    candidates: dict[float, np.ndarray], baseline_rows: list[dict],
    candidate_rows: dict[float, list[dict]],
) -> None:
    quiet = legacy.recency_values(graphs) >= gate["threshold"]
    for index, graph in enumerate(graphs):
        shared = {
            "validation_event": event, "thread_id": str(graph.thread_id),
            "expert": "quiet" if quiet[index] else "active",
            "target": float(graph.preventable_y.item()),
        }
        baseline_rows.append({**shared, "raw_score": float(baseline_raw[index])})
        for weight in legacy.UTILITY_WEIGHTS:
            candidate_rows[weight].append({
                **shared, "raw_score": float(candidates[weight][index])
            })


def calibrate_oof(oof: pd.DataFrame, calibrators: dict) -> pd.DataFrame:
    result = oof.copy()
    quiet = result.expert.eq("quiet").to_numpy()
    result["score"] = legacy.apply_calibration(
        result.raw_score.to_numpy(float), quiet, calibrators
    )
    result["event_id"] = result.validation_event
    result["preventable_impact"] = np.expm1(result.target)
    return result


def nested_score_evidence(
    outer_event: str, baseline_oof: pd.DataFrame, baseline_calibrators: dict,
    hybrid_oof: pd.DataFrame, hybrid_calibrators: dict, selected_weight: float,
) -> pd.DataFrame:
    baseline = calibrate_oof(baseline_oof, baseline_calibrators)
    hybrid = calibrate_oof(hybrid_oof, hybrid_calibrators)
    keys = ["validation_event", "thread_id", "expert", "target", "event_id", "preventable_impact"]
    evidence = baseline[keys + ["raw_score", "score"]].rename(columns={
        "raw_score": "baseline_raw_score", "score": "baseline_calibrated_score",
    }).merge(
        hybrid[keys + ["raw_score", "score"]].rename(columns={
            "raw_score": "hybrid_raw_score", "score": "hybrid_calibrated_score",
        }), on=keys, how="inner", validate="one_to_one",
    )
    evidence["outer_event"] = outer_event
    evidence["selected_utility_weight"] = selected_weight
    if outer_event in set(evidence.event_id):
        raise ValueError("outer event leaked into nested hybrid evidence")
    return evidence


def run(mode: str) -> None:
    if mode not in {"smoke", "full"}:
        raise ValueError(mode)
    suffix = f"_corrected_quiet_tier_hybrid_{mode}"
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + suffix)
    output.mkdir(parents=True, exist_ok=False)
    model_params = dict(v5_common.MODEL_PARAMS)
    tier_trees = 50 if mode == "smoke" else 300
    if mode == "smoke":
        model_params.update(n_estimators=20, max_depth=5)
    legacy.TIER_TREES = tier_trees
    record = {
        "status": "running", "started_at": now(), "mode": mode,
        "experiment": "corrected-target active/quiet plus quiet-tier hybrid",
        "inputs": {
            "protected_dataset": str(PROTECTED_DATASET.resolve()),
            "corrected_dataset": str(CORRECTED_DATASET.resolve()),
            "protected_feature_materialization": str(legacy.MATERIALIZATION.resolve()),
        },
        "split": "eligible outer LOEO; all remaining events form nested inner OOF",
        "cutoff_seconds": 1800, "candidate_pool": "same protected v5 rumour-only identities",
        "model_params": model_params, "tier_trees": tier_trees,
        "source_weight": legacy.SOURCE_WEIGHT,
        "utility_weights": list(legacy.UTILITY_WEIGHTS), "budgets": list(legacy.BUDGETS),
        "seed": legacy.SEED,
        "target_change_only": "corrected all-traversable-roots preventable impact",
        "research_safety": (
            "Protected inputs are read-only. Corrected and protected graphs are compared field by "
            "field; only preventable target fields may differ. Tier outcomes are overlaid in memory. "
            "Gate, PCA, expert models, calibration, and lambda selection exclude the predicted event."
        ),
    }
    write_record(output, record)
    try:
        print(f"Starting {mode} corrected-target active/quiet + quiet-tier hybrid.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/8] Loading and auditing corrected target-only graph artifact...", flush=True)
        graphs, events, eligible, target_audit = load_and_audit_graphs()
        print(
            f"  target changes={target_audit['changed_targets']}; impact "
            f"{target_audit['legacy_impact_sum']} -> {target_audit['corrected_impact_sum']}",
            flush=True,
        )
        print("[2/8] Loading cutoff-safe node fields and validating strict-30 snapshots...", flush=True)
        depth_nodes, account_age = legacy.load_safe_nodes()
        offsets = legacy.load_node_offsets()
        # This validates every graph node timestamp and every snapshot edge before fitting.
        v5_common.validate_fold(
            [graph for graph in graphs if str(graph.event_id) != events[0]],
            [graph for graph in graphs if str(graph.event_id) == events[0]],
            depth_nodes, account_age, offsets,
        )
        print("[3/8] Building unchanged features and overlaying corrected tier outcomes in memory...", flush=True)
        frame, temporal_columns, embeddings = legacy.graph_inputs(graphs)
        frame, overlay_changes = overlay_corrected_targets(frame, graphs)

        outer_events = eligible[:1] if mode == "smoke" else eligible
        inner_limit = 1 if mode == "smoke" else None
        all_scores, all_curves, all_selections = [], [], []
        nested_evidence, gates, fold_details = [], [], []
        print("[4/8] Running corrected nested inner and held-out outer folds...", flush=True)
        for outer_index, outer_event in enumerate(outer_events, 1):
            outer_train = [graph for graph in graphs if str(graph.event_id) != outer_event]
            held_out = [graph for graph in graphs if str(graph.event_id) == outer_event]
            inner_events = sorted({str(graph.event_id) for graph in outer_train})
            if inner_limit:
                inner_events = [event for event in eligible if event != outer_event][:inner_limit]
            print(
                f"  outer {outer_index}/{len(outer_events)}: {outer_event}; "
                f"inner folds={len(inner_events)}", flush=True,
            )
            baseline_rows: list[dict] = []
            candidate_rows = {weight: [] for weight in legacy.UTILITY_WEIGHTS}
            for inner_index, inner_event in enumerate(inner_events, 1):
                print(f"    inner {inner_index}/{len(inner_events)}: {inner_event}", flush=True)
                inner_train = [
                    graph for graph in outer_train if str(graph.event_id) != inner_event
                ]
                validation = [
                    graph for graph in outer_train if str(graph.event_id) == inner_event
                ]
                baseline_raw, gate, candidates = raw_components(
                    inner_train, validation, depth_nodes, account_age, offsets,
                    frame, temporal_columns, embeddings,
                    legacy.SEED + outer_index * 1000 + inner_index, model_params,
                )
                append_oof_rows(
                    validation, inner_event, baseline_raw, gate, candidates,
                    baseline_rows, candidate_rows,
                )

            print("    fitting corrected expert calibrators and selecting utility weight...", flush=True)
            baseline_oof = legacy.calibrator_frame(baseline_rows)
            baseline_calibrators, baseline_details = legacy.fit_calibrators(baseline_oof)
            selection_rows, hybrid_calibrators, hybrid_details = [], {}, {}
            for weight in legacy.UTILITY_WEIGHTS:
                oof = legacy.calibrator_frame(candidate_rows[weight])
                calibrators, details = legacy.fit_calibrators(oof)
                hybrid_calibrators[weight] = calibrators
                hybrid_details[weight] = details
                scored = calibrate_oof(oof, calibrators)
                selection_rows.append({
                    "outer_event": outer_event, "utility_weight": weight,
                    "inner_quiet_high_overlap": legacy.quiet_high_overlap(oof),
                    "inner_oof_reduction_at_10": legacy.reduction_at_10(scored, "score"),
                })
            selection = pd.DataFrame(selection_rows).sort_values(
                ["inner_quiet_high_overlap", "inner_oof_reduction_at_10", "utility_weight"],
                ascending=[False, False, False], kind="stable",
            )
            selected_weight = float(selection.iloc[0].utility_weight)
            all_selections.append(selection)
            selected_hybrid_oof = legacy.calibrator_frame(candidate_rows[selected_weight])
            nested_evidence.append(nested_score_evidence(
                outer_event, baseline_oof, baseline_calibrators,
                selected_hybrid_oof, hybrid_calibrators[selected_weight], selected_weight,
            ))

            print(f"    selected weight={selected_weight:.2f}; fitting outer scorer...", flush=True)
            baseline_outer, outer_gate, outer_candidates = raw_components(
                outer_train, held_out, depth_nodes, account_age, offsets,
                frame, temporal_columns, embeddings,
                legacy.SEED + 100000 + outer_index, model_params,
            )
            outer_quiet = legacy.recency_values(held_out) >= outer_gate["threshold"]
            baseline_score = legacy.apply_calibration(
                baseline_outer, outer_quiet, baseline_calibrators
            )
            hybrid_score = legacy.apply_calibration(
                outer_candidates[selected_weight], outer_quiet,
                hybrid_calibrators[selected_weight],
            )
            scores = pd.DataFrame({
                "thread_id": [str(graph.thread_id) for graph in held_out],
                "event_id": [str(graph.event_id) for graph in held_out],
                "preventable_impact": [float(graph.preventable_impact.item()) for graph in held_out],
                "baseline_calibrated_score": baseline_score,
                "hybrid_calibrated_score": hybrid_score,
                "quiet": outer_quiet, "selected_utility_weight": selected_weight,
            })
            all_scores.append(scores)
            all_curves.extend(legacy.curve_rows(
                scores, "baseline_calibrated_score", "corrected_calibrated_active_quiet_rf"
            ))
            all_curves.extend(legacy.curve_rows(
                scores, "hybrid_calibrated_score", "corrected_quiet_tier_hybrid"
            ))
            gates.append({"outer_event": outer_event, **outer_gate})
            fold_details.append({
                "outer_event": outer_event, "selected_utility_weight": selected_weight,
                "baseline_calibrators": baseline_details,
                "hybrid_calibrators": hybrid_details[selected_weight],
            })
            pd.concat(all_scores, ignore_index=True).to_csv(
                output / "outer_scores_partial.csv", index=False
            )
            pd.concat(nested_evidence, ignore_index=True).to_csv(
                output / "nested_inner_oof_scores_partial.csv", index=False
            )

        print("[5/8] Computing corrected held-out budget summaries...", flush=True)
        scores_frame = pd.concat(all_scores, ignore_index=True)
        nested_frame = pd.concat(nested_evidence, ignore_index=True)
        curves = pd.DataFrame(all_curves)
        summary = curves.groupby(["model", "budget"], as_index=False).agg(
            eligible_events=("event_id", "nunique"),
            mean_oracle_efficiency=("oracle_efficiency", "mean"),
            mean_model_reduction=("model_reduction", "mean"),
        )
        comparison = summary.pivot(
            index="budget", columns="model",
            values=["mean_model_reduction", "mean_oracle_efficiency"],
        )
        comparison.columns = [f"{metric}_{model}" for metric, model in comparison.columns]
        comparison = comparison.reset_index()
        comparison["hybrid_minus_baseline_reduction"] = (
            comparison["mean_model_reduction_corrected_quiet_tier_hybrid"]
            - comparison["mean_model_reduction_corrected_calibrated_active_quiet_rf"]
        )
        comparison["hybrid_minus_baseline_efficiency"] = (
            comparison["mean_oracle_efficiency_corrected_quiet_tier_hybrid"]
            - comparison["mean_oracle_efficiency_corrected_calibrated_active_quiet_rf"]
        )

        print("[6/8] Validating nested/outer score identities and split isolation...", flush=True)
        if scores_frame.duplicated(["thread_id", "event_id"]).any():
            raise ValueError("duplicate corrected outer score")
        if nested_frame.duplicated(["outer_event", "event_id", "thread_id"]).any():
            raise ValueError("duplicate corrected nested score")
        if (nested_frame.outer_event == nested_frame.event_id).any():
            raise ValueError("outer event appeared in corrected nested scores")
        numeric = scores_frame[[
            "preventable_impact", "baseline_calibrated_score", "hybrid_calibrated_score"
        ]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("non-finite corrected outer score")

        print("[7/8] Writing corrected outer scores and reusable nested OOF...", flush=True)
        scores_frame.to_csv(output / "outer_scores.csv", index=False)
        nested_frame.to_csv(output / "nested_inner_oof_scores.csv", index=False)
        curves.to_csv(output / "per_event_budget_curve.csv", index=False)
        summary.to_csv(output / "eligible_event_budget_summary.csv", index=False)
        comparison.to_csv(output / "hybrid_vs_baseline_budget_comparison.csv", index=False)
        pd.concat(all_selections, ignore_index=True).to_csv(
            output / "inner_weight_selection.csv", index=False
        )
        pd.DataFrame(gates).to_csv(output / "outer_gate_details.csv", index=False)

        print("[8/8] Finalizing complete run record...", flush=True)
        result = {
            "mode": mode, "target_audit": target_audit,
            "budget_summary": summary.to_dict(orient="records"),
            "interpretation_boundary": (
                "Corrected-target candidate; prior legacy hybrid remains a separate reference."
            ),
        }
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        record.update({
            "status": "complete", "completed_at": now(),
            "target_audit": target_audit, "tier_frame_overlay_changes": overlay_changes,
            "events": events, "eligible_outer_events": outer_events,
            "inner_folds_per_outer": inner_limit or (len(events) - 1),
            "fold_details": fold_details,
            "output_files": [
                "outer_scores_partial.csv", "nested_inner_oof_scores_partial.csv",
                "outer_scores.csv", "nested_inner_oof_scores.csv",
                "per_event_budget_curve.csv", "eligible_event_budget_summary.csv",
                "hybrid_vs_baseline_budget_comparison.csv", "inner_weight_selection.csv",
                "outer_gate_details.csv", "result.json", "run_record.json",
            ],
        })
        write_record(output, record)
        print(f"SUCCESS: {mode} corrected quiet-tier hybrid saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({
            "status": "failed", "failed_at": now(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "output_files": sorted(path.name for path in output.iterdir()),
        })
        write_record(output, record)
        print(f"FAILURE: corrected quiet-tier hybrid preserved at: {output.resolve()}", flush=True)
        raise
