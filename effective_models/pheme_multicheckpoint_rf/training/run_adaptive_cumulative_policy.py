"""Runner shared by smoke and full nested adaptive cumulative policy studies."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import pandas as pd

import adaptive_policy_common as policy
import multicheckpoint_model_common as model_common


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL_ROOT = HERE.parent
OUT_ROOT = MODEL_ROOT / "experiments"
MATERIALIZATION = (
    MODEL_ROOT / "experiments"
    / "20260903_153526_240803_multicheckpoint_materialization_full"
)
OUTER_OOF = (
    MODEL_ROOT / "experiments"
    / "20260903_153740_370305_multicheckpoint_models_full"
    / "oof_predictions.csv"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run(mode: str) -> None:
    if mode not in {"smoke", "full"}:
        raise ValueError(mode)
    suffix = f"_adaptive_cumulative_policy_{mode}"
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + suffix)
    output.mkdir(parents=True, exist_ok=False)
    model_params = dict(model_common.MODEL_PARAMS)
    if mode == "smoke":
        model_params.update(n_estimators=20, max_depth=5, n_jobs=1)
    record = {
        "status": "running",
        "started_at": now(),
        "mode": mode,
        "model": "pooled elapsed-time-aware cumulative Random Forest",
        "model_params": model_params,
        "input_materialization": str(MATERIALIZATION.resolve()),
        "input_outer_oof": str(OUTER_OOF.resolve()),
        "split": "outer event held out; policy selected from fresh inner-event OOF",
        "checkpoints_seconds": list(policy.CHECKPOINTS),
        "candidate_max_budgets": list(policy.MAX_BUDGETS),
        "candidate_min_predicted_impacts": list(policy.IMPACT_THRESHOLDS),
        "target_reductions": list(policy.TARGET_REDUCTIONS),
        "cost_weights": list(policy.COST_WEIGHTS),
        "target_worst_event_fraction": policy.TARGET_WORST_EVENT_FRACTION,
        "seed": model_params["random_state"],
        "research_safety": (
            "For each outer event, every inner prediction is produced by a model that excludes "
            "both the outer and inner event. Threshold/cap selection uses inner outcomes only; "
            "the frozen selected policy is then replayed once on saved outer-event OOF scores."
        ),
    }
    write_record(output, record)
    try:
        print(f"Starting {mode} nested adaptive cumulative policy study.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/7] Loading fixed full materialization and saved outer OOF scores...", flush=True)
        frame, schema = model_common.load_materialization(MATERIALIZATION)
        columns = model_common.family_columns(schema, "cumulative")
        outer_scores = policy.load_outer_oof(OUTER_OOF)
        frame_ids = set(zip(frame.thread_id, frame.event_id, frame.checkpoint_sec.astype(int)))
        score_ids = set(zip(
            outer_scores.thread_id, outer_scores.event_id,
            outer_scores.checkpoint_sec.astype(int),
        ))
        if frame_ids != score_ids:
            raise ValueError("materialization and outer OOF identities differ")
        events = sorted(frame.event_id.unique())
        counts = frame[["thread_id", "event_id"]].drop_duplicates().groupby("event_id").size()
        eligible = sorted(counts.loc[counts.ge(100)].index)
        if mode == "smoke":
            outer_events = eligible[:1]
            smoke_inner_limit = 3
        else:
            outer_events = eligible
            smoke_inner_limit = None
        record.update({
            "all_events": events,
            "eligible_outer_events": outer_events,
            "feature_count": len(columns),
            "rows": len(frame),
            "threads": frame.thread_id.nunique(),
        })
        write_record(output, record)

        print("[2/7] Validating outcome boundaries and nested split identities...", flush=True)
        if not (
            frame.dynamic_preventable_impact.le(frame.future_growth).all()
            and frame.dynamic_preventable_impact.ge(0).all()
        ):
            raise ValueError("invalid dynamic preventable target boundary")

        inner_prediction_chunks = []
        inner_metric_chunks = []
        inner_grid_chunks = []
        selected_config_chunks = []
        outer_metric_chunks = []
        outer_selected_chunks = []
        baseline_rows = []
        print("[3/7] Building fresh nested inner-event OOF predictions...", flush=True)
        for outer_index, outer_event in enumerate(outer_events, 1):
            inner_events = [event for event in events if event != outer_event]
            if smoke_inner_limit is not None:
                inner_events = inner_events[:smoke_inner_limit]
            print(
                f"  outer event {outer_index}/{len(outer_events)}: {outer_event}; "
                f"inner folds={len(inner_events)}",
                flush=True,
            )
            inner_oof = policy.fit_inner_oof(
                frame=frame,
                outer_event=outer_event,
                inner_events=inner_events,
                columns=columns,
                model_params=model_params,
                progress=lambda message: print(message, flush=True),
            )
            inner_prediction_chunks.append(inner_oof)
            print("    evaluating 50 predeclared threshold/cap configurations", flush=True)
            inner_metrics, _ = policy.evaluate_configs(
                inner_oof.drop(columns="outer_event")
            )
            inner_metrics["outer_event"] = outer_event
            inner_metric_chunks.append(inner_metrics)
            summary = policy.summarize_grid(inner_metrics)
            summary["outer_event"] = outer_event
            inner_grid_chunks.append(summary)
            selected_configs = policy.select_policies(summary.drop(columns="outer_event"))
            selected_configs["outer_event"] = outer_event
            selected_config_chunks.append(selected_configs)

            frozen_metrics, frozen_selected = policy.apply_selected_policies(
                outer_scores=outer_scores,
                selected_configs=selected_configs,
                outer_event=outer_event,
            )
            outer_metric_chunks.append(frozen_metrics)
            if len(frozen_selected):
                outer_selected_chunks.append(frozen_selected)

            outer_event_scores = outer_scores.loc[outer_scores.event_id.eq(outer_event)]
            fixed = policy.fixed_balanced_select(outer_event_scores)
            horizon = float(
                outer_event_scores.loc[
                    outer_event_scores.checkpoint_sec.eq(policy.CHECKPOINTS[0]),
                    "future_growth",
                ].sum()
            )
            blocked = float(fixed.dynamic_preventable_impact.sum())
            baseline_rows.append({
                "outer_event": outer_event,
                "policy_name": "fixed_balanced_cumulative_50",
                "config_id": "quotas_9_9_8_8_8_8",
                "max_budget": 50,
                "min_predicted_impact": 0.0,
                "interventions": 50,
                "positive_interventions": int(fixed.dynamic_preventable_impact.gt(0).sum()),
                "blocked_impact": blocked,
                "horizon_future_at_10m": horizon,
                "horizon_reduction": blocked / horizon if horizon else 0.0,
                "inner_mean_interventions": float("nan"),
                "inner_mean_horizon_reduction": float("nan"),
                "inner_worst_event_horizon_reduction": float("nan"),
                "used_fallback": False,
            })
            pd.concat(inner_prediction_chunks, ignore_index=True).to_csv(
                output / "nested_inner_oof_predictions_partial.csv", index=False
            )
            pd.concat(outer_metric_chunks + [pd.DataFrame(baseline_rows)], ignore_index=True).to_csv(
                output / "outer_policy_metrics_partial.csv", index=False
            )

        print("[4/7] Aggregating frozen outer-event adaptive policy results...", flush=True)
        inner_predictions = pd.concat(inner_prediction_chunks, ignore_index=True)
        inner_metrics = pd.concat(inner_metric_chunks, ignore_index=True)
        inner_grid = pd.concat(inner_grid_chunks, ignore_index=True)
        selected_configs = pd.concat(selected_config_chunks, ignore_index=True)
        outer_metrics = pd.concat(
            outer_metric_chunks + [pd.DataFrame(baseline_rows)], ignore_index=True
        )
        outer_selected = (
            pd.concat(outer_selected_chunks, ignore_index=True)
            if outer_selected_chunks else pd.DataFrame()
        )
        macro = outer_metrics.groupby("policy_name", as_index=False).agg(
            outer_events=("outer_event", "nunique"),
            mean_interventions=("interventions", "mean"),
            max_interventions=("interventions", "max"),
            total_interventions=("interventions", "sum"),
            total_positive_interventions=("positive_interventions", "sum"),
            total_blocked_impact=("blocked_impact", "sum"),
            mean_horizon_reduction=("horizon_reduction", "mean"),
            worst_event_horizon_reduction=("horizon_reduction", "min"),
            fallback_outer_folds=("used_fallback", "sum"),
        ).sort_values(
            ["mean_interventions", "mean_horizon_reduction", "policy_name"],
            ascending=[True, False, True], kind="stable",
        )
        pareto = policy.pareto_frontier(macro)

        print("[5/7] Checking policy caps, uniqueness, and outer isolation...", flush=True)
        if (outer_metrics.interventions > outer_metrics.max_budget).any():
            raise ValueError("outer policy exceeded its selected cap")
        if len(outer_selected) and outer_selected.duplicated(
            ["outer_event", "policy_name", "thread_id"]
        ).any():
            raise ValueError("thread selected more than once by an outer policy")
        for outer_event in outer_events:
            inner_part = inner_predictions.loc[inner_predictions.outer_event.eq(outer_event)]
            if outer_event in set(inner_part.event_id):
                raise ValueError(f"outer leakage in saved inner predictions: {outer_event}")

        print("[6/7] Writing complete nested selection and Pareto evidence...", flush=True)
        inner_predictions.to_csv(output / "nested_inner_oof_predictions.csv", index=False)
        inner_metrics.to_csv(output / "inner_policy_metrics.csv", index=False)
        inner_grid.to_csv(output / "inner_policy_grid.csv", index=False)
        selected_configs.to_csv(output / "selected_configs.csv", index=False)
        outer_metrics.to_csv(output / "outer_policy_metrics.csv", index=False)
        outer_selected.to_csv(output / "outer_selected_threads.csv", index=False)
        macro.to_csv(output / "adaptive_policy_macro.csv", index=False)
        pareto.to_csv(output / "adaptive_policy_pareto.csv", index=False)

        print("[7/7] Completing result and run record...", flush=True)
        result = {
            "mode": mode,
            "eligible_outer_events": outer_events,
            "adaptive_policy_macro": macro.to_dict(orient="records"),
            "adaptive_policy_pareto": pareto.to_dict(orient="records"),
            "selection_boundary": (
                "Each outer policy configuration is selected only from fresh inner-event OOF "
                "predictions that exclude the outer event. Outer outcomes are evaluation-only."
            ),
            "interpretation": (
                "Target policies minimize validation intervention count subject to macro and "
                "worst-event reduction constraints. Cost policies optimize a predeclared "
                "normalized reduction-minus-action-cost objective."
            ),
        }
        (output / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        record.update({
            "status": "complete",
            "completed_at": now(),
            "inner_fits": sum(
                len([event for event in events if event != outer])
                if smoke_inner_limit is None else smoke_inner_limit
                for outer in outer_events
            ),
            "selected_policy_rows": len(selected_configs),
            "outer_metric_rows": len(outer_metrics),
            "output_files": [
                "nested_inner_oof_predictions_partial.csv",
                "outer_policy_metrics_partial.csv",
                "nested_inner_oof_predictions.csv",
                "inner_policy_metrics.csv",
                "inner_policy_grid.csv",
                "selected_configs.csv",
                "outer_policy_metrics.csv",
                "outer_selected_threads.csv",
                "adaptive_policy_macro.csv",
                "adaptive_policy_pareto.csv",
                "result.json",
                "run_record.json",
            ],
        })
        write_record(output, record)
        print(
            f"SUCCESS: {mode} adaptive cumulative policy saved to: {output.resolve()}",
            flush=True,
        )
    except BaseException as error:
        record.update({
            "status": "failed",
            "failed_at": now(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "output_files": sorted(path.name for path in output.iterdir()),
        })
        write_record(output, record)
        print(
            f"FAILURE: {mode} adaptive cumulative policy preserved at: {output.resolve()}",
            flush=True,
        )
        raise
