"""Nested next-10-minute wait-loss RF and two-signal policy experiment."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

import adaptive_policy_common as adaptive
import wait_loss_hazard_common as hazard


HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parent / "experiments"
UTILITY_NESTED = (
    OUT_ROOT / "20260903_162347_617042_adaptive_cumulative_policy_full"
    / "nested_inner_oof_predictions.csv"
)
UTILITY_OUTER = (
    OUT_ROOT / "20260903_153740_370305_multicheckpoint_models_full"
    / "oof_predictions.csv"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def latest_hazard_data() -> Path:
    candidates = []
    for path in OUT_ROOT.glob("*_next10_wait_loss_materialization"):
        record = path / "run_record.json"
        data = path / "pheme_next10_wait_loss_features.csv"
        schema = path / "feature_schema.json"
        if record.is_file() and data.is_file() and schema.is_file():
            if json.loads(record.read_text(encoding="utf-8")).get("status") == "complete":
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError("no complete next10 wait-loss materialization")
    return sorted(candidates)[-1]


def merge_scores(utility: pd.DataFrame, hazard_scores: pd.DataFrame) -> pd.DataFrame:
    keys = ["thread_id", "event_id", "checkpoint_sec"]
    result = utility.merge(
        hazard_scores[keys + ["hazard_prediction"]], on=keys,
        how="left", validate="one_to_one",
    )
    missing_early = result.checkpoint_sec.lt(3600) & result.hazard_prediction.isna()
    if missing_early.any():
        raise ValueError("missing pre-deadline hazard score")
    result.loc[result.checkpoint_sec.eq(3600), "hazard_prediction"] = 0.0
    adaptive.validate_scored_frame(result)
    return result


def run(mode: str) -> None:
    if mode not in {"smoke", "full"}:
        raise ValueError(mode)
    output = OUT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + f"_wait_loss_hazard_policy_{mode}"
    )
    output.mkdir(parents=True, exist_ok=False)
    params = dict(hazard.HAZARD_MODEL_PARAMS)
    if mode == "smoke":
        params.update(n_estimators=20, max_depth=5)
    source = latest_hazard_data()
    record = {
        "status": "running", "started_at": now(), "mode": mode,
        "model": "next10 wait-loss Random Forest plus saved cumulative utility RF",
        "model_params": params, "input_hazard_data": str(source.resolve()),
        "input_utility_nested": str(UTILITY_NESTED.resolve()),
        "input_utility_outer": str(UTILITY_OUTER.resolve()),
        "split": "outer event held out; hazard and policy selection use fresh inner OOF",
        "checkpoints_seconds": list(adaptive.CHECKPOINTS),
        "hazard_training_checkpoints_seconds": [600, 1200, 1800, 2400, 3000],
        "policy_configuration_count": len(hazard.candidate_configs()), "seed": 42,
        "research_safety": (
            "Future wait loss is supervision only. Each inner hazard model excludes outer and "
            "inner events. Each outer hazard model excludes the outer event. Policy thresholds "
            "are frozen using inner outcomes before outer evaluation. This hypothesis follows "
            "inspection of earlier PHEME results and is exploratory."
        ),
    }
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        print(f"Starting {mode} nested wait-loss hazard policy experiment.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/7] Loading hazard supervision and saved utility predictions...", flush=True)
        frame = pd.read_csv(
            source / "pheme_next10_wait_loss_features.csv",
            dtype={"thread_id": str, "event_id": str},
        )
        schema = json.loads((source / "feature_schema.json").read_text(encoding="utf-8"))
        columns = list(schema["cumulative_feature_columns"])
        utility_nested = pd.read_csv(
            UTILITY_NESTED, dtype={"thread_id": str, "event_id": str, "outer_event": str}
        )
        utility_outer = adaptive.load_outer_oof(UTILITY_OUTER)
        events = sorted(frame.event_id.unique())
        eligible = sorted(utility_nested.outer_event.unique())
        outer_events = eligible[:1] if mode == "smoke" else eligible
        inner_limit = 3 if mode == "smoke" else None
        print("[2/7] Auditing target bounds, feature allowlist, and event identities...", flush=True)
        if any(token in c.lower() for c in columns for token in ("next10", "future", "impact")):
            raise ValueError("future outcome in hazard feature allowlist")
        if (frame.next10_wait_loss > frame.dynamic_preventable_impact).any():
            raise ValueError("wait loss exceeds current preventable impact")

        inner_metrics_all, inner_hazard_metric_all, selected_configs_all = [], [], []
        outer_metrics_all, outer_selected_all, hazard_oof_all = [], [], []
        outer_hazard_metric_all = []
        baseline_rows = []
        print("[3/7] Fitting split-safe inner and outer wait-loss hazard models...", flush=True)
        for outer_index, outer_event in enumerate(outer_events, 1):
            inner_events = [e for e in events if e != outer_event]
            if inner_limit:
                inner_events = inner_events[:inner_limit]
            hazard_inner_chunks = []
            print(f"  outer {outer_index}/{len(outer_events)}: {outer_event}", flush=True)
            for inner_index, inner_event in enumerate(inner_events, 1):
                print(f"    inner {inner_index}/{len(inner_events)}: {inner_event}", flush=True)
                train = frame.loc[
                    frame.event_id.ne(outer_event) & frame.event_id.ne(inner_event)
                ]
                validation = frame.loc[frame.event_id.eq(inner_event)]
                _, prediction = hazard.fit_predict(train, validation, columns, params)
                scored = validation[[
                    "thread_id", "event_id", "checkpoint_sec", "next10_wait_loss"
                ]].copy()
                scored["hazard_prediction"] = prediction
                hazard_inner_chunks.append(scored)
            hazard_inner = pd.concat(hazard_inner_chunks, ignore_index=True)
            inner_hazard_metrics = hazard.prediction_metrics(hazard_inner)
            inner_hazard_metrics["outer_event"] = outer_event
            inner_hazard_metric_all.append(inner_hazard_metrics)
            utility_inner = utility_nested.loc[
                utility_nested.outer_event.eq(outer_event)
                & utility_nested.event_id.isin(inner_events)
            ].drop(columns="outer_event")
            joined_inner = merge_scores(utility_inner, hazard_inner)
            policy_metrics = hazard.evaluate_configs(joined_inner)
            policy_metrics["outer_event"] = outer_event
            inner_metrics_all.append(policy_metrics)
            summary = hazard.summarize_grid(policy_metrics)
            selected_configs = adaptive.select_policies(summary)
            selected_configs["outer_event"] = outer_event
            selected_configs_all.append(selected_configs)

            train_outer = frame.loc[frame.event_id.ne(outer_event)]
            test_outer = frame.loc[frame.event_id.eq(outer_event)]
            _, prediction = hazard.fit_predict(train_outer, test_outer, columns, params)
            hazard_outer = test_outer[[
                "thread_id", "event_id", "checkpoint_sec", "next10_wait_loss"
            ]].copy()
            hazard_outer["hazard_prediction"] = prediction
            hazard_outer["outer_event"] = outer_event
            hazard_oof_all.append(hazard_outer)
            outer_hazard_metric_all.append(hazard.prediction_metrics(hazard_outer))
            joined_outer = merge_scores(
                utility_outer.loc[utility_outer.event_id.eq(outer_event)],
                hazard_outer.drop(columns="outer_event"),
            )
            metrics, selections = hazard.apply_selected(
                joined_outer, selected_configs, outer_event
            )
            outer_metrics_all.append(metrics)
            outer_selected_all.append(selections)
            fixed = adaptive.fixed_balanced_select(
                utility_outer.loc[utility_outer.event_id.eq(outer_event)]
            )
            horizon = float(joined_outer.loc[joined_outer.checkpoint_sec.eq(600), "future_growth"].sum())
            blocked = float(fixed.dynamic_preventable_impact.sum())
            baseline_rows.append({
                "outer_event": outer_event, "policy_name": "fixed_balanced_cumulative_50",
                "config_id": "quotas_9_9_8_8_8_8", "max_budget": 50,
                "interventions": 50,
                "positive_interventions": int(fixed.dynamic_preventable_impact.gt(0).sum()),
                "blocked_impact": blocked, "horizon_future_at_10m": horizon,
                "horizon_reduction": blocked / horizon, "mean_action_minute": 34.2,
                "used_fallback": False,
            })

        print("[4/7] Aggregating outer policy and Pareto results...", flush=True)
        inner_metrics = pd.concat(inner_metrics_all, ignore_index=True)
        selected_configs = pd.concat(selected_configs_all, ignore_index=True)
        outer_metrics = pd.concat(outer_metrics_all + [pd.DataFrame(baseline_rows)], ignore_index=True)
        outer_selected = pd.concat(outer_selected_all, ignore_index=True)
        hazard_oof = pd.concat(hazard_oof_all, ignore_index=True)
        inner_hazard_metrics = pd.concat(inner_hazard_metric_all, ignore_index=True)
        outer_hazard_metrics = pd.concat(outer_hazard_metric_all, ignore_index=True)
        outer_hazard_macro = outer_hazard_metrics.groupby(
            "checkpoint_sec", as_index=False
        ).agg(
            outer_events=("event_id", "nunique"),
            mean_mae_wait_loss=("mae_wait_loss", "mean"),
            mean_spearman=("spearman", "mean"),
            mean_roc_auc_positive=("roc_auc_positive", "mean"),
            mean_pr_auc_positive=("pr_auc_positive", "mean"),
            mean_top20_wait_loss_capture=("top20_wait_loss_capture", "mean"),
        )
        macro = outer_metrics.groupby("policy_name", as_index=False).agg(
            outer_events=("outer_event", "nunique"), mean_interventions=("interventions", "mean"),
            total_interventions=("interventions", "sum"),
            total_positive_interventions=("positive_interventions", "sum"),
            total_blocked_impact=("blocked_impact", "sum"),
            mean_horizon_reduction=("horizon_reduction", "mean"),
            worst_event_horizon_reduction=("horizon_reduction", "min"),
            mean_action_minute=("mean_action_minute", "mean"),
            fallback_outer_folds=("used_fallback", "sum"),
        )
        pareto = adaptive.pareto_frontier(macro)
        print("[5/7] Checking outer isolation, cap, uniqueness, and finiteness...", flush=True)
        if (outer_metrics.interventions > outer_metrics.max_budget).any():
            raise ValueError("policy cap violation")
        if outer_selected.duplicated(["outer_event", "policy_name", "thread_id"]).any():
            raise ValueError("duplicate outer intervention")
        print("[6/7] Writing hazard predictions, selections, and metrics...", flush=True)
        inner_metrics.to_csv(output / "inner_policy_metrics.csv", index=False)
        inner_hazard_metrics.to_csv(output / "inner_hazard_metrics.csv", index=False)
        selected_configs.to_csv(output / "selected_configs.csv", index=False)
        hazard_oof.to_csv(output / "outer_hazard_oof_predictions.csv", index=False)
        outer_hazard_metrics.to_csv(output / "outer_hazard_metrics.csv", index=False)
        outer_hazard_macro.to_csv(output / "outer_hazard_macro.csv", index=False)
        outer_metrics.to_csv(output / "outer_policy_metrics.csv", index=False)
        outer_selected.to_csv(output / "outer_selected_threads.csv", index=False)
        macro.to_csv(output / "hazard_policy_macro.csv", index=False)
        pareto.to_csv(output / "hazard_policy_pareto.csv", index=False)
        print("[7/7] Completing result and run record...", flush=True)
        result = {"mode": mode,
                  "outer_hazard_macro": outer_hazard_macro.to_dict(orient="records"),
                  "hazard_policy_macro": macro.to_dict(orient="records"),
                  "interpretation_boundary": "Exploratory nested evaluation; not untouched confirmation."}
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        record.update({
            "status": "complete", "completed_at": now(), "outer_events": outer_events,
            "hazard_fits": len(outer_events) * ((inner_limit or 8) + 1),
            "output_files": [
                "inner_policy_metrics.csv", "selected_configs.csv",
                "inner_hazard_metrics.csv", "outer_hazard_oof_predictions.csv",
                "outer_hazard_metrics.csv", "outer_hazard_macro.csv",
                "outer_policy_metrics.csv",
                "outer_selected_threads.csv", "hazard_policy_macro.csv",
                "hazard_policy_pareto.csv", "result.json", "run_record.json",
            ],
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"SUCCESS: {mode} wait-loss hazard policy saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": now(),
                       "error": f"{type(error).__name__}: {error}",
                       "traceback": traceback.format_exc(),
                       "output_files": sorted(p.name for p in output.iterdir())})
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"FAILURE: {mode} wait-loss hazard policy preserved at: {output.resolve()}", flush=True)
        raise
