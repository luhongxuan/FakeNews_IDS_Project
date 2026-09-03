"""Join saved acceleration/utility OOF and evaluate a two-threshold quiet gate."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

import quiet_acceleration_qualification_common as acceleration_common
import quiet_dual_threshold_common as common


HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parent / "experiments"
ACCELERATION = OUT_ROOT / "20260903_194947_531733_quiet_acceleration_rf_full"
UTILITY_NESTED = (
    OUT_ROOT / "20260903_162347_617042_adaptive_cumulative_policy_full"
    / "nested_inner_oof_predictions.csv"
)
UTILITY_OUTER = (
    OUT_ROOT / "20260903_153740_370305_multicheckpoint_models_full"
    / "oof_predictions.csv"
)
OUTCOMES = (
    OUT_ROOT / "20260903_183904_151619_quiet_acceleration_materialization"
    / "pheme_quiet_acceleration_features.csv"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def join_scores(acceleration: pd.DataFrame, utility: pd.DataFrame, inner: bool) -> pd.DataFrame:
    keys = ["thread_id", "event_id", "checkpoint_sec"]
    if inner:
        keys.insert(0, "outer_event")
    utility = utility[keys + ["prediction", "dynamic_preventable_impact", "future_growth"]].rename(
        columns={"prediction": "utility_prediction"}
    )
    acceleration = acceleration.rename(columns={"prediction": "acceleration_prediction"})
    result = acceleration.merge(utility, on=keys, how="left", validate="one_to_one")
    if result.utility_prediction.isna().any():
        raise ValueError("quiet acceleration rows do not align with cumulative utility OOF")
    result["predicted_impact"] = np.maximum(
        np.expm1(result.utility_prediction.to_numpy(dtype=float)), 0.0
    )
    common.validate_scored(result, inner=inner)
    return result


def main() -> None:
    output = OUT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + "_saved_quiet_dual_threshold_qualification"
    )
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "started_at": now(),
        "purpose": "threshold-only quiet qualification using acceleration and future-impact scores",
        "inputs": {
            "acceleration_oof": str(ACCELERATION.resolve()),
            "utility_inner_oof": str(UTILITY_NESTED.resolve()),
            "utility_outer_oof": str(UTILITY_OUTER.resolve()),
            "outcomes": str(OUTCOMES.resolve()),
        },
        "split": "configuration selected from matching nested inner OOF for each outer event",
        "checkpoints_seconds": list(common.CHECKPOINTS),
        "acceleration_quantiles": list(common.ACCELERATION_QUANTILES),
        "predicted_impact_thresholds": list(common.UTILITY_THRESHOLDS),
        "target_quiet_impact_capture": list(common.CAPTURE_TARGETS),
        "selection_rule": "both thresholds must pass; select all qualifiers; no Top-K; no repeats",
        "research_safety": (
            "No model fitting. Both saved scores are event-separated. Configurations use inner "
            "future-impact outcomes only; outer outcomes and Oracle membership are evaluation-only."
        ),
        "interpretation_boundary": (
            "Post-result exploratory analysis on already-consumed PHEME events."
        ),
    }
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        print("Starting saved quiet dual-threshold qualification.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/7] Loading acceleration and cumulative utility OOF predictions...", flush=True)
        inner_acc = pd.read_csv(
            ACCELERATION / "inner_quiet_acceleration_oof.csv",
            dtype={"thread_id": str, "event_id": str, "outer_event": str},
        )
        outer_acc = pd.read_csv(
            ACCELERATION / "outer_quiet_acceleration_oof.csv",
            dtype={"thread_id": str, "event_id": str},
        )
        inner_utility = pd.read_csv(
            UTILITY_NESTED, dtype={"thread_id": str, "event_id": str, "outer_event": str}
        )
        outer_utility = pd.read_csv(
            UTILITY_OUTER, dtype={"thread_id": str, "event_id": str}
        )
        outer_utility = outer_utility.loc[outer_utility.family.eq("cumulative")].copy()
        outcomes = pd.read_csv(OUTCOMES, dtype={"thread_id": str, "event_id": str})
        inner_acc = inner_acc.loc[inner_acc.checkpoint_sec.isin(common.CHECKPOINTS)].copy()
        outer_acc = outer_acc.loc[outer_acc.checkpoint_sec.isin(common.CHECKPOINTS)].copy()
        inner_utility = inner_utility.loc[inner_utility.checkpoint_sec.isin(common.CHECKPOINTS)].copy()
        outer_utility = outer_utility.loc[outer_utility.checkpoint_sec.isin(common.CHECKPOINTS)].copy()

        print("[2/7] Auditing exact fold/key alignment and joining both scores...", flush=True)
        inner = join_scores(inner_acc, inner_utility, inner=True)
        outer = join_scores(outer_acc, outer_utility, inner=False)
        outcomes = outcomes.loc[outcomes.checkpoint_sec.isin(common.CHECKPOINTS)].copy()
        if set(inner.outer_event) != set(outer.event_id):
            raise ValueError("nested inner and outer event coverage differ")

        print("[3/7] Evaluating 80 inner-only dual-threshold configurations per outer fold...", flush=True)
        inner_event_rows, inner_macro_rows, selected_config_rows = [], [], []
        for index, outer_event in enumerate(sorted(outer.event_id.unique()), 1):
            print(f"  outer event {index}/7: {outer_event}; configs=80", flush=True)
            fold = inner.loc[inner.outer_event.eq(outer_event)].copy()
            per_event, macro = common.evaluate_inner_grid(fold)
            per_event["outer_event"] = outer_event
            macro["outer_event"] = outer_event
            selected = common.select_capture_policies(macro)
            selected["outer_event"] = outer_event
            inner_event_rows.append(per_event)
            inner_macro_rows.append(macro)
            selected_config_rows.append(selected)
        inner_per_event = pd.concat(inner_event_rows, ignore_index=True)
        inner_macro = pd.concat(inner_macro_rows, ignore_index=True)
        selected_configs = pd.concat(selected_config_rows, ignore_index=True)

        print("[4/7] Replaying frozen dual thresholds on held-out outer events...", flush=True)
        metric_rows, selected_rows = [], []
        for outer_event in sorted(outer.event_id.unique()):
            event = outer.loc[outer.event_id.eq(outer_event)].copy()
            all_event = outcomes.loc[outcomes.event_id.eq(outer_event)].copy()
            oracle_union = acceleration_common.oracle_top50_union(all_event)
            quiet_oracle_union = oracle_union & set(event.thread_id)
            reference = common.apply_config(
                event, common.DualThresholdConfig(0.0, 0.0),
                {checkpoint: -np.inf for checkpoint in common.CHECKPOINTS},
                policy_name="all_quiet_reference",
            )
            reference_impact = float(reference.dynamic_preventable_impact.sum())
            horizon = float(
                all_event.loc[all_event.checkpoint_sec.eq(1200), "future_growth"].sum()
            )
            for row in selected_configs.loc[
                selected_configs.outer_event.eq(outer_event)
            ].itertuples(index=False):
                config = common.DualThresholdConfig(
                    float(row.acceleration_quantile), float(row.min_predicted_impact)
                )
                cutoffs = {
                    1200: float(row.acceleration_threshold_20m),
                    1800: float(row.acceleration_threshold_30m),
                }
                chosen = common.apply_config(event, config, cutoffs, str(row.policy_name))
                ceiling = acceleration_common.same_count_quiet_oracle(event, chosen)
                blocked = float(chosen.dynamic_preventable_impact.sum())
                ceiling_impact = float(ceiling.dynamic_preventable_impact.sum())
                chosen_ids = set(chosen.thread_id)
                hits = len(chosen_ids & quiet_oracle_union)
                metric_rows.append({
                    "outer_event": outer_event, "policy_name": row.policy_name,
                    "target_quiet_impact_capture": row.target_quiet_impact_capture,
                    "config_id": config.config_id,
                    "acceleration_quantile": config.acceleration_quantile,
                    "min_predicted_impact": config.min_predicted_impact,
                    "acceleration_threshold_20m": cutoffs[1200],
                    "acceleration_threshold_30m": cutoffs[1800],
                    "interventions": len(chosen),
                    "interventions_at_20m": int(chosen.action_checkpoint_sec.eq(1200).sum()),
                    "interventions_at_30m": int(chosen.action_checkpoint_sec.eq(1800).sum()),
                    "positive_interventions": int(chosen.dynamic_preventable_impact.gt(0).sum()),
                    "blocked_impact": blocked,
                    "horizon_future_at_20m": horizon,
                    "horizon_reduction": blocked / horizon if horizon else 0.0,
                    "all_quiet_reference_impact": reference_impact,
                    "quiet_impact_capture": blocked / reference_impact if reference_impact else 0.0,
                    "quiet_oracle_top50_union": len(quiet_oracle_union),
                    "oracle_top50_hits": hits,
                    "quiet_oracle_top50_recall": hits / len(quiet_oracle_union)
                    if quiet_oracle_union else 0.0,
                    "same_count_quiet_oracle_impact": ceiling_impact,
                    "same_count_oracle_efficiency": blocked / ceiling_impact
                    if ceiling_impact else 0.0,
                    "inner_mean_interventions": row.mean_interventions,
                    "inner_mean_quiet_impact_capture": row.mean_quiet_impact_capture,
                })
                if len(chosen):
                    selected_rows.append(chosen.assign(outer_event=outer_event)[[
                        "outer_event", "policy_name", "config_id", "thread_id", "event_id",
                        "action_checkpoint_sec", "acceleration_threshold",
                        "acceleration_prediction", "predicted_positive_acceleration",
                        "min_predicted_impact", "utility_prediction", "predicted_impact",
                        "dynamic_preventable_impact", "future_growth",
                    ]])

        print("[5/7] Summarizing held-out intervention efficiency and Oracle coverage...", flush=True)
        metrics = pd.DataFrame(metric_rows)
        selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
        macro = metrics.groupby(
            ["policy_name", "target_quiet_impact_capture"], as_index=False
        ).agg(
            outer_events=("outer_event", "nunique"),
            mean_interventions=("interventions", "mean"),
            min_interventions=("interventions", "min"),
            max_interventions=("interventions", "max"),
            total_interventions=("interventions", "sum"),
            mean_positive_interventions=("positive_interventions", "mean"),
            mean_blocked_impact=("blocked_impact", "mean"),
            mean_horizon_reduction=("horizon_reduction", "mean"),
            mean_quiet_impact_capture=("quiet_impact_capture", "mean"),
            worst_event_quiet_impact_capture=("quiet_impact_capture", "min"),
            total_oracle_top50_hits=("oracle_top50_hits", "sum"),
            total_quiet_oracle_top50_union=("quiet_oracle_top50_union", "sum"),
            mean_quiet_oracle_top50_recall=("quiet_oracle_top50_recall", "mean"),
            mean_same_count_oracle_efficiency=("same_count_oracle_efficiency", "mean"),
        )
        macro["micro_quiet_oracle_top50_recall"] = (
            macro.total_oracle_top50_hits / macro.total_quiet_oracle_top50_union
        )

        print("[6/7] Checking identity, split, and frozen-configuration boundaries...", flush=True)
        if selected.duplicated(["outer_event", "policy_name", "thread_id"]).any():
            raise ValueError("duplicate held-out intervention")
        if selected_configs.groupby("outer_event").policy_name.nunique().ne(
            len(common.CAPTURE_TARGETS)
        ).any():
            raise ValueError("missing inner-selected capture policy")

        print("[7/7] Writing complete result bundle and run record...", flush=True)
        inner_per_event.to_csv(output / "inner_grid_per_event.csv", index=False)
        inner_macro.to_csv(output / "inner_grid_macro.csv", index=False)
        pd.concat([
            common.pareto_frontier(group).assign(outer_event=outer_event)
            for outer_event, group in inner_macro.groupby("outer_event", sort=True)
        ], ignore_index=True).to_csv(output / "inner_pareto_frontier.csv", index=False)
        selected_configs.to_csv(output / "inner_selected_configs.csv", index=False)
        metrics.to_csv(output / "outer_policy_per_event.csv", index=False)
        macro.to_csv(output / "outer_policy_macro.csv", index=False)
        selected.to_csv(output / "outer_selected_threads.csv", index=False)
        result = {
            "policy_macro": macro.to_dict(orient="records"),
            "interpretation_boundary": record["interpretation_boundary"],
        }
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        record.update({
            "status": "complete", "completed_at": now(),
            "outer_events": sorted(outer.event_id.unique()),
            "aligned_rows": {"inner": len(inner), "outer": len(outer)},
            "candidate_configs_per_outer": (
                len(common.ACCELERATION_QUANTILES) * len(common.UTILITY_THRESHOLDS)
            ),
            "metrics": macro.to_dict(orient="records"),
            "output_files": [
                "inner_grid_per_event.csv", "inner_grid_macro.csv",
                "inner_pareto_frontier.csv", "inner_selected_configs.csv",
                "outer_policy_per_event.csv", "outer_policy_macro.csv",
                "outer_selected_threads.csv", "result.json", "run_record.json",
            ],
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"SUCCESS: saved quiet dual-threshold qualification saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({
            "status": "failed", "failed_at": now(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "output_files": sorted(path.name for path in output.iterdir()),
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"FAILURE: saved quiet dual-threshold qualification preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
