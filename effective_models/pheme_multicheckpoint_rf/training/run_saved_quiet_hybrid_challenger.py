"""Fixed reserve sensitivity for quiet challengers plus frozen legacy hybrid scores."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

import quiet_hybrid_challenger_common as common


HERE = Path(__file__).resolve().parent
MULTI_ROOT = HERE.parent / "experiments"
DUAL_RUN = MULTI_ROOT / "20260903_205951_708582_saved_quiet_dual_threshold_qualification"
ACCELERATION_RUN = MULTI_ROOT / "20260903_194947_531733_quiet_acceleration_rf_full"
UTILITY_OUTER = (
    MULTI_ROOT / "20260903_153740_370305_multicheckpoint_models_full"
    / "oof_predictions.csv"
)
OUTCOMES = (
    MULTI_ROOT / "20260903_183904_151619_quiet_acceleration_materialization"
    / "pheme_quiet_acceleration_features.csv"
)
HYBRID_RUN = (
    HERE.parent.parent / "pheme_active_quiet_calibrated_rf" / "experiments"
    / "20260902_132700_413954_quiet_tier_hybrid_policy_full"
)
GATE_POLICIES = ("inner_impact_capture_25", "inner_impact_capture_40")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    output = MULTI_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + "_saved_quiet_hybrid_challenger"
    )
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "started_at": now(),
        "purpose": "fixed-budget-50 early quiet challenger sensitivity",
        "inputs": {
            "dual_threshold_run": str(DUAL_RUN.resolve()),
            "acceleration_oof": str(ACCELERATION_RUN.resolve()),
            "cumulative_utility_oof": str(UTILITY_OUTER.resolve()),
            "corrected_outcomes": str(OUTCOMES.resolve()),
            "frozen_legacy_hybrid_scores": str(HYBRID_RUN.resolve()),
        },
        "budget": common.BUDGET,
        "reserve_profiles_20m_30m": [list(profile) for profile in common.RESERVE_PROFILES],
        "gate_policies": list(GATE_POLICIES),
        "split": "all component scores are outer-event OOF; dual gates were selected from nested inner OOF",
        "selection_rule": (
            "dual-qualified quiet candidates ranked by saved cumulative utility only within fixed "
            "reserve; frozen 30m hybrid score fills remaining slots to exactly 50"
        ),
        "research_safety": (
            "The legacy hybrid preventable-impact column is discarded. Every policy is evaluated "
            "against corrected multi-checkpoint outcomes. Reserve profiles are predeclared and no "
            "outer result selects a winner."
        ),
        "interpretation_boundary": (
            "The hybrid scorer was trained before the 23 corrected-target changes. This is a frozen "
            "legacy-scorer sensitivity, not a corrected-label retraining result."
        ),
    }
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        print("Starting saved fixed-budget quiet challenger sensitivity.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/7] Loading frozen hybrid, acceleration, utility, and corrected outcomes...", flush=True)
        hybrid = pd.read_csv(
            HYBRID_RUN / "outer_scores.csv", dtype={"thread_id": str, "event_id": str}
        ).rename(columns={"preventable_impact": "legacy_preventable_impact"})
        acceleration = pd.read_csv(
            ACCELERATION_RUN / "outer_quiet_acceleration_oof.csv",
            dtype={"thread_id": str, "event_id": str},
        )
        utility = pd.read_csv(
            UTILITY_OUTER, dtype={"thread_id": str, "event_id": str}
        )
        utility = utility.loc[
            utility.family.eq("cumulative") & utility.checkpoint_sec.isin((1200, 1800))
        ].copy()
        outcomes = pd.read_csv(OUTCOMES, dtype={"thread_id": str, "event_id": str})
        outcomes = outcomes.loc[outcomes.checkpoint_sec.isin((1200, 1800))].copy()
        configs = pd.read_csv(DUAL_RUN / "inner_selected_configs.csv")
        configs = configs.loc[configs.policy_name.isin(GATE_POLICIES)].copy()

        print("[2/7] Discarding legacy labels and joining only OOF scores to corrected outcomes...", flush=True)
        keys = ["thread_id", "event_id", "checkpoint_sec"]
        dual = acceleration.loc[acceleration.checkpoint_sec.isin((1200, 1800))].rename(
            columns={"prediction": "acceleration_prediction"}
        ).merge(
            utility[keys + ["prediction"]].rename(columns={"prediction": "utility_prediction"}),
            on=keys, how="left", validate="one_to_one",
        ).merge(
            outcomes[keys + ["dynamic_preventable_impact", "future_growth"]],
            on=keys, how="left", validate="one_to_one",
        )
        dual["predicted_impact"] = np.maximum(
            np.expm1(dual.utility_prediction.to_numpy(dtype=float)), 0.0
        )
        corrected30 = outcomes.loc[outcomes.checkpoint_sec.eq(1800), [
            "thread_id", "event_id", "dynamic_preventable_impact", "future_growth"
        ]]
        hybrid = hybrid.merge(
            corrected30, on=["thread_id", "event_id"], how="left", validate="one_to_one"
        )
        if dual[["utility_prediction", "dynamic_preventable_impact"]].isna().any().any():
            raise ValueError("dual score/outcome join failed")
        if hybrid[["dynamic_preventable_impact", "future_growth"]].isna().any().any():
            raise ValueError("hybrid corrected-outcome join failed")
        changed_targets = int(
            hybrid.legacy_preventable_impact.ne(hybrid.dynamic_preventable_impact).sum()
        )
        if changed_targets != 23:
            raise ValueError(f"expected 23 corrected hybrid targets, found {changed_targets}")

        print("[3/7] Building predeclared frozen gate/reserve policy configurations...", flush=True)
        policy_configs = []
        for event_id in sorted(hybrid.event_id.unique()):
            event_configs = configs.loc[configs.outer_event.eq(event_id)]
            for gate_policy in GATE_POLICIES:
                row = event_configs.loc[event_configs.policy_name.eq(gate_policy)].iloc[0]
                for reserve20, reserve30 in common.RESERVE_PROFILES[1:]:
                    policy_configs.append((event_id, common.ChallengerConfig(
                        gate_policy=gate_policy,
                        acceleration_quantile=float(row.acceleration_quantile),
                        min_predicted_impact=float(row.min_predicted_impact),
                        acceleration_threshold_20m=float(row.acceleration_threshold_20m),
                        acceleration_threshold_30m=float(row.acceleration_threshold_30m),
                        reserve_20m=reserve20, reserve_30m=reserve30,
                    )))

        print("[4/7] Replaying baseline and six fixed-reserve policies on held-out events...", flush=True)
        metric_rows, selected_rows = [], []
        for event_index, event_id in enumerate(sorted(hybrid.event_id.unique()), 1):
            print(f"  event {event_index}/7: {event_id}", flush=True)
            hybrid_event = hybrid.loc[hybrid.event_id.eq(event_id)].copy()
            dual_event = dual.loc[dual.event_id.eq(event_id)].copy()
            baseline_config = common.ChallengerConfig(
                "baseline", 0.0, 0.0, -np.inf, -np.inf, 0, 0,
            )
            baseline = common.replay(dual_event, hybrid_event, baseline_config)
            baseline_ids = set(baseline.thread_id)
            baseline_blocked = float(baseline.dynamic_preventable_impact.sum())
            event_outcomes = outcomes.loc[outcomes.event_id.eq(event_id)]
            oracle_ids = set(
                event_outcomes.loc[event_outcomes.checkpoint_sec.eq(1800)].sort_values(
                    ["dynamic_preventable_impact", "thread_id"],
                    ascending=[False, True], kind="stable",
                ).head(50).thread_id
            )
            horizon = float(
                event_outcomes.loc[event_outcomes.checkpoint_sec.eq(1200), "future_growth"].sum()
            )
            event_policy_configs = [
                config for configured_event, config in policy_configs
                if configured_event == event_id
            ]
            all_configs = [baseline_config, *event_policy_configs]
            for config in all_configs:
                chosen = baseline if config.policy_name == baseline_config.policy_name else common.replay(
                    dual_event, hybrid_event, config
                )
                chosen_ids = set(chosen.thread_id)
                challengers = chosen.loc[chosen.selection_source.eq("quiet_challenger")]
                challenger_ids = set(challengers.thread_id)
                blocked = float(chosen.dynamic_preventable_impact.sum())
                quiet_map = hybrid_event.set_index("thread_id").quiet.astype(bool)
                strict30_quiet = int(chosen.thread_id.map(quiet_map).sum())
                early_gain = 0.0
                if len(challengers):
                    at30 = hybrid_event.set_index("thread_id").dynamic_preventable_impact
                    early_gain = float(
                        challengers.dynamic_preventable_impact.to_numpy(dtype=float).sum()
                        - challengers.thread_id.map(at30).to_numpy(dtype=float).sum()
                    )
                metric_rows.append({
                    "event_id": event_id, "policy_name": config.policy_name,
                    "gate_policy": config.gate_policy,
                    "reserve_20m": config.reserve_20m, "reserve_30m": config.reserve_30m,
                    "interventions": len(chosen), "challenger_interventions": len(challengers),
                    "novel_challengers_vs_baseline": len(challenger_ids - baseline_ids),
                    "displaced_baseline_threads": len(baseline_ids - chosen_ids),
                    "strict30_quiet_interventions": strict30_quiet,
                    "blocked_impact": blocked, "baseline_blocked_impact": baseline_blocked,
                    "blocked_impact_delta": blocked - baseline_blocked,
                    "horizon_future_at_20m": horizon,
                    "horizon_reduction": blocked / horizon if horizon else 0.0,
                    "baseline_horizon_reduction": baseline_blocked / horizon if horizon else 0.0,
                    "early_timing_gain": early_gain,
                    "oracle_top50_hits": len(chosen_ids & oracle_ids),
                    "baseline_oracle_top50_hits": len(baseline_ids & oracle_ids),
                    "oracle_top50_hit_delta": len(chosen_ids & oracle_ids) - len(baseline_ids & oracle_ids),
                })
                selected_rows.append(chosen.assign(
                    gate_policy=config.gate_policy,
                    reserve_20m=config.reserve_20m, reserve_30m=config.reserve_30m,
                ))

        print("[5/7] Summarizing fixed reserve curves without outer-driven selection...", flush=True)
        metrics = pd.DataFrame(metric_rows)
        selected = pd.concat(selected_rows, ignore_index=True)
        macro = metrics.groupby(
            ["policy_name", "gate_policy", "reserve_20m", "reserve_30m"], as_index=False
        ).agg(
            events=("event_id", "nunique"),
            mean_challenger_interventions=("challenger_interventions", "mean"),
            mean_novel_challengers=("novel_challengers_vs_baseline", "mean"),
            mean_strict30_quiet_interventions=("strict30_quiet_interventions", "mean"),
            mean_blocked_impact=("blocked_impact", "mean"),
            mean_blocked_impact_delta=("blocked_impact_delta", "mean"),
            events_improved=("blocked_impact_delta", lambda x: int(x.gt(0).sum())),
            events_worsened=("blocked_impact_delta", lambda x: int(x.lt(0).sum())),
            mean_horizon_reduction=("horizon_reduction", "mean"),
            mean_early_timing_gain=("early_timing_gain", "mean"),
            total_oracle_top50_hits=("oracle_top50_hits", "sum"),
            mean_oracle_top50_hit_delta=("oracle_top50_hit_delta", "mean"),
        ).sort_values(
            ["gate_policy", "reserve_20m", "reserve_30m"], kind="stable"
        )

        print("[6/7] Auditing fixed budget, corrected labels, and identity boundaries...", flush=True)
        if metrics.interventions.ne(50).any():
            raise ValueError("policy did not preserve budget 50")
        if selected.duplicated(["event_id", "policy_name", "thread_id"]).any():
            raise ValueError("duplicate selected thread within policy")
        if metrics.groupby("policy_name").event_id.nunique().ne(7).any():
            raise ValueError("incomplete eligible-event policy curve")

        print("[7/7] Writing fixed-policy comparison and complete run record...", flush=True)
        metrics.to_csv(output / "per_event_policy_comparison.csv", index=False)
        macro.to_csv(output / "policy_macro.csv", index=False)
        selected.to_csv(output / "selected_threads.csv", index=False)
        pd.DataFrame([{
            "outer_event": outer_event,
            "gate_policy": config.gate_policy,
            "policy_name": config.policy_name,
            "acceleration_quantile": config.acceleration_quantile,
            "min_predicted_impact": config.min_predicted_impact,
            "acceleration_threshold_20m": config.acceleration_threshold_20m,
            "acceleration_threshold_30m": config.acceleration_threshold_30m,
            "reserve_20m": config.reserve_20m, "reserve_30m": config.reserve_30m,
        } for outer_event, config in policy_configs]).to_csv(
            output / "frozen_policy_configs.csv", index=False
        )
        result = {
            "policy_macro": macro.to_dict(orient="records"),
            "winner_selection": "none; all reserve curves are descriptive held-out results",
            "interpretation_boundary": record["interpretation_boundary"],
        }
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        record.update({
            "status": "complete", "completed_at": now(),
            "corrected_target_differences_discarded_from_legacy_input": changed_targets,
            "outer_events": sorted(hybrid.event_id.unique()),
            "metrics": macro.to_dict(orient="records"),
            "output_files": [
                "per_event_policy_comparison.csv", "policy_macro.csv",
                "selected_threads.csv", "frozen_policy_configs.csv",
                "result.json", "run_record.json",
            ],
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"SUCCESS: saved quiet hybrid challenger saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({
            "status": "failed", "failed_at": now(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "output_files": sorted(path.name for path in output.iterdir()),
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"FAILURE: saved quiet hybrid challenger preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
