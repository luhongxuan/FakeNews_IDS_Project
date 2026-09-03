"""Timing-only acceleration diagnostic with Top-50 identities committed at 20m."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

import quiet_timing_only_common as common


HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parent / "experiments"
UTILITY = OUT_ROOT / "20260903_153740_370305_multicheckpoint_models_full" / "oof_predictions.csv"
ACCELERATION = OUT_ROOT / "20260903_194947_531733_quiet_acceleration_rf_full"
DUAL = OUT_ROOT / "20260903_205951_708582_saved_quiet_dual_threshold_qualification"
OUTCOMES = (
    OUT_ROOT / "20260903_183904_151619_quiet_acceleration_materialization"
    / "pheme_quiet_acceleration_features.csv"
)
GATE_POLICIES = ("inner_impact_capture_25", "inner_impact_capture_40")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    output = OUT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + "_saved_quiet_acceleration_timing_only"
    )
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "started_at": now(),
        "purpose": "leakage-safe acceleration timing-only intervention diagnostic",
        "inputs": {
            "cumulative_outer_oof": str(UTILITY.resolve()),
            "quiet_acceleration_outer_oof": str(ACCELERATION.resolve()),
            "inner_selected_dual_gates": str(DUAL.resolve()),
            "corrected_outcomes": str(OUTCOMES.resolve()),
        },
        "identity_rule": "commit cumulative RF Top-50 using only minute-20 scores",
        "timing_rule": "dual-qualified committed quiet threads act at 20m; all others act at 30m",
        "gate_policies": list(GATE_POLICIES), "budget": common.BUDGET,
        "split": "all prediction inputs are outer-event OOF; gates were selected from nested inner OOF",
        "research_safety": (
            "No minute-30 score determines a minute-20 action. Identities are frozen from minute-20 "
            "OOF scores before the acceleration gate changes timing. Future outcomes are evaluation-only."
        ),
        "interpretation_boundary": (
            "This is not the strict-30 active/quiet hybrid identity set; using that future identity set "
            "at minute 20 would be leakage. The experiment isolates timing value for a deployable 20m commitment."
        ),
    }
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        print("Starting leakage-safe quiet acceleration timing-only diagnostic.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/6] Loading minute-20 utility, quiet acceleration, gates, and outcomes...", flush=True)
        utility = pd.read_csv(UTILITY, dtype={"thread_id": str, "event_id": str})
        utility = utility.loc[
            utility.family.eq("cumulative") & utility.checkpoint_sec.eq(1200)
        ].rename(columns={"prediction": "utility_prediction"}).copy()
        utility["predicted_impact"] = np.maximum(
            np.expm1(utility.utility_prediction.to_numpy(dtype=float)), 0.0
        )
        acceleration = pd.read_csv(
            ACCELERATION / "outer_quiet_acceleration_oof.csv",
            dtype={"thread_id": str, "event_id": str},
        )
        acceleration = acceleration.loc[acceleration.checkpoint_sec.eq(1200)].rename(
            columns={"prediction": "acceleration_prediction"}
        ).copy()
        gates = pd.read_csv(DUAL / "inner_selected_configs.csv")
        gates = gates.loc[gates.policy_name.isin(GATE_POLICIES)].copy()
        outcomes = pd.read_csv(OUTCOMES, dtype={"thread_id": str, "event_id": str})
        outcomes = outcomes.loc[outcomes.checkpoint_sec.isin((1200, 1800))].copy()

        print("[2/6] Auditing time direction, event separation, and exact identities...", flush=True)
        if utility.duplicated(["thread_id", "event_id"]).any():
            raise ValueError("duplicate minute-20 utility score")
        if acceleration.duplicated(["thread_id", "event_id"]).any():
            raise ValueError("duplicate minute-20 acceleration score")
        if gates.groupby(["outer_event", "policy_name"]).size().ne(1).any():
            raise ValueError("ambiguous inner-selected timing gate")
        wide = outcomes.pivot(
            index=["thread_id", "event_id"], columns="checkpoint_sec",
            values="dynamic_preventable_impact",
        )
        if (wide[1200] < wide[1800]).any():
            raise ValueError("negative 20-to-30-minute wait loss")

        print("[3/6] Committing held-out Top-50 identities at minute 20...", flush=True)
        metric_rows, selected_rows = [], []
        for index, event_id in enumerate(sorted(utility.event_id.unique()), 1):
            if event_id not in set(gates.outer_event):
                continue
            print(f"  event {index}/9: {event_id}", flush=True)
            event_utility = utility.loc[utility.event_id.eq(event_id), [
                "thread_id", "event_id", "utility_prediction", "predicted_impact"
            ]]
            event_outcomes = outcomes.loc[outcomes.event_id.eq(event_id)]
            committed = common.commit_at_20(event_utility, event_outcomes)
            event_acceleration = acceleration.loc[acceleration.event_id.eq(event_id)]

            print(f"[4/6] Applying frozen inner gates for {event_id}...", flush=True)
            for policy_name in GATE_POLICIES:
                gate = gates.loc[
                    gates.outer_event.eq(event_id) & gates.policy_name.eq(policy_name)
                ].iloc[0]
                selected = common.timing_replay(
                    committed, event_acceleration,
                    float(gate.acceleration_threshold_20m),
                    float(gate.min_predicted_impact), policy_name,
                )
                metrics = common.timing_metrics(selected)
                metrics.update({
                    "event_id": event_id, "policy_name": policy_name,
                    "acceleration_quantile": float(gate.acceleration_quantile),
                    "acceleration_threshold_20m": float(gate.acceleration_threshold_20m),
                    "min_predicted_impact": float(gate.min_predicted_impact),
                })
                metric_rows.append(metrics)
                selected_rows.append(selected)

        print("[5/6] Summarizing timing gain against random and oracle timing...", flush=True)
        metrics = pd.DataFrame(metric_rows)
        selected = pd.concat(selected_rows, ignore_index=True)
        macro = metrics.groupby("policy_name", as_index=False).agg(
            events=("event_id", "nunique"),
            mean_early_interventions=("early_interventions", "mean"),
            min_early_interventions=("early_interventions", "min"),
            max_early_interventions=("early_interventions", "max"),
            mean_baseline_blocked_impact=("baseline_blocked_impact_at_30m", "mean"),
            mean_timing_blocked_impact=("timing_blocked_impact", "mean"),
            mean_timing_gain=("timing_gain", "mean"),
            events_with_positive_gain=("timing_gain", lambda x: int(x.gt(0).sum())),
            mean_wait_loss_capture=("wait_loss_capture", "mean"),
            mean_oracle_timing_efficiency=("oracle_timing_efficiency", "mean"),
            mean_random_expected_gain=("same_count_random_expected_gain", "mean"),
            mean_gain_minus_random=("gain_minus_random_expected", "mean"),
            mean_early_acceleration_precision=("early_positive_acceleration_precision", "mean"),
        )

        print("[6/6] Auditing identity invariance and writing complete result bundle...", flush=True)
        if metrics.interventions.ne(common.BUDGET).any():
            raise ValueError("timing policy changed intervention count")
        if selected.groupby(["event_id", "policy_name"]).thread_id.nunique().ne(common.BUDGET).any():
            raise ValueError("timing policy changed committed identities")
        metrics.to_csv(output / "per_event_timing_metrics.csv", index=False)
        macro.to_csv(output / "timing_macro.csv", index=False)
        selected.to_csv(output / "committed_thread_timing.csv", index=False)
        result = {
            "timing_macro": macro.to_dict(orient="records"),
            "interpretation_boundary": record["interpretation_boundary"],
        }
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        record.update({
            "status": "complete", "completed_at": now(),
            "outer_events": sorted(metrics.event_id.unique()),
            "metrics": macro.to_dict(orient="records"),
            "output_files": [
                "per_event_timing_metrics.csv", "timing_macro.csv",
                "committed_thread_timing.csv", "result.json", "run_record.json",
            ],
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"SUCCESS: quiet acceleration timing-only saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({
            "status": "failed", "failed_at": now(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "output_files": sorted(path.name for path in output.iterdir()),
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"FAILURE: quiet acceleration timing-only preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
