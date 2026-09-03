"""Dynamic quiet audit and fixed-budget sequential replay from saved OOF scores."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

import multicheckpoint_model_common as model_common


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL_ROOT = HERE.parent
OUT_ROOT = MODEL_ROOT / "experiments"
SOURCE_RUN = MODEL_ROOT / "experiments" / "20260903_153740_370305_multicheckpoint_models_full"
PREDICTIONS = SOURCE_RUN / "oof_predictions.csv"
MATERIALIZATION = (
    MODEL_ROOT / "experiments" / "20260903_153526_240803_multicheckpoint_materialization_full"
)
FEATURES = MATERIALIZATION / "pheme_multicheckpoint_features.csv"
CORRECTED_V5 = (
    ROOT / "effective_models" / "pheme_v5_corrected_target" / "experiments"
    / "20260903_143643_168807_corrected_v5_nested_full" / "oof_thread_scores.csv"
)
HYBRID = (
    ROOT / "effective_models" / "pheme_active_quiet_calibrated_rf" / "experiments"
    / "20260902_132700_413954_quiet_tier_hybrid_policy_full" / "outer_scores.csv"
)
CHECKPOINTS = (600, 1200, 1800, 2400, 3000, 3600)
BALANCED_QUOTAS = (9, 9, 8, 8, 8, 8)
QUIET_QUANTILE = 0.60


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_joined() -> pd.DataFrame:
    scores = pd.read_csv(PREDICTIONS, dtype={"thread_id": str, "event_id": str})
    features = pd.read_csv(
        FEATURES,
        usecols=[
            "thread_id", "event_id", "checkpoint_sec",
            "cumulative__cum_time_since_last_activity",
        ],
        dtype={"thread_id": str, "event_id": str},
    )
    index = ["thread_id", "event_id", "checkpoint_sec"]
    cumulative = scores.loc[scores.family.eq("cumulative")].rename(
        columns={"prediction": "cumulative_score"}
    ).drop(columns="family")
    window = scores.loc[scores.family.eq("window_delta"), index + ["prediction"]].rename(
        columns={"prediction": "window_delta_score"}
    )
    joined = cumulative.merge(window, on=index, validate="one_to_one")
    joined = joined.merge(features, on=index, validate="one_to_one")
    if len(joined) != 14412 or joined.duplicated(["thread_id", "checkpoint_sec"]).any():
        raise ValueError("unexpected joined multi-checkpoint identity")
    return joined


def dynamic_quiet_rows(frame: pd.DataFrame, eligible: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = []
    selected_rows = []
    for event in eligible:
        for checkpoint in CHECKPOINTS:
            train = frame.loc[
                frame.event_id.ne(event) & frame.checkpoint_sec.eq(checkpoint)
            ]
            test = frame.loc[
                frame.event_id.eq(event) & frame.checkpoint_sec.eq(checkpoint)
            ].copy()
            threshold = float(
                train["cumulative__cum_time_since_last_activity"].quantile(QUIET_QUANTILE)
            )
            test["dynamic_quiet"] = test[
                "cumulative__cum_time_since_last_activity"
            ].ge(threshold)
            oracle = test.sort_values(
                ["dynamic_preventable_impact", "thread_id"],
                ascending=[False, True], kind="stable",
            ).head(50)
            oracle_ids = set(oracle.thread_id)
            oracle_quiet = int(oracle.dynamic_quiet.sum())
            for family, score_column in (
                ("cumulative", "cumulative_score"),
                ("window_delta", "window_delta_score"),
            ):
                selected = test.sort_values(
                    [score_column, "thread_id"], ascending=[False, True], kind="stable"
                ).head(50).copy()
                selected["oracle_top50"] = selected.thread_id.isin(oracle_ids)
                quiet = selected.loc[selected.dynamic_quiet]
                active = selected.loc[~selected.dynamic_quiet]
                quiet_hits = int(quiet.oracle_top50.sum())
                active_hits = int(active.oracle_top50.sum())
                detail.append({
                    "event_id": event, "checkpoint_sec": checkpoint, "family": family,
                    "quiet_threshold_sec": threshold, "oracle_quiet_top50": oracle_quiet,
                    "oracle_active_top50": 50 - oracle_quiet,
                    "selected_quiet": len(quiet), "selected_active": len(active),
                    "quiet_hits": quiet_hits, "active_hits": active_hits,
                    "quiet_precision": quiet_hits / len(quiet) if len(quiet) else 0.0,
                    "quiet_recall": quiet_hits / oracle_quiet if oracle_quiet else 0.0,
                    "active_precision": active_hits / len(active) if len(active) else 0.0,
                    "active_recall": active_hits / (50 - oracle_quiet) if oracle_quiet < 50 else 0.0,
                })
                selected_rows.append(selected.assign(family=family)[[
                    "thread_id", "event_id", "checkpoint_sec", "family", "dynamic_quiet",
                    "oracle_top50", "dynamic_preventable_impact", score_column,
                ]].rename(columns={score_column: "prediction"}))
    return pd.DataFrame(detail), pd.concat(selected_rows, ignore_index=True)


def sequential_select(
    event: pd.DataFrame,
    policy: str,
    quotas: tuple[int, ...],
    corrected_v5: pd.Series,
    hybrid: pd.Series,
) -> pd.DataFrame:
    selected_ids: set[str] = set()
    chunks = []
    for checkpoint, quota in zip(CHECKPOINTS, quotas):
        if quota <= 0:
            continue
        available = event.loc[
            event.checkpoint_sec.eq(checkpoint) & ~event.thread_id.isin(selected_ids)
        ].copy()
        if policy == "balanced_cumulative":
            chosen = available.sort_values(
                ["cumulative_score", "thread_id"], ascending=[False, True], kind="stable"
            ).head(quota)
            chosen = chosen.assign(selection_channel="cumulative")
        elif policy == "balanced_window_delta":
            chosen = available.sort_values(
                ["window_delta_score", "thread_id"], ascending=[False, True], kind="stable"
            ).head(quota)
            chosen = chosen.assign(selection_channel="window_delta")
        elif policy == "balanced_cumulative_plus_one_window":
            main = available.sort_values(
                ["cumulative_score", "thread_id"], ascending=[False, True], kind="stable"
            ).head(max(quota - 1, 0)).assign(selection_channel="cumulative")
            challenger_pool = available.loc[~available.thread_id.isin(set(main.thread_id))]
            challenger = challenger_pool.sort_values(
                ["window_delta_score", "thread_id"], ascending=[False, True], kind="stable"
            ).head(min(1, quota)).assign(selection_channel="window_challenger")
            chosen = pd.concat([main, challenger], ignore_index=False)
        elif policy == "balanced_dynamic_oracle":
            chosen = available.sort_values(
                ["dynamic_preventable_impact", "thread_id"],
                ascending=[False, True], kind="stable",
            ).head(quota).assign(selection_channel="dynamic_oracle")
        elif policy in {"single30_cumulative", "single30_corrected_v5", "single30_hybrid"}:
            if checkpoint != 1800:
                continue
            if policy == "single30_cumulative":
                available["single_score"] = available.cumulative_score
            elif policy == "single30_corrected_v5":
                available["single_score"] = available.thread_id.map(corrected_v5)
            else:
                available["single_score"] = available.thread_id.map(hybrid)
            if available.single_score.isna().any():
                raise ValueError(f"missing score for {policy}")
            chosen = available.sort_values(
                ["single_score", "thread_id"], ascending=[False, True], kind="stable"
            ).head(quota).assign(selection_channel=policy)
        else:
            raise ValueError(policy)
        if len(chosen) != quota:
            raise ValueError(f"{policy}: insufficient available candidates at {checkpoint}")
        selected_ids.update(chosen.thread_id)
        chunks.append(chosen.assign(action_checkpoint_sec=checkpoint, policy=policy))
    result = pd.concat(chunks, ignore_index=True) if chunks else event.head(0).copy()
    if result.thread_id.duplicated().any() or len(result) != sum(quotas):
        raise ValueError(f"{policy}: duplicate or wrong total interventions")
    return result


def main() -> None:
    output = OUT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + "_dynamic_quiet_sequential_policy_diagnostic"
    )
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "started_at": now(), "mode": "saved OOF diagnostic; no fitting",
        "inputs": {
            "predictions": str(PREDICTIONS.resolve()), "features": str(FEATURES.resolve()),
            "corrected_v5": str(CORRECTED_V5.resolve()), "hybrid": str(HYBRID.resolve()),
        },
        "split": "seven eligible held-out-event OOF predictions",
        "checkpoints_seconds": list(CHECKPOINTS), "total_budget": 50,
        "balanced_quotas": list(BALANCED_QUOTAS), "quiet_quantile": QUIET_QUANTILE,
        "research_safety": (
            "Policies are fixed before reading event outcomes. No outer result selects a schedule or "
            "challenger count. Dynamic quiet thresholds use only other events at the same checkpoint."
        ),
    }
    write_record(output, record)
    try:
        print("Starting dynamic-quiet and fixed-budget sequential policy diagnostic.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/6] Loading saved held-out multi-checkpoint predictions...", flush=True)
        frame = load_joined()
        counts = frame[["thread_id", "event_id"]].drop_duplicates().groupby("event_id").size()
        eligible = sorted(counts.loc[counts.ge(100)].index)
        print("[2/6] Applying train-event-only dynamic quiet gates at every checkpoint...", flush=True)
        quiet_detail, quiet_selected = dynamic_quiet_rows(frame, eligible)
        quiet_macro = quiet_detail.groupby(["checkpoint_sec", "family"], as_index=False).agg(
            eligible_events=("event_id", "nunique"),
            oracle_quiet_top50=("oracle_quiet_top50", "sum"),
            selected_quiet=("selected_quiet", "sum"),
            quiet_hits=("quiet_hits", "sum"),
            active_hits=("active_hits", "sum"),
            mean_quiet_precision=("quiet_precision", "mean"),
            mean_quiet_recall=("quiet_recall", "mean"),
            mean_active_recall=("active_recall", "mean"),
        )

        print("[3/6] Replaying fixed total-Budget-50 sequential policies...", flush=True)
        corrected_v5 = pd.read_csv(CORRECTED_V5, dtype={"thread_id": str}).set_index("thread_id").prediction
        hybrid = pd.read_csv(HYBRID, dtype={"thread_id": str}).set_index("thread_id").hybrid_calibrated_score
        policies = {
            "balanced_cumulative": BALANCED_QUOTAS,
            "balanced_window_delta": BALANCED_QUOTAS,
            "balanced_cumulative_plus_one_window": BALANCED_QUOTAS,
            "balanced_dynamic_oracle": BALANCED_QUOTAS,
            "single30_cumulative": (0, 0, 50, 0, 0, 0),
            "single30_corrected_v5": (0, 0, 50, 0, 0, 0),
            "single30_hybrid": (0, 0, 50, 0, 0, 0),
        }
        metric_rows, selected_rows = [], []
        for event_index, event_id in enumerate(eligible, 1):
            print(f"  event {event_index}/{len(eligible)}: {event_id}", flush=True)
            event = frame.loc[frame.event_id.eq(event_id)].copy()
            horizon_future = float(
                event.loc[event.checkpoint_sec.eq(600), "future_growth"].sum()
            )
            oracle_results = {}
            for policy, quotas in policies.items():
                selected = sequential_select(event, policy, quotas, corrected_v5, hybrid)
                blocked = float(selected.dynamic_preventable_impact.sum())
                if policy == "balanced_dynamic_oracle":
                    oracle_results["balanced"] = blocked
                selected_rows.append(selected[[
                    "policy", "selection_channel", "thread_id", "event_id",
                    "action_checkpoint_sec", "dynamic_preventable_impact", "future_growth",
                    "cumulative_score", "window_delta_score",
                ]])
                metric_rows.append({
                    "event_id": event_id, "policy": policy, "interventions": len(selected),
                    "positive_interventions": int(selected.dynamic_preventable_impact.gt(0).sum()),
                    "blocked_impact": blocked,
                    "horizon_future_at_10m": horizon_future,
                    "horizon_reduction": blocked / horizon_future if horizon_future else 0.0,
                    "mean_action_minute": float(selected.action_checkpoint_sec.mean() / 60.0),
                })
            # Same-time Oracle@30 makes single-shot efficiencies directly interpretable.
            at30 = event.loc[event.checkpoint_sec.eq(1800)].sort_values(
                ["dynamic_preventable_impact", "thread_id"], ascending=[False, True], kind="stable"
            ).head(50)
            oracle30 = float(at30.dynamic_preventable_impact.sum())
            for row in metric_rows[-len(policies):]:
                if row["policy"].startswith("balanced_"):
                    row["schedule_oracle_efficiency"] = (
                        row["blocked_impact"] / oracle_results["balanced"]
                        if oracle_results.get("balanced") else 0.0
                    )
                else:
                    row["schedule_oracle_efficiency"] = (
                        row["blocked_impact"] / oracle30 if oracle30 else 0.0
                    )

        print("[4/6] Aggregating event-balanced policy results...", flush=True)
        policy_detail = pd.DataFrame(metric_rows)
        policy_macro = policy_detail.groupby("policy", as_index=False).agg(
            eligible_events=("event_id", "nunique"),
            total_blocked_impact=("blocked_impact", "sum"),
            total_positive_interventions=("positive_interventions", "sum"),
            mean_horizon_reduction=("horizon_reduction", "mean"),
            mean_schedule_oracle_efficiency=("schedule_oracle_efficiency", "mean"),
            mean_action_minute=("mean_action_minute", "mean"),
            worst_event_horizon_reduction=("horizon_reduction", "min"),
        ).sort_values(["mean_horizon_reduction", "policy"], ascending=[False, True])

        print("[5/6] Writing dynamic quiet and sequential selection evidence...", flush=True)
        quiet_detail.to_csv(output / "dynamic_quiet_per_event.csv", index=False)
        quiet_macro.to_csv(output / "dynamic_quiet_macro.csv", index=False)
        quiet_selected.to_csv(output / "dynamic_quiet_selected_top50.csv", index=False)
        policy_detail.to_csv(output / "sequential_policy_per_event.csv", index=False)
        policy_macro.to_csv(output / "sequential_policy_macro.csv", index=False)
        pd.concat(selected_rows, ignore_index=True).to_csv(
            output / "sequential_selected_threads.csv", index=False
        )
        print("[6/6] Completing result and run record...", flush=True)
        result = {
            "eligible_events": eligible,
            "dynamic_quiet_macro": quiet_macro.to_dict(orient="records"),
            "sequential_policy_macro": policy_macro.to_dict(orient="records"),
            "interpretation_boundary": (
                "This is a fixed-policy OOF replay. It does not select a policy using outer outcomes."
            ),
        }
        (output / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        record.update({
            "status": "complete", "completed_at": now(), "eligible_events": eligible,
            "metrics": {
                "dynamic_quiet_rows": len(quiet_detail), "policy_event_rows": len(policy_detail),
                "best_descriptive_policy": str(policy_macro.iloc[0].policy),
                "best_mean_horizon_reduction": float(policy_macro.iloc[0].mean_horizon_reduction),
            },
            "output_files": [
                "dynamic_quiet_per_event.csv", "dynamic_quiet_macro.csv",
                "dynamic_quiet_selected_top50.csv", "sequential_policy_per_event.csv",
                "sequential_policy_macro.csv", "sequential_selected_threads.csv",
                "result.json", "run_record.json",
            ],
        })
        write_record(output, record)
        print(f"SUCCESS: dynamic quiet/sequential diagnostic saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({
            "status": "failed", "failed_at": now(), "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(), "output_files": sorted(path.name for path in output.iterdir()),
        })
        write_record(output, record)
        print(f"FAILURE: dynamic quiet/sequential diagnostic preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
