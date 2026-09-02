"""Full-policy stacking ensemble of two already-fitted, already-OOF models.

Hypothesis: pheme_v5_text_rf and pheme_active_quiet_calibrated_rf were
trained independently on the identical protected v5 strict30 candidate pool
(2402 threads, 7 eligible events, 30-minute cutoff, preventable_impact
label) and each already has a saved nested-LOEO out-of-fold (OOF) score per
thread. Neither model has ever been blended with the other. If their errors
are even partially uncorrelated, a rank-fusion blend selected by a proper
leave-one-event-out meta-selection (never touching the outer event) should
get closer to the oracle ranking (higher oracle_efficiency@K) than either
model alone, without any new feature engineering, model refitting, or new
information beyond what both models already legitimately see by cutoff T.

This performs NO model training. It only reads two existing OOF CSVs and
does rank arithmetic. Runtime is well under a second, so this script is a
short, bounded evaluation/diagnostic per AGENTS.md and is run directly
(no separate smoke/full split, no user hand-off required).

Selection methodology reuses the lesson learned from the quiet-tier pool
experiment (see pheme_analysis_claude/quiet_pool_rerank_common.py): the
blend weight is selected by the per-inner-event MEDIAN oracle_efficiency
across a leave-one-event-out meta-selection (using the six other eligible
events, never the outer event), with a guard requiring every budget's
median oracle_efficiency not to regress the better of the two single-model
endpoints (weight=0.0 = active/quiet calibrated RF only, weight=1.0 = v5
text RF only) by more than GUARD_TOLERANCE. If no blend weight clears the
guard for an outer event, selection falls back to whichever single model
has the higher inner mean oracle_efficiency.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ACTIVE_QUIET_TRAINING = ROOT / "effective_models" / "pheme_active_quiet_calibrated_rf" / "training"
sys.path.insert(0, str(ACTIVE_QUIET_TRAINING))

from quiet_tier_hybrid_score import percentile_rank  # noqa: E402


V5_OOF = ROOT / "effective_models" / "pheme_v5_text_rf" / "reference_result" / "20260831_223827_909201_v5_best_text_oof_min_interventions" / "oof_thread_scores.csv"
AQ_OOF = ROOT / "effective_models" / "pheme_active_quiet_calibrated_rf" / "reference_result" / "20260901_210513_260899_paired_oof_pheme_quiet_expert_calibration" / "paired_oof_scores.csv"
EXPECTED_THREADS = 2402
ELIGIBLE_MIN_THREADS = 100
BUDGETS = (1, 3, 5, 10, 20, 50, 100)
WEIGHTS = tuple(round(w, 2) for w in np.linspace(0.0, 1.0, 11))  # 0.0 .. 1.0 step 0.1
GUARD_TOLERANCE = 0.02
OUT_ROOT = HERE / "experiments"


def write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def load_and_join_oof() -> tuple[pd.DataFrame, list[str]]:
    """Join the two independently-produced OOF files and prove they agree on identity/labels."""
    v5 = pd.read_csv(V5_OOF, dtype={"thread_id": str, "event_id": str})
    aq = pd.read_csv(AQ_OOF, dtype={"thread_id": str, "event_id": str})
    for name, df in (("v5_text_rf", v5), ("active_quiet_calibrated_rf", aq)):
        if len(df) != EXPECTED_THREADS or df.thread_id.duplicated().any():
            raise ValueError(f"{name} OOF file does not have {EXPECTED_THREADS} unique threads")
    merged = v5[["thread_id", "event_id", "prediction", "preventable_impact"]].merge(
        aq[["thread_id", "event_id", "calibrated_two_expert_score", "preventable_impact"]],
        on="thread_id", suffixes=("_v5", "_aq"), validate="one_to_one",
    )
    if len(merged) != EXPECTED_THREADS:
        raise ValueError("OOF join did not produce the expected 2402 threads")
    if (merged.event_id_v5 != merged.event_id_aq).any():
        raise ValueError("v5 and active/quiet OOF files disagree on event_id for a shared thread_id")
    if not np.isclose(merged.preventable_impact_v5, merged.preventable_impact_aq).all():
        raise ValueError("v5 and active/quiet OOF files disagree on preventable_impact for a shared thread_id -- labels are not identical, do not blend")
    merged = merged.rename(columns={"event_id_v5": "event_id", "preventable_impact_v5": "preventable_impact"})
    merged = merged.drop(columns=["event_id_aq", "preventable_impact_aq"])
    sizes = merged.groupby("event_id").size()
    eligible = sorted(sizes[sizes >= ELIGIBLE_MIN_THREADS].index.tolist())
    if len(eligible) != 7:
        raise ValueError(f"Expected 7 eligible events, found {len(eligible)}")
    merged["v5_rank"] = np.nan
    merged["aq_rank"] = np.nan
    for event, group in merged.groupby("event_id"):
        idx = group.index
        merged.loc[idx, "v5_rank"] = percentile_rank(group.prediction.to_numpy(), group.thread_id.tolist())
        merged.loc[idx, "aq_rank"] = percentile_rank(group.calibrated_two_expert_score.to_numpy(), group.thread_id.tolist())
    return merged, eligible


def event_curve(frame: pd.DataFrame, score_col: str, budgets: tuple[int, ...] = BUDGETS) -> dict[int, dict[str, float]]:
    """Per-event reduction and oracle_efficiency at each budget; identical convention to the rest of the repo."""
    ranked = frame.sort_values([score_col, "thread_id"], ascending=[False, True], kind="stable")
    oracle = frame.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable")
    total = float(frame.preventable_impact.sum())
    result = {}
    for budget in budgets:
        actual = min(budget, len(frame))
        blocked = float(ranked.preventable_impact.iloc[:actual].sum())
        optimal = float(oracle.preventable_impact.iloc[:actual].sum())
        result[int(budget)] = {
            "reduction": blocked / total if total else 0.0,
            "oracle_efficiency": blocked / optimal if optimal else 0.0,
        }
    return result


def blended_score(frame: pd.DataFrame, weight: float) -> np.ndarray:
    return weight * frame.v5_rank.to_numpy() + (1.0 - weight) * frame.aq_rank.to_numpy()


def select_weight(frame: pd.DataFrame, eligible_events: list[str], outer_event: str) -> tuple[float, pd.DataFrame, bool]:
    """Leave-one-event-out meta-selection using only the six other eligible events."""
    inner_events = [event for event in eligible_events if event != outer_event]
    records = []
    for event in inner_events:
        event_frame = frame.loc[frame.event_id == event]
        for weight in WEIGHTS:
            score = blended_score(event_frame, weight)
            curve = event_curve(event_frame.assign(_score=score), "_score")
            for budget, values in curve.items():
                records.append({"inner_event": event, "weight": weight, "budget": budget, **values})
    detail = pd.DataFrame(records)
    median = detail.groupby(["weight", "budget"], as_index=False)[["reduction", "oracle_efficiency"]].median()
    pivot_eff = median.pivot(index="weight", columns="budget", values="oracle_efficiency")
    baseline_ref = pivot_eff.loc[[0.0, 1.0]].max(axis=0)
    guard_ok = (pivot_eff.sub(baseline_ref, axis=1) >= -GUARD_TOLERANCE).all(axis=1)
    gain = pivot_eff.sub(baseline_ref, axis=1).mean(axis=1)
    candidates = pivot_eff.index[guard_ok]
    used_fallback = False
    if len(candidates) == 0:
        used_fallback = True
        end_means = pivot_eff.loc[[0.0, 1.0]].mean(axis=1)
        chosen = float(end_means.idxmax())
    else:
        candidate_gain = gain.loc[candidates]
        best_gain = candidate_gain.max()
        tied = sorted(float(w) for w in candidate_gain[candidate_gain == best_gain].index)
        chosen = tied[0]
    median["outer_event"] = outer_event
    median["guard_ok"] = median.weight.map(lambda w: bool(guard_ok.loc[w]))
    median["gain_vs_best_single_model"] = median.weight.map(lambda w: float(gain.loc[w]))
    return chosen, median, used_fallback


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_full_policy_stacking_ensemble")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"v5_text_rf_oof": str(V5_OOF), "active_quiet_calibrated_rf_oof": str(AQ_OOF)},
        "cutoff_seconds": 1800,
        "candidate_pool": "protected v5 rumour-only strict-30 snapshot, 2402 threads, 7 eligible events",
        "method": "within-event percentile-rank fusion of two already-OOF models; no new training or features",
        "weights": list(WEIGHTS), "budgets": list(BUDGETS), "guard_tolerance": GUARD_TOLERANCE,
        "selection_metric": "leave-one-event-out meta-selection; per-inner-event MEDIAN oracle_efficiency; guard on every budget vs max(weight=0.0, weight=1.0)",
        "hypothesis": (
            "v5_text_rf and active_quiet_calibrated_rf are independently trained on the same "
            "protected candidate pool and have never been blended. If their OOF errors are "
            "partially uncorrelated, a leave-one-event-out-selected rank fusion should get "
            "closer to oracle (higher oracle_efficiency@K) than either alone, with no new "
            "features, no retraining, and no risk beyond what both source models already carry."
        ),
        "research_safety": (
            "No model is retrained. Both source OOF files are themselves nested-nested-LOEO "
            "out-of-fold predictions where each thread was scored by a model that never saw its "
            "own event during training. This script adds one more leave-one-event-out layer "
            "purely to select the blend weight: for each outer event, weight selection uses only "
            "the OOF scores of the other six eligible events; the outer event's own scores never "
            "participate in selecting the weight applied to it, and are read only once for final "
            "evaluation."
        ),
    }
    write_record(output, record)
    try:
        print("Starting full-policy stacking ensemble (v5_text_rf + active_quiet_calibrated_rf).", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/5] Loading and validating both OOF files...", flush=True)
        frame, eligible = load_and_join_oof()
        print(f"  joined {len(frame)} threads across {len(eligible)} eligible events: {eligible}", flush=True)
        print("[2/5] Running leave-one-event-out weight selection...", flush=True)
        outer_rows, selection_frames, fold_details = [], [], []
        for index, outer_event in enumerate(eligible, 1):
            print(f"  outer {index}/{len(eligible)}: {outer_event}", flush=True)
            weight, selection, used_fallback = select_weight(frame, eligible, outer_event)
            selection_frames.append(selection)
            outer_frame = frame.loc[frame.event_id == outer_event]
            score = blended_score(outer_frame, weight)
            curve = event_curve(outer_frame.assign(_score=score), "_score")
            for budget, values in curve.items():
                outer_rows.append({"outer_event": outer_event, "model": "ensemble", "budget": budget, "selected_weight": weight, **values})
            v5_curve = event_curve(outer_frame.assign(_score=outer_frame.v5_rank), "_score")
            aq_curve = event_curve(outer_frame.assign(_score=outer_frame.aq_rank), "_score")
            for budget in BUDGETS:
                outer_rows.append({"outer_event": outer_event, "model": "v5_text_rf_only", "budget": budget, "selected_weight": 1.0, **v5_curve[budget]})
                outer_rows.append({"outer_event": outer_event, "model": "active_quiet_calibrated_rf_only", "budget": budget, "selected_weight": 0.0, **aq_curve[budget]})
            fold_details.append({"outer_event": outer_event, "selected_weight": weight, "used_fallback": used_fallback, "n_threads": len(outer_frame)})
            print(f"    selected weight={weight:.2f} (fallback={used_fallback})", flush=True)
        print("[3/5] Aggregating macro reduction/oracle_efficiency curves...", flush=True)
        outer_curves = pd.DataFrame(outer_rows)
        macro = outer_curves.groupby(["model", "budget"], as_index=False)[["reduction", "oracle_efficiency"]].mean()
        macro = macro.rename(columns={"reduction": "mean_reduction", "oracle_efficiency": "mean_oracle_efficiency"})
        comparison = macro.pivot(index="budget", columns="model", values=["mean_reduction", "mean_oracle_efficiency"])
        comparison.columns = [f"{metric}_{model}" for metric, model in comparison.columns]
        comparison = comparison.reset_index()
        comparison["ensemble_minus_best_single_reduction"] = comparison["mean_reduction_ensemble"] - comparison[["mean_reduction_v5_text_rf_only", "mean_reduction_active_quiet_calibrated_rf_only"]].max(axis=1)
        comparison["ensemble_minus_best_single_oracle_efficiency"] = comparison["mean_oracle_efficiency_ensemble"] - comparison[["mean_oracle_efficiency_v5_text_rf_only", "mean_oracle_efficiency_active_quiet_calibrated_rf_only"]].max(axis=1)
        print("[4/5] Validating output integrity...", flush=True)
        if not np.isfinite(outer_curves[["reduction", "oracle_efficiency"]].to_numpy(float)).all():
            raise ValueError("Non-finite reduction/oracle_efficiency in ensemble evaluation")
        v5_only_reduction_10 = float(macro.loc[(macro.model == "v5_text_rf_only") & (macro.budget == 10), "mean_reduction"].iloc[0])
        aq_only_reduction_10 = float(macro.loc[(macro.model == "active_quiet_calibrated_rf_only") & (macro.budget == 10), "mean_reduction"].iloc[0])
        print(f"  integrity check: v5-only reduction@10={v5_only_reduction_10:.4f} (reference ~0.1467), active/quiet-only reduction@10={aq_only_reduction_10:.4f} (reference ~0.1461)", flush=True)
        print("[5/5] Writing evidence...", flush=True)
        outer_curves.to_csv(output / "outer_scores_by_model.csv", index=False)
        macro.to_csv(output / "macro_curve.csv", index=False)
        comparison.to_csv(output / "ensemble_vs_single_model_comparison.csv", index=False)
        pd.concat(selection_frames, ignore_index=True).to_csv(output / "inner_weight_selection.csv", index=False)
        record.update({
            "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
            "eligible_events": eligible, "fold_details": fold_details,
            "integrity_check": {"v5_only_reduction_at_10": v5_only_reduction_10, "active_quiet_only_reduction_at_10": aq_only_reduction_10},
            "output_files": ["outer_scores_by_model.csv", "macro_curve.csv", "ensemble_vs_single_model_comparison.csv", "inner_weight_selection.csv"],
        })
        write_record(output, record)
        print(f"SUCCESS: full-policy stacking ensemble saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({
            "status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
        })
        write_record(output, record)
        print(f"FAILURE: full-policy stacking ensemble preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
