"""Compare strict-30 rankers on one corrected target and Top-50 protocol."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT_ROOT = HERE.parent / "experiments"
CORRECTED_HYBRID = OUT_ROOT / "20260903_214222_297421_corrected_quiet_tier_hybrid_full"
LEGACY_HYBRID = (
    ROOT / "effective_models" / "pheme_active_quiet_calibrated_rf" / "experiments"
    / "20260902_132700_413954_quiet_tier_hybrid_policy_full" / "outer_scores.csv"
)
CORRECTED_V5 = (
    ROOT / "effective_models" / "pheme_v5_corrected_target" / "experiments"
    / "20260903_143643_168807_corrected_v5_nested_full" / "oof_thread_scores.csv"
)
TOP_K = 50


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    output = OUT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + "_corrected_top50_comparison"
    )
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "started_at": now(),
        "purpose": "same-target same-split strict-30 Top-50 comparison",
        "inputs": {
            "corrected_hybrid": str(CORRECTED_HYBRID.resolve()),
            "legacy_hybrid_scores": str(LEGACY_HYBRID.resolve()),
            "corrected_v5": str(CORRECTED_V5.resolve()),
        },
        "cutoff_seconds": 1800, "budget": TOP_K,
        "split": "same seven eligible outer events",
        "outcome": "corrected all-traversable-roots preventable impact",
        "research_safety": (
            "All ranks are existing outer-event OOF scores. Legacy outcomes are discarded; "
            "every model is evaluated on the corrected outcome from the corrected hybrid run."
        ),
    }
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        print("Starting corrected strict-30 Top-50 comparison.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/5] Loading corrected hybrid, frozen legacy hybrid, and corrected v5 OOF...", flush=True)
        base = pd.read_csv(
            CORRECTED_HYBRID / "outer_scores.csv",
            dtype={"thread_id": str, "event_id": str},
        )
        legacy = pd.read_csv(LEGACY_HYBRID, dtype={"thread_id": str, "event_id": str})
        v5 = pd.read_csv(CORRECTED_V5, dtype={"thread_id": str, "event_id": str})
        eligible = sorted(base.event_id.unique())
        legacy = legacy.loc[legacy.event_id.isin(eligible), [
            "thread_id", "event_id", "hybrid_calibrated_score"
        ]].rename(columns={"hybrid_calibrated_score": "legacy_hybrid_score"})
        v5 = v5.loc[v5.event_id.isin(eligible), [
            "thread_id", "event_id", "prediction"
        ]].rename(columns={"prediction": "corrected_v5_score"})

        print("[2/5] Joining scores by protected identity and auditing corrected outcomes...", flush=True)
        frame = base.merge(
            legacy, on=["thread_id", "event_id"], how="left", validate="one_to_one"
        ).merge(v5, on=["thread_id", "event_id"], how="left", validate="one_to_one")
        score_columns = {
            "corrected_active_quiet": "baseline_calibrated_score",
            "corrected_quiet_tier_hybrid": "hybrid_calibrated_score",
            "frozen_legacy_quiet_tier_hybrid": "legacy_hybrid_score",
            "corrected_v5": "corrected_v5_score",
        }
        if len(frame) != 2327 or frame[list(score_columns.values())].isna().any().any():
            raise ValueError("Top-50 comparison identity/score join failed")
        if not np.isfinite(
            frame[["preventable_impact", *score_columns.values()]].to_numpy(dtype=float)
        ).all():
            raise ValueError("non-finite corrected comparison value")

        print("[3/5] Selecting Top-50 and measuring active/quiet Oracle overlap...", flush=True)
        metric_rows, selected_rows = [], []
        for event_id, event in frame.groupby("event_id", sort=True):
            oracle = event.sort_values(
                ["preventable_impact", "thread_id"],
                ascending=[False, True], kind="stable",
            ).head(TOP_K)
            oracle_ids = set(oracle.thread_id)
            quiet_oracle_ids = set(oracle.loc[oracle.quiet, "thread_id"])
            total_impact = float(event.preventable_impact.sum())
            oracle_impact = float(oracle.preventable_impact.sum())
            for model, score_column in score_columns.items():
                chosen = event.sort_values(
                    [score_column, "thread_id"],
                    ascending=[False, True], kind="stable",
                ).head(TOP_K).copy()
                chosen_ids = set(chosen.thread_id)
                quiet_chosen = chosen.loc[chosen.quiet]
                blocked = float(chosen.preventable_impact.sum())
                metric_rows.append({
                    "event_id": event_id, "model": model, "selected": len(chosen),
                    "blocked_impact": blocked,
                    "total_event_impact": total_impact,
                    "model_reduction": blocked / total_impact if total_impact else 0.0,
                    "oracle_top50_impact": oracle_impact,
                    "oracle_efficiency": blocked / oracle_impact if oracle_impact else 0.0,
                    "oracle_top50_hits": len(chosen_ids & oracle_ids),
                    "quiet_selected": len(quiet_chosen),
                    "active_selected": len(chosen) - len(quiet_chosen),
                    "quiet_oracle_top50": len(quiet_oracle_ids),
                    "quiet_oracle_hits": len(set(quiet_chosen.thread_id) & quiet_oracle_ids),
                    "quiet_selection_precision": (
                        len(set(quiet_chosen.thread_id) & quiet_oracle_ids) / len(quiet_chosen)
                        if len(quiet_chosen) else 0.0
                    ),
                    "quiet_oracle_recall": (
                        len(set(quiet_chosen.thread_id) & quiet_oracle_ids) / len(quiet_oracle_ids)
                        if quiet_oracle_ids else 0.0
                    ),
                })
                selected_rows.append(chosen.assign(
                    model=model,
                    oracle_top50=chosen.thread_id.isin(oracle_ids),
                    quiet_oracle_top50=chosen.thread_id.isin(quiet_oracle_ids),
                    rank=np.arange(1, len(chosen) + 1),
                )[[
                    "event_id", "model", "rank", "thread_id", "quiet",
                    "preventable_impact", "oracle_top50", "quiet_oracle_top50",
                ]])

        print("[4/5] Building macro and pairwise corrected-target comparisons...", flush=True)
        metrics = pd.DataFrame(metric_rows)
        selected = pd.concat(selected_rows, ignore_index=True)
        macro = metrics.groupby("model", as_index=False).agg(
            events=("event_id", "nunique"),
            mean_blocked_impact=("blocked_impact", "mean"),
            mean_model_reduction=("model_reduction", "mean"),
            mean_oracle_efficiency=("oracle_efficiency", "mean"),
            total_oracle_top50_hits=("oracle_top50_hits", "sum"),
            total_quiet_selected=("quiet_selected", "sum"),
            total_active_selected=("active_selected", "sum"),
            total_quiet_oracle_top50=("quiet_oracle_top50", "sum"),
            total_quiet_oracle_hits=("quiet_oracle_hits", "sum"),
            mean_quiet_selection_precision=("quiet_selection_precision", "mean"),
            mean_quiet_oracle_recall=("quiet_oracle_recall", "mean"),
        )
        hybrid = macro.loc[macro.model.eq("corrected_quiet_tier_hybrid")].iloc[0]
        pairwise_rows = []
        for row in macro.itertuples(index=False):
            if row.model == "corrected_quiet_tier_hybrid":
                continue
            pairwise_rows.append({
                "reference_model": row.model,
                "hybrid_minus_reference_reduction": (
                    float(hybrid.mean_model_reduction) - float(row.mean_model_reduction)
                ),
                "hybrid_minus_reference_oracle_efficiency": (
                    float(hybrid.mean_oracle_efficiency) - float(row.mean_oracle_efficiency)
                ),
                "hybrid_minus_reference_oracle_hits": (
                    int(hybrid.total_oracle_top50_hits) - int(row.total_oracle_top50_hits)
                ),
                "hybrid_minus_reference_quiet_hits": (
                    int(hybrid.total_quiet_oracle_hits) - int(row.total_quiet_oracle_hits)
                ),
            })
        pairwise = pd.DataFrame(pairwise_rows)

        print("[5/5] Writing complete comparison evidence and run record...", flush=True)
        metrics.to_csv(output / "per_event_top50_metrics.csv", index=False)
        macro.to_csv(output / "top50_macro.csv", index=False)
        pairwise.to_csv(output / "hybrid_pairwise_deltas.csv", index=False)
        selected.to_csv(output / "selected_top50_threads.csv", index=False)
        result = {
            "top50_macro": macro.to_dict(orient="records"),
            "hybrid_pairwise_deltas": pairwise.to_dict(orient="records"),
            "interpretation_boundary": (
                "The frozen legacy scorer was trained on legacy targets but is evaluated here on "
                "corrected outcomes. Corrected models use corrected training targets."
            ),
        }
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        record.update({
            "status": "complete", "completed_at": now(), "rows": len(frame),
            "events": eligible, "models": list(score_columns),
            "metrics": macro.to_dict(orient="records"),
            "output_files": [
                "per_event_top50_metrics.csv", "top50_macro.csv",
                "hybrid_pairwise_deltas.csv", "selected_top50_threads.csv",
                "result.json", "run_record.json",
            ],
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"SUCCESS: corrected Top-50 comparison saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({
            "status": "failed", "failed_at": now(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "output_files": sorted(path.name for path in output.iterdir()),
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"FAILURE: corrected Top-50 comparison preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
