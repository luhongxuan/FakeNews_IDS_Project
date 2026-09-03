"""Full event-separated evaluation of cumulative and window/delta pooled RFs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import joblib
import pandas as pd

import multicheckpoint_model_common as common


HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parent / "experiments"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_multicheckpoint_models_full")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "started_at": now(), "mode": "full event-separated LOEO",
        "models": list(common.FAMILIES), "model_params": common.MODEL_PARAMS,
        "budgets": list(common.BUDGETS), "eligible_min_threads": 100, "seed": 42,
    }
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        print("Starting full cumulative/window-delta multi-checkpoint RF evaluation.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/6] Loading latest complete full multi-checkpoint materialization...", flush=True)
        source = common.latest_complete_materialization("full")
        frame, schema = common.load_materialization(source)
        events = sorted(frame.event_id.unique())
        thread_counts = frame[["thread_id", "event_id"]].drop_duplicates().groupby("event_id").size()
        eligible = sorted(thread_counts.loc[thread_counts.ge(100)].index)
        record.update({
            "input": str(source.resolve()), "split": "outer leave-one-event-out",
            "events": events, "eligible_events": eligible,
            "checkpoints_seconds": sorted(frame.checkpoint_sec.unique().astype(int).tolist()),
            "target": "log1p(corrected dynamic preventable impact)",
            "research_safety": (
                "Every outer test event is excluded from fitting at all checkpoints. Feature families "
                "are predeclared and compared without outer-test-driven selection."
            ),
        })
        print("[2/6] Validating both feature allowlists and event separation...", flush=True)
        family_columns = {family: common.family_columns(schema, family) for family in common.FAMILIES}
        all_predictions, all_metrics = [], []
        print("[3/6] Running outer LOEO folds for both model families...", flush=True)
        for outer_index, event in enumerate(events, 1):
            train = frame.loc[frame.event_id.ne(event)].copy()
            test = frame.loc[frame.event_id.eq(event)].copy()
            print(f"  outer event {outer_index}/{len(events)}: {event}", flush=True)
            for family in common.FAMILIES:
                print(f"    fitting {family}", flush=True)
                _, prediction = common.fit_predict(train, test, family_columns[family])
                scored = test[[
                    "thread_id", "event_id", "checkpoint_sec", "dynamic_preventable_impact",
                    "dynamic_preventable_y", "future_growth",
                ]].copy()
                scored["family"] = family
                scored["prediction"] = prediction
                all_predictions.append(scored)
                all_metrics.extend(common.metric_rows(scored, family))
            pd.concat(all_predictions, ignore_index=True).to_csv(
                output / "oof_predictions_partial.csv", index=False
            )
        print("[4/6] Aggregating checkpoint-specific official-event metrics...", flush=True)
        predictions = pd.concat(all_predictions, ignore_index=True)
        metrics = pd.DataFrame(all_metrics)
        official = metrics.loc[metrics.event_id.isin(eligible)].copy()
        macro = official.groupby(["family", "checkpoint_sec", "budget"], as_index=False).agg(
            eligible_events=("event_id", "nunique"),
            total_oracle_top50_hits=("oracle_top50_hits", "sum"),
            mean_crr=("crr", "mean"),
            mean_preventable_recall=("preventable_recall", "mean"),
            mean_oracle_efficiency=("oracle_efficiency", "mean"),
        )
        print("[5/6] Fitting and saving both all-event deployment candidates...", flush=True)
        for family in common.FAMILIES:
            columns = family_columns[family]
            model, _ = common.fit_predict(
                frame.loc[~frame.event_id.eq(events[-1])],
                frame.loc[frame.event_id.eq(events[-1])], columns,
            )
            # Refit after pipeline validation; no held-out metric is used to configure this model.
            model.fit(frame[columns], frame.dynamic_preventable_y)
            joblib.dump({"model": model, "feature_columns": columns}, output / f"{family}_all_events.joblib")
        print("[6/6] Writing complete OOF results and run record...", flush=True)
        predictions.to_csv(output / "oof_predictions.csv", index=False)
        metrics.to_csv(output / "per_event_checkpoint_metrics.csv", index=False)
        macro.to_csv(output / "macro_checkpoint_metrics.csv", index=False)
        result = {
            "eligible_events": eligible,
            "macro_checkpoint_metrics": macro.to_dict(orient="records"),
            "selection_note": "No winning family is promoted using outer results.",
        }
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        record.update({
            "status": "complete", "completed_at": now(), "rows": len(frame),
            "threads": frame.thread_id.nunique(), "metrics_rows": len(metrics),
            "output_files": [
                "oof_predictions_partial.csv", "oof_predictions.csv",
                "per_event_checkpoint_metrics.csv", "macro_checkpoint_metrics.csv",
                "cumulative_all_events.joblib", "window_delta_all_events.joblib",
                "result.json", "run_record.json",
            ],
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"SUCCESS: full multi-checkpoint RF evaluation saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({
            "status": "failed", "failed_at": now(), "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(), "output_files": sorted(path.name for path in output.iterdir()),
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"FAILURE: multi-checkpoint RF evaluation preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
