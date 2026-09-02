"""Full eligible-event LOEO quiet classifier for event-global Oracle Top-50."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd

import run_strict30_quiet_global_top50_smoke as common


common.RUN_MODE = "full"
common.N_ESTIMATORS = 300


def main() -> None:
    output = common.OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_strict30_quiet_global_top50_full")
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "mode": "full", "cutoff_seconds": 1800, "label": "event-global Oracle Top-50 membership; outcome-only", "split": "seven eligible outer LOEO; inner LOEO selects source/temporal fusion", "source_weights": list(common.SOURCE_WEIGHTS), "trees": common.N_ESTIMATORS, "seed": common.SEED}
    common._record(output, record)
    try:
        print("Starting strict-30 quiet global-Top50 classifier full.", flush=True); print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/5] Loading protected features and snapshots...", flush=True)
        if json.loads(common.MATERIALIZATION_RECORD.read_text(encoding="utf-8")).get("status") != "complete": raise ValueError("materialization incomplete")
        frame = pd.read_csv(common.FEATURE_TABLE, dtype={"thread_id": str, "event_id": str})
        if len(frame) != 2402 or frame.thread_id.duplicated().any(): raise ValueError("invalid feature table")
        columns = [column for column in frame if common.feature_group(column) in {"activity_level", "temporal_dynamics"}]
        frame = common._global_top50(frame); sizes = frame.groupby("event_id").size(); eligible = sorted(sizes[sizes >= 100].index.tolist())
        graphs = common.torch.load(common.DATASET, weights_only=False); recency, embeddings = common._embeddings(graphs); frame = frame.loc[frame.event_id.isin(eligible)].copy()
        all_metrics, all_predictions, all_selection = [], [], []
        print("[2/5] Running nested LOEO source/temporal selection...", flush=True)
        for outer_index, outer_event in enumerate(eligible, 1):
            print(f"  outer {outer_index}/{len(eligible)}: {outer_event}", flush=True)
            inner_events = [event for event in eligible if event != outer_event]; rows = []
            for weight in common.SOURCE_WEIGHTS:
                fold = []
                for inner_index, event in enumerate(inner_events, 1):
                    train, test, _ = common._quiet_split(frame, recency, [value for value in inner_events if value != event], event)
                    fold.append(common._metrics(test, common._scores(train, test, columns, embeddings, weight, common.SEED + outer_index * 1000 + inner_index)))
                rows.append({"outer_event": outer_event, "source_weight": weight, "inner_folds": len(fold), "mean_inner_ap": float(np.mean([item["quiet_global_top50_ap"] for item in fold])), "mean_inner_recall": float(np.mean([item["quiet_global_top50_recall_at_oracle_quiet_k"] for item in fold]))})
            selection = pd.DataFrame(rows).sort_values(["mean_inner_ap", "mean_inner_recall", "source_weight"], ascending=[False, False, True], kind="stable")
            weight = float(selection.iloc[0].source_weight); all_selection.append(selection)
            print(f"    selected source_weight={weight:.2f}", flush=True)
            train, test, threshold = common._quiet_split(frame, recency, inner_events, outer_event)
            score = common._scores(train, test, columns, embeddings, weight, common.SEED + 100000 + outer_index); metrics = common._metrics(test, score)
            metrics.update({"outer_event": outer_event, "selected_source_weight": weight, "quiet_threshold": threshold, "train_quiet_threads": len(train), "test_quiet_threads": len(test)})
            prediction = test[["thread_id", "event_id", "preventable_impact", "oracle_global_top50"]].copy(); prediction["global_top50_score"] = score; prediction["outer_event"] = outer_event; prediction["selected_source_weight"] = weight
            all_metrics.append(metrics); all_predictions.append(prediction)
            pd.concat(all_predictions, ignore_index=True).to_csv(output / "outer_quiet_predictions_partial.csv", index=False)
            (output / "partial_result.json").write_text(json.dumps({"completed_outer_events": [item["outer_event"] for item in all_metrics]}, indent=2), encoding="utf-8")
        print("[3/5] Aggregating quiet global-Top50 results...", flush=True)
        metrics_frame = pd.DataFrame(all_metrics); predictions_frame = pd.concat(all_predictions, ignore_index=True)
        print("[4/5] Writing predictions and selections...", flush=True)
        metrics_frame.to_csv(output / "outer_quiet_global_top50_metrics.csv", index=False); predictions_frame.to_csv(output / "outer_quiet_predictions.csv", index=False); pd.concat(all_selection, ignore_index=True).to_csv(output / "inner_source_weight_selection.csv", index=False)
        print("[5/5] Finalizing full run record...", flush=True)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "eligible_events": eligible, "metrics": all_metrics, "output_files": ["outer_quiet_global_top50_metrics.csv", "outer_quiet_predictions.csv", "inner_source_weight_selection.csv"]}); common._record(output, record)
        print(f"SUCCESS: strict-30 quiet global-Top50 full saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}"}); common._record(output, record); print(f"FAILURE: output preserved at: {output.resolve()}", flush=True); raise


if __name__ == "__main__": main()
