"""Read-only diagnostic: can quiet tier pools retain global Oracle Top-50 threads?"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
TIER_RUN = ROOT / "experiments" / "20260902_114211_166148_strict30_quiet_two_expert_tier_full"
FEATURE_TABLE = ROOT / "experiments" / "20260902_030441_849111_strict30_protected_feature_materialization" / "strict30_protected_snapshot_features.csv"
OUT_ROOT = ROOT / "experiments"
POOL_FRACTIONS = (0.20, 0.30, 0.40, 0.50, 0.60, 1.00)


def write_record(directory: Path, record: dict) -> None:
    (directory / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_existing_tier_global_top50_pool_diagnostic")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "read-only coverage diagnostic of existing held-out quiet-tier predictions",
        "tier_run": str(TIER_RUN), "feature_table": str(FEATURE_TABLE),
        "cutoff_seconds": 1800, "pool_fractions": list(POOL_FRACTIONS),
        "label": "event-global Oracle Top-50 by preventable impact; analysis only",
    }
    write_record(output, record)
    try:
        print("Starting existing quiet-tier global-Top50 pool diagnostic.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/4] Loading protected outcome table and existing held-out tier predictions...", flush=True)
        features = pd.read_csv(FEATURE_TABLE, dtype={"thread_id": str, "event_id": str})
        predictions = pd.read_csv(TIER_RUN / "outer_tier_predictions.csv", dtype={"thread_id": str, "event_id": str})
        if features.thread_id.duplicated().any() or predictions.thread_id.duplicated().any():
            raise ValueError("thread IDs must be unique in both inputs")
        print("[2/4] Constructing event-global Oracle Top-50 outcome label...", flush=True)
        features["oracle_global_top50"] = False
        for _, group in features.groupby("event_id", sort=True):
            top_ids = group.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable").head(50).thread_id
            features.loc[features.thread_id.isin(top_ids), "oracle_global_top50"] = True
        frame = predictions.merge(features[["thread_id", "oracle_global_top50"]], on="thread_id", how="left", validate="one_to_one")
        if frame.oracle_global_top50.isna().any():
            raise ValueError("tier predictions contain unknown thread IDs")
        frame.oracle_global_top50 = frame.oracle_global_top50.astype(bool)
        print("[3/4] Measuring candidate-pool coverage for each held-out event...", flush=True)
        rows = []
        for event_index, (event, group) in enumerate(frame.groupby("outer_event", sort=True), 1):
            oracle_count = int(group.oracle_global_top50.sum())
            if oracle_count == 0:
                raise ValueError(f"{event} has no quiet global Oracle Top-50 threads")
            print(f"  event {event_index}/7: {event} (quiet={len(group)}, oracle={oracle_count})", flush=True)
            ordered = group.sort_values(["prob_high", "thread_id"], ascending=[False, True], kind="stable")
            for fraction in POOL_FRACTIONS:
                size = min(len(ordered), max(1, round(len(ordered) * fraction)))
                selected = ordered.head(size)
                hits = int(selected.oracle_global_top50.sum())
                prevalence = oracle_count / len(ordered)
                precision = hits / size
                rows.append({"outer_event": event, "pool_fraction": fraction, "quiet_threads": len(ordered), "pool_size": size, "oracle_global_top50_quiet": oracle_count, "hits": hits, "coverage_recall": hits / oracle_count, "pool_precision": precision, "random_expected_precision": prevalence, "precision_enrichment": precision / prevalence})
        print("[4/4] Writing event and pooled summaries...", flush=True)
        detail = pd.DataFrame(rows)
        summary = detail.groupby("pool_fraction", as_index=False).agg(pool_size=("pool_size", "sum"), oracle_global_top50_quiet=("oracle_global_top50_quiet", "sum"), hits=("hits", "sum"))
        summary["coverage_recall"] = summary.hits / summary.oracle_global_top50_quiet
        summary["pool_precision"] = summary.hits / summary.pool_size
        total_quiet = int(frame.shape[0]); total_oracle = int(frame.oracle_global_top50.sum())
        summary["random_expected_precision"] = total_oracle / total_quiet
        summary["precision_enrichment"] = summary.pool_precision / summary.random_expected_precision
        detail.to_csv(output / "per_event_pool_coverage.csv", index=False)
        summary.to_csv(output / "pooled_pool_coverage.csv", index=False)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "quiet_threads": total_quiet, "quiet_global_oracle_top50": total_oracle, "output_files": ["per_event_pool_coverage.csv", "pooled_pool_coverage.csv"]})
        write_record(output, record)
        print(f"SUCCESS: existing quiet-tier global-Top50 pool diagnostic saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}"})
        write_record(output, record)
        print(f"FAILURE: output preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
