"""Re-score saved outer predictions with deterministic top-K tie-breaking.

This is a post-hoc sensitivity audit only.  It never changes the source full
run, and it cannot retroactively re-run that experiment's inner selection.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import traceback

import pandas as pd


BASE = Path(__file__).resolve().parent
SOURCE_RUN = BASE / "experiments" / "20260902_031535_048090_strict30_clean_quiet_tier_full"
OUT_ROOT = BASE / "experiments"
HIGH_FRACTION = 0.20


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_saved_full_tie_break_audit")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "post-hoc deterministic top-K tie-break sensitivity audit; no fitting and no source-result modification",
        "source_run": str(SOURCE_RUN.resolve()),
        "tie_break": "prob_high descending, then thread_id ascending",
        "limitation": "Cannot repair inner feature-group selection because inner per-thread probabilities were not saved.",
    }
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        print("Starting saved full-run deterministic tie-break audit.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/3] Loading saved outer predictions and metrics...", flush=True)
        predictions = pd.read_csv(SOURCE_RUN / "outer_tier_predictions.csv", dtype={"thread_id": str, "outer_event": str})
        old = pd.read_csv(SOURCE_RUN / "outer_fold_metrics.csv", dtype={"outer_event": str})
        print("[2/3] Recomputing deterministic top-20% overlaps...", flush=True)
        rows = []
        for event, frame in predictions.groupby("outer_event", sort=True):
            k = max(1, math.ceil(len(frame) * HIGH_FRACTION))
            selected = frame.sort_values(["prob_high", "thread_id"], ascending=[False, True], kind="stable").head(k)
            new_overlap = float(selected.tier.eq("high").mean())
            old_overlap = float(old.loc[old.outer_event.eq(event), "high_top_fraction_overlap"].iloc[0])
            boundary = float(selected.prob_high.iloc[-1])
            rows.append({
                "outer_event": event, "quiet_threads": len(frame), "selected_k": k,
                "old_row_order_overlap": old_overlap, "deterministic_overlap": new_overlap,
                "difference": new_overlap - old_overlap,
                "boundary_probability": boundary,
                "boundary_tie_count": int((frame.prob_high == boundary).sum()),
                "boundary_selected_count": int((selected.prob_high == boundary).sum()),
            })
        audit = pd.DataFrame(rows)
        print("[3/3] Writing sensitivity evidence and run record...", flush=True)
        audit.to_csv(output / "deterministic_outer_overlap_audit.csv", index=False)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "metrics": {"mean_old_row_order_overlap": float(audit.old_row_order_overlap.mean()), "mean_deterministic_overlap": float(audit.deterministic_overlap.mean()), "mean_difference": float(audit.difference.mean()), "events_with_boundary_ties": int((audit.boundary_tie_count > 1).sum())}, "output_files": ["deterministic_outer_overlap_audit.csv"]})
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"SUCCESS: saved full-run deterministic tie-break audit saved to: {output.resolve()}", flush=True)
    except Exception as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"FAILURE: tie-break audit preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
