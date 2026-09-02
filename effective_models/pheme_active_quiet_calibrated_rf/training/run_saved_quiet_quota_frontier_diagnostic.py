"""Read-only Top-50 quiet-quota frontier diagnostic from held-out policy scores."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE_RUN = HERE.parent / "experiments" / "20260902_142129_203276_quiet_tier_soft_boost_full"
SCORES = SOURCE_RUN / "outer_scores.csv"
OUT_ROOT = HERE.parent / "experiments"
BUDGET = 50
QUIET_QUOTAS = tuple(range(0, BUDGET + 1, 5))


def write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def select_quota(event: pd.DataFrame, quiet_quota: int, score_column: str) -> tuple[pd.DataFrame, int]:
    quiet = event.loc[event.quiet].sort_values([score_column, "thread_id"], ascending=[False, True], kind="stable")
    active = event.loc[~event.quiet].sort_values([score_column, "thread_id"], ascending=[False, True], kind="stable")
    # A fixed Top-50 quota must remain feasible when an event contains fewer
    # than 50 active or quiet threads.  Clip only to the event's feasible range.
    effective_quota = min(max(quiet_quota, BUDGET - len(active)), min(BUDGET, len(quiet)))
    chosen = pd.concat([quiet.head(effective_quota), active.head(BUDGET - effective_quota)], ignore_index=True)
    if len(chosen) != BUDGET or chosen.thread_id.duplicated().any(): raise ValueError("invalid quota selection")
    return chosen, effective_quota


def event_metrics(event: pd.DataFrame, chosen: pd.DataFrame) -> dict:
    oracle = event.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable").head(BUDGET)
    oracle_ids = set(oracle.thread_id); quiet_oracle = set(oracle.loc[oracle.quiet, "thread_id"])
    selected_ids = set(chosen.thread_id); selected_quiet_oracle = set(chosen.loc[chosen.quiet, "thread_id"])
    total = float(event.preventable_impact.sum())
    return {"model_blocked_impact": float(chosen.preventable_impact.sum()), "model_reduction": float(chosen.preventable_impact.sum()) / total if total else 0.0, "oracle_top50_hits": len(selected_ids & oracle_ids), "selected_quiet": int(chosen.quiet.sum()), "quiet_oracle_hits": len(selected_quiet_oracle & quiet_oracle), "quiet_oracle_total": len(quiet_oracle)}


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_saved_quiet_quota_frontier_diagnostic")
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "purpose": "read-only Top-50 quota frontier from saved held-out baseline scores", "source_run": str(SOURCE_RUN.resolve()), "budget": BUDGET, "quiet_quotas": list(QUIET_QUOTAS), "research_safety": "No model fitting, feature construction, labels, split, or scores are changed. Oracle rows are diagnostic ceilings only, never model inputs."}
    write_record(output, record)
    try:
        print("Starting saved quiet-quota frontier diagnostic.", flush=True); print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/4] Loading held-out baseline scores...", flush=True)
        frame = pd.read_csv(SCORES, dtype={"thread_id": str, "event_id": str})
        required = {"thread_id", "event_id", "preventable_impact", "baseline_calibrated_score", "quiet"}
        if required - set(frame) or frame.thread_id.duplicated().any(): raise ValueError("invalid held-out score artifact")
        if frame.event_id.nunique() != 7: raise ValueError("expected seven eligible held-out events")
        print("[2/4] Evaluating score-based and oracle quota frontiers...", flush=True)
        rows = []
        for quota in QUIET_QUOTAS:
            print(f"  quota {quota}/{BUDGET} quiet slots", flush=True)
            for event_id, event in frame.groupby("event_id", sort=True):
                score_choice, effective = select_quota(event, quota, "baseline_calibrated_score")
                oracle_choice, oracle_effective = select_quota(event, quota, "preventable_impact")
                if effective != oracle_effective: raise AssertionError("quota feasibility depends only on event composition")
                rows.append({"selection": "score_quota", "quiet_quota": quota, "effective_quiet_quota": effective, "event_id": event_id, **event_metrics(event, score_choice)})
                rows.append({"selection": "oracle_quota_ceiling", "quiet_quota": quota, "effective_quiet_quota": effective, "event_id": event_id, **event_metrics(event, oracle_choice)})
        print("[3/4] Computing unconstrained baseline reference...", flush=True)
        for event_id, event in frame.groupby("event_id", sort=True):
            choice = event.sort_values(["baseline_calibrated_score", "thread_id"], ascending=[False, True], kind="stable").head(BUDGET)
            rows.append({"selection": "unconstrained_baseline", "quiet_quota": -1, "effective_quiet_quota": int(choice.quiet.sum()), "event_id": event_id, **event_metrics(event, choice)})
        detail = pd.DataFrame(rows)
        summary = detail.groupby(["selection", "quiet_quota"], as_index=False).agg(eligible_events=("event_id", "nunique"), mean_model_reduction=("model_reduction", "mean"), total_oracle_top50_hits=("oracle_top50_hits", "sum"), total_selected_quiet=("selected_quiet", "sum"), total_quiet_oracle_hits=("quiet_oracle_hits", "sum"), total_quiet_oracle=("quiet_oracle_total", "sum"))
        summary["quiet_oracle_recall"] = summary.total_quiet_oracle_hits / summary.total_quiet_oracle
        print("[4/4] Writing quota frontier evidence...", flush=True)
        detail.to_csv(output / "per_event_quota_frontier.csv", index=False); summary.to_csv(output / "pooled_quota_frontier.csv", index=False)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "output_files": ["per_event_quota_frontier.csv", "pooled_quota_frontier.csv"]}); write_record(output, record)
        print(f"SUCCESS: saved quiet-quota frontier diagnostic saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}); write_record(output, record); print(f"FAILURE: quota frontier preserved at: {output.resolve()}", flush=True); raise


if __name__ == "__main__":
    main()
