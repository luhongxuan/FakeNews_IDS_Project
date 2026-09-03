"""Read-only before/after reply-tree escalation diagnostic for quiet threads."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCORES = HERE.parent / "experiments" / "20260902_132700_413954_quiet_tier_hybrid_policy_full" / "outer_scores.csv"
REPLIES = ROOT / "pheme_reply_level.csv"
OUT_ROOT = HERE.parent / "experiments"
CUTOFF_SECONDS = 1800.0
TOP_K = 50
FEATURES = ("log1p_reply_count", "max_depth", "reply_to_reply_rate", "parent_diversity", "max_branching", "log1p_time_span_seconds")


def write_record(output: Path, value: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(value, indent=2), encoding="utf-8")


def thread_metrics(rows: pd.DataFrame, label_pool: pd.DataFrame, phase: str) -> pd.DataFrame:
    phase_rows = rows.loc[(rows.offset_sec <= CUTOFF_SECONDS) if phase == "observed_0_30m" else (rows.offset_sec > CUTOFF_SECONDS)].copy()
    if phase_rows.empty: return label_pool.assign(phase=phase, **{feature: 0.0 for feature in FEATURES})
    source_ids = rows.loc[rows.is_source.eq(1), ["thread_id", "event_id", "tweet_id"]].rename(columns={"tweet_id": "source_tweet_id"})
    replies = phase_rows.loc[phase_rows.is_source.ne(1)].merge(source_ids, on=["thread_id", "event_id"], how="left", validate="many_to_one")
    if replies.empty: return label_pool.assign(phase=phase, **{feature: 0.0 for feature in FEATURES})
    replies["reply_to_reply"] = replies.parent_id.ne(replies.source_tweet_id)
    child_counts = replies.groupby(["thread_id", "event_id", "parent_id"]).size().groupby(level=[0, 1]).max().rename("max_branching")
    grouped = replies.groupby(["thread_id", "event_id"]).agg(reply_count=("tweet_id", "size"), max_depth=("depth", "max"), reply_to_reply_rate=("reply_to_reply", "mean"), parent_count=("parent_id", "nunique"), first_reply_offset=("offset_sec", "min"), last_reply_offset=("offset_sec", "max")).join(child_counts).reset_index()
    grouped["log1p_reply_count"] = np.log1p(grouped.reply_count)
    grouped["parent_diversity"] = grouped.parent_count / grouped.reply_count
    grouped["log1p_time_span_seconds"] = np.log1p(grouped.last_reply_offset - grouped.first_reply_offset)
    result = label_pool.merge(grouped[["thread_id", "event_id", *FEATURES]], on=["thread_id", "event_id"], how="left")
    result[list(FEATURES)] = result[list(FEATURES)].fillna(0.0)
    return result.assign(phase=phase)


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_quiet_conversation_escalation_diagnostic")
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "purpose": "read-only full-timeline then strict-30 reply-tree escalation diagnostic", "cutoff_seconds": CUTOFF_SECONDS, "features": list(FEATURES), "research_safety": "The after-30-minute panel is outcome explanation only. Only the observed 0–30-minute panel could ever motivate an early feature, after separate validation."}
    write_record(output, record)
    try:
        print("Starting quiet conversation escalation diagnostic (read-only).", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/5] Loading quiet Oracle diagnostic labels...", flush=True)
        scores = pd.read_csv(SCORES, dtype={"thread_id": str, "event_id": str})
        labels = []
        for event_id, event in scores.groupby("event_id", sort=True):
            oracle = set(event.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable").head(TOP_K).thread_id)
            labels.extend({"thread_id": thread_id, "event_id": event_id, "quiet_oracle_top50": bool(thread_id in oracle)} for thread_id in event.loc[event.quiet, "thread_id"])
        labels = pd.DataFrame(labels)
        print("[2/5] Loading reply trees and constructing before/after metrics...", flush=True)
        raw = pd.read_csv(REPLIES, dtype={"thread_id": str, "tweet_id": str, "parent_id": str, "event_id": str}, usecols=["thread_id", "tweet_id", "parent_id", "is_source", "offset_sec", "depth", "event_id"])
        raw = raw.loc[raw.offset_sec.notna()].copy()
        per_thread = pd.concat([thread_metrics(raw, labels, "observed_0_30m"), thread_metrics(raw, labels, "future_after_30m")], ignore_index=True)
        print("[3/5] Computing event-balanced Oracle-minus-control differences...", flush=True)
        per_event = per_thread.groupby(["phase", "event_id", "quiet_oracle_top50"])[list(FEATURES)].mean().reset_index()
        macro = per_event.groupby(["phase", "quiet_oracle_top50"])[list(FEATURES)].mean().reset_index()
        pieces = []
        for phase, values in macro.groupby("phase"):
            indexed = values.set_index("quiet_oracle_top50")
            diff = indexed.loc[True, list(FEATURES)] - indexed.loc[False, list(FEATURES)]
            pieces.extend({"phase": phase, "feature": feature, "oracle_minus_other": float(value)} for feature, value in diff.items())
        contrast = pd.DataFrame(pieces)
        print("[4/5] Rendering full-timeline versus strict-30 comparison...", flush=True)
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True, sharey=True)
        for axis, phase in zip(axes, ("future_after_30m", "observed_0_30m")):
            view = contrast.loc[contrast.phase.eq(phase)].sort_values("oracle_minus_other")
            axis.barh(view.feature, view.oracle_minus_other)
            axis.axvline(0, color="black", linewidth=.8)
            axis.set_title("After 30 min: explanatory only" if phase == "future_after_30m" else "0–30 min: allowable early signal")
            axis.set_xlabel("Event-balanced mean difference\nquiet Oracle Top-50 − other quiet")
        fig.suptitle("Does later reply-tree escalation leave an early structural precursor?")
        figure = output / "quiet_conversation_escalation_before_after.png"; fig.savefig(figure, dpi=180); plt.close(fig)
        print("[5/5] Writing metrics and reproducibility record...", flush=True)
        per_thread.to_csv(output / "per_thread_before_after_escalation_metrics.csv", index=False); contrast.to_csv(output / "event_balanced_before_after_contrast.csv", index=False); per_event.to_csv(output / "per_event_escalation_means.csv", index=False)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "quiet_threads": len(labels), "quiet_oracle_top50": int(labels.quiet_oracle_top50.sum()), "output_files": ["quiet_conversation_escalation_before_after.png", "per_thread_before_after_escalation_metrics.csv", "event_balanced_before_after_contrast.csv", "per_event_escalation_means.csv"]}); write_record(output, record)
        print(f"SUCCESS: quiet conversation escalation diagnostic saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}); write_record(output, record); print(f"FAILURE: escalation diagnostic preserved at: {output.resolve()}", flush=True); raise


if __name__ == "__main__":
    main()
