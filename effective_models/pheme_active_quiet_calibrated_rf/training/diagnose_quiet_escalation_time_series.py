"""Read-only event-balanced reply escalation time series for quiet threads."""
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
TOP_K = 50
BINS = ((0, 600, "0–10m"), (600, 1800, "10–30m"), (1800, 3600, "30–60m"), (3600, 7200, "60–120m"), (7200, 14400, "120–240m"), (14400, np.inf, "240m+"))

def save(output: Path, value: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(value, indent=2), encoding="utf-8")

def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_quiet_escalation_time_series")
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "purpose": "read-only full-timeline escalation timing diagnostic", "bins_seconds": [list(x[:2]) for x in BINS], "research_safety": "All post-30-minute bins are explanatory only. They are never used as early model inputs, labels, selectors, or normalizers."}
    save(output, record)
    try:
        print("Starting quiet escalation time-series diagnostic (read-only).", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/4] Loading event-local quiet Oracle labels...", flush=True)
        scores = pd.read_csv(SCORES, dtype={"thread_id": str, "event_id": str})
        labels = []
        for event_id, event in scores.groupby("event_id", sort=True):
            oracle = set(event.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable").head(TOP_K).thread_id)
            labels.extend({"thread_id": tid, "event_id": event_id, "quiet_oracle_top50": bool(tid in oracle)} for tid in event.loc[event.quiet, "thread_id"])
        labels = pd.DataFrame(labels)
        print("[2/4] Assigning every observed reply to a time bin...", flush=True)
        raw = pd.read_csv(REPLIES, dtype={"thread_id": str, "tweet_id": str, "parent_id": str, "event_id": str}, usecols=["thread_id", "tweet_id", "parent_id", "is_source", "offset_sec", "event_id"])
        sources = raw.loc[raw.is_source.eq(1), ["thread_id", "event_id", "tweet_id"]].rename(columns={"tweet_id": "source_tweet_id"})
        replies = raw.loc[raw.is_source.ne(1) & raw.offset_sec.notna()].merge(sources, on=["thread_id", "event_id"], how="inner", validate="many_to_one").merge(labels, on=["thread_id", "event_id"], how="inner", validate="many_to_one")
        replies["bin"] = pd.cut(replies.offset_sec, bins=[x[0] for x in BINS] + [np.inf], labels=[x[2] for x in BINS], right=False)
        replies["reply_to_reply"] = replies.parent_id.ne(replies.source_tweet_id)
        print("[3/4] Computing event-balanced per-thread reply rates by time band...", flush=True)
        grid = labels.assign(_key=1).merge(pd.DataFrame({"bin": [x[2] for x in BINS], "_key": 1}), on="_key").drop(columns="_key")
        counts = replies.groupby(["thread_id", "event_id", "quiet_oracle_top50", "bin"], observed=False).agg(reply_count=("tweet_id", "size"), reply_to_reply_count=("reply_to_reply", "sum")).reset_index()
        data = grid.merge(counts, on=["thread_id", "event_id", "quiet_oracle_top50", "bin"], how="left").fillna({"reply_count": 0, "reply_to_reply_count": 0})
        event_mean = data.groupby(["event_id", "quiet_oracle_top50", "bin"], observed=False)[["reply_count", "reply_to_reply_count"]].mean().reset_index()
        macro = event_mean.groupby(["quiet_oracle_top50", "bin"], observed=False)[["reply_count", "reply_to_reply_count"]].mean().reset_index()
        print("[4/4] Rendering time series and writing evidence...", flush=True)
        order = [x[2] for x in BINS]; fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
        for value, label, axis in (("reply_count", "Mean new replies per quiet thread", axes[0]), ("reply_to_reply_count", "Mean new reply-to-reply edges per quiet thread", axes[1])):
            for group, name in ((False, "other quiet"), (True, "quiet Oracle Top-50")):
                view = macro.loc[macro.quiet_oracle_top50.eq(group)].set_index("bin").reindex(order)
                axis.plot(order, view[value], marker="o", label=name)
            axis.set_title(label); axis.set_xlabel("Time since source"); axis.set_ylabel("Event-balanced mean"); axis.legend()
        fig.suptitle("When do quiet high-impact threads diverge from other quiet threads?")
        figure = output / "quiet_escalation_time_series.png"; fig.savefig(figure, dpi=180); plt.close(fig)
        macro.to_csv(output / "event_balanced_time_series.csv", index=False); event_mean.to_csv(output / "per_event_time_series.csv", index=False)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "quiet_threads": len(labels), "quiet_oracle_top50": int(labels.quiet_oracle_top50.sum()), "output_files": ["quiet_escalation_time_series.png", "event_balanced_time_series.csv", "per_event_time_series.csv"]}); save(output, record)
        print(f"SUCCESS: quiet escalation time-series saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}); save(output, record); print(f"FAILURE: quiet escalation time-series preserved at: {output.resolve()}", flush=True); raise

if __name__ == "__main__":
    main()
