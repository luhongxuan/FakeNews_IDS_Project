"""Read-only strict-30 heuristic stance audit for quiet threads.

This creates an interpretable case diagnostic, not a training label or model
feature.  Future impact supplies only the outcome grouping after extraction.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
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
CONTRADICT = re.compile(r"\b(fake|false|wrong|lie|lies|liar|hoax|nonsense|bullshit|debunk(?:ed|ing)?|misinformation|rumou?r|not\s+true|is(?:\s+not|n't)\s+true)\b", re.I)
SUPPORT = re.compile(r"\b(true|correct|right|confirmed|agree|exactly|indeed|yes|absolutely|thanks)\b", re.I)
QUESTION = re.compile(r"\?|\b(what|why|how|really|is\s+this|can\s+you|any\s+source)\b", re.I)


def record(output: Path, payload: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stance(text: str) -> str:
    value = str(text)
    if CONTRADICT.search(value): return "contradict_or_factcheck"
    if SUPPORT.search(value): return "support_or_affirm"
    if QUESTION.search(value): return "question_or_uncertainty"
    return "other_or_neutral"


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_quiet_early_stance_audit")
    output.mkdir(parents=True, exist_ok=False)
    info = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "purpose": "read-only strict-30 early reply stance diagnostic", "cutoff_seconds": CUTOFF_SECONDS, "outcome_group": "quiet thread that is event Oracle Top-50 by future preventable impact", "method": "transparent English lexical heuristic; exploratory only, not a validated stance classifier", "research_safety": "Only <=30-minute source/reply text is classified. Future impact labels the diagnostic groups after extraction and is never a feature or training target in this script."}
    record(output, info)
    try:
        print("Starting quiet early stance audit (read-only).", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/5] Loading saved quiet labels and event-local Oracle membership...", flush=True)
        scores = pd.read_csv(SCORES, dtype={"thread_id": str, "event_id": str})
        labels = []
        for event_id, event in scores.groupby("event_id", sort=True):
            oracle = set(event.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable").head(TOP_K).thread_id)
            labels.extend({"thread_id": thread_id, "event_id": event_id, "quiet_oracle_top50": bool(quiet and thread_id in oracle)} for thread_id, quiet in zip(event.loc[event.quiet, "thread_id"], event.loc[event.quiet, "quiet"]))
        labels = pd.DataFrame(labels)
        print(f"  quiet threads={len(labels)}, quiet Oracle Top-{TOP_K}={int(labels.quiet_oracle_top50.sum())}", flush=True)
        print("[2/5] Loading only observed 0–30 minute source/reply text...", flush=True)
        raw = pd.read_csv(REPLIES, dtype={"thread_id": str, "tweet_id": str, "parent_id": str, "event_id": str}, usecols=["thread_id", "tweet_id", "parent_id", "is_source", "is_text_available", "offset_sec", "text", "event_id"])
        raw = raw.loc[raw.is_text_available.eq(1) & raw.text.notna() & raw.offset_sec.notna() & raw.offset_sec.le(CUTOFF_SECONDS)].copy()
        sources = raw.loc[raw.is_source.eq(1), ["thread_id", "event_id", "tweet_id", "text"]].rename(columns={"tweet_id": "source_tweet_id", "text": "source_text"})
        replies = raw.loc[raw.is_source.ne(1)].merge(sources, on=["thread_id", "event_id"], how="inner", validate="many_to_one")
        replies = replies.merge(labels, on=["thread_id", "event_id"], how="inner", validate="many_to_one")
        replies["stance"] = replies.text.map(stance)
        replies["direct_to_source"] = replies.parent_id.eq(replies.source_tweet_id)
        if replies.empty: raise ValueError("No observed replies joined to quiet thread labels")
        print("[3/5] Aggregating support, contradiction, and question rates by event...", flush=True)
        categories = ["support_or_affirm", "contradict_or_factcheck", "question_or_uncertainty", "other_or_neutral"]
        table = replies.assign(count=1).pivot_table(index=["event_id", "quiet_oracle_top50"], columns="stance", values="count", aggfunc="sum", fill_value=0)
        for category in categories:
            if category not in table: table[category] = 0
        table["reply_count"] = table[categories].sum(axis=1)
        table["direct_to_source_rate"] = replies.groupby(["event_id", "quiet_oracle_top50"]).direct_to_source.mean()
        for category in categories: table[f"rate_{category}"] = table[category] / table.reply_count
        event_summary = table.reset_index()
        macro = event_summary.groupby("quiet_oracle_top50")[[f"rate_{x}" for x in categories] + ["direct_to_source_rate", "reply_count"]].mean().reset_index()
        print("[4/5] Rendering event-balanced stance comparison and sampling examples...", flush=True)
        plot = macro.set_index("quiet_oracle_top50")[[f"rate_{x}" for x in categories]].T
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        x = np.arange(len(categories)); width = .36
        axes[0].bar(x - width / 2, plot.get(False, pd.Series(0, index=plot.index)), width, label="other quiet")
        axes[0].bar(x + width / 2, plot.get(True, pd.Series(0, index=plot.index)), width, label="quiet Oracle Top-50")
        axes[0].set_xticks(x, ["support", "contradict /\nfact-check", "question /\nuncertainty", "other"]); axes[0].set_ylabel("Mean event-level fraction of observed replies"); axes[0].set_title("Early reply stance: observed 0–30 min"); axes[0].legend()
        rate = macro.set_index("quiet_oracle_top50").direct_to_source_rate
        axes[1].bar(["other quiet", "quiet Oracle Top-50"], [rate.get(False, 0.0), rate.get(True, 0.0)])
        axes[1].set_ylim(0, 1); axes[1].set_ylabel("Mean event-level rate"); axes[1].set_title("Observed replies directly addressing source")
        fig.suptitle("Exploratory lexical stance audit — not a model feature")
        figure = output / "quiet_early_reply_stance.png"; fig.savefig(figure, dpi=180); plt.close(fig)
        examples = replies.sort_values(["quiet_oracle_top50", "event_id", "offset_sec"], ascending=[False, True, True], kind="stable").groupby(["quiet_oracle_top50", "stance"], group_keys=False).head(5)
        print("[5/5] Writing audit tables and reproducibility record...", flush=True)
        event_summary.to_csv(output / "per_event_stance_rates.csv", index=False); macro.to_csv(output / "event_balanced_stance_summary.csv", index=False); examples[["thread_id", "event_id", "quiet_oracle_top50", "offset_sec", "direct_to_source", "stance", "source_text", "text"]].to_csv(output / "stance_examples.csv", index=False)
        info.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "quiet_threads": len(labels), "quiet_oracle_top50": int(labels.quiet_oracle_top50.sum()), "observed_reply_rows": len(replies), "output_files": ["quiet_early_reply_stance.png", "per_event_stance_rates.csv", "event_balanced_stance_summary.csv", "stance_examples.csv"]}); record(output, info)
        print(f"SUCCESS: quiet early stance audit saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        info.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}); record(output, info); print(f"FAILURE: quiet stance audit preserved at: {output.resolve()}", flush=True); raise


if __name__ == "__main__":
    main()
