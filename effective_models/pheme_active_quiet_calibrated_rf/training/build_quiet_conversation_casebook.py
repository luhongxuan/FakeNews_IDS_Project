"""Create a read-only, matched strict-30 quiet-thread conversation casebook."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCORES = HERE.parent / "experiments" / "20260902_132700_413954_quiet_tier_hybrid_policy_full" / "outer_scores.csv"
REPLIES = ROOT / "pheme_reply_level.csv"
OUT_ROOT = HERE.parent / "experiments"
CUTOFF_SECONDS = 1800.0
TOP_K = 50
MAX_REPLIES_SHOWN = 10


def write_record(output: Path, payload: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clean(value: object) -> str:
    return " ".join(str(value).replace("\n", " ").split())


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_quiet_conversation_casebook")
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "purpose": "read-only manual review packet for early conversation tension", "cutoff_seconds": CUTOFF_SECONDS, "pairing": "within event, quiet Oracle Top-50 case paired to quiet non-Oracle control with nearest observed reply count", "research_safety": "Only source and replies at or before 30 minutes are displayed. Future preventable impact labels cases after extraction and is never used as a feature or model input."}
    write_record(output, record)
    try:
        print("Starting quiet conversation casebook (read-only).", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/5] Loading saved quiet outcomes and forming event-local Oracle labels...", flush=True)
        scores = pd.read_csv(SCORES, dtype={"thread_id": str, "event_id": str})
        labels = []
        for event_id, event in scores.groupby("event_id", sort=True):
            oracle = set(event.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable").head(TOP_K).thread_id)
            labels.extend({"thread_id": thread_id, "event_id": event_id, "preventable_impact": impact, "quiet_oracle_top50": bool(thread_id in oracle)} for thread_id, impact in zip(event.loc[event.quiet, "thread_id"], event.loc[event.quiet, "preventable_impact"]))
        labels = pd.DataFrame(labels)
        print("[2/5] Loading only <=30-minute source/reply text and early reply counts...", flush=True)
        raw = pd.read_csv(REPLIES, dtype={"thread_id": str, "tweet_id": str, "parent_id": str, "event_id": str}, usecols=["thread_id", "tweet_id", "parent_id", "is_source", "is_text_available", "offset_sec", "depth", "text", "event_id"])
        observed = raw.loc[raw.offset_sec.notna() & raw.offset_sec.le(CUTOFF_SECONDS)].copy()
        sources = observed.loc[observed.is_source.eq(1), ["thread_id", "event_id", "tweet_id", "text"]].rename(columns={"tweet_id": "source_tweet_id", "text": "source_text"})
        replies = observed.loc[observed.is_source.ne(1) & observed.is_text_available.eq(1) & observed.text.notna()].copy()
        reply_counts = replies.groupby(["thread_id", "event_id"]).size().rename("observed_text_replies").reset_index()
        pool = labels.merge(reply_counts, on=["thread_id", "event_id"], how="left").merge(sources, on=["thread_id", "event_id"], how="left", validate="one_to_one")
        pool["observed_text_replies"] = pool.observed_text_replies.fillna(0).astype(int)
        pool = pool.loc[pool.source_text.notna()].copy()
        print("[3/5] Selecting one low-activity high-impact case and matched control per event...", flush=True)
        pairs = []
        for event_id, event in pool.groupby("event_id", sort=True):
            cases = event.loc[event.quiet_oracle_top50].sort_values(["observed_text_replies", "preventable_impact", "thread_id"], ascending=[True, False, True])
            controls = event.loc[~event.quiet_oracle_top50].copy()
            if cases.empty or controls.empty: continue
            case = cases.iloc[0]
            control = controls.assign(distance=(controls.observed_text_replies - case.observed_text_replies).abs()).sort_values(["distance", "preventable_impact", "thread_id"], ascending=[True, False, True]).iloc[0]
            pairs.extend([{**case.to_dict(), "role": "quiet_oracle_case", "paired_thread_id": control.thread_id}, {**control.to_dict(), "role": "matched_quiet_control", "paired_thread_id": case.thread_id}])
        pairs = pd.DataFrame(pairs)
        print(f"  selected {len(pairs) // 2} event-matched case/control pairs", flush=True)
        print("[4/5] Rendering source and early reply trees into reviewable Markdown...", flush=True)
        lines = ["# Quiet-thread 0–30 minute conversation casebook", "", "This is a read-only qualitative diagnostic. Future impact only defines which cases to inspect; all displayed text is at or before 30 minutes.", "", "Review prompts: Does a reply support, challenge, seek evidence for, mock, or redirect the source claim? Does a reply-to-reply exchange begin? Is there an identifiable tension signal that generic embeddings could average away?", ""]
        shown_rows = []
        for event_id, event_pairs in pairs.groupby("event_id", sort=True):
            lines.extend([f"## {event_id}", ""])
            for _, item in event_pairs.sort_values("role", ascending=False).iterrows():
                lines.extend([f"### {item.role}", f"- thread: `{item.thread_id}`", f"- paired thread: `{item.paired_thread_id}`", f"- future preventable impact: {item.preventable_impact:g}", f"- observed text replies: {item.observed_text_replies}", f"- source: {clean(item.source_text)}", "", "Observed replies:"])
                thread_replies = replies.loc[(replies.thread_id.eq(item.thread_id)) & (replies.event_id.eq(event_id))].sort_values(["offset_sec", "tweet_id"], kind="stable").head(MAX_REPLIES_SHOWN)
                if thread_replies.empty:
                    lines.append("- *(no text-bearing reply inside 30 minutes)*")
                for _, reply in thread_replies.iterrows():
                    direct = "direct-to-source" if reply.parent_id == item.source_tweet_id else f"reply-to {reply.parent_id}"
                    text = clean(reply.text)
                    lines.append(f"- t={reply.offset_sec:.0f}s, depth={reply.depth}, {direct}: {text}")
                    shown_rows.append({"event_id": event_id, "role": item.role, "thread_id": item.thread_id, "offset_sec": reply.offset_sec, "depth": reply.depth, "direct_to_source": direct == "direct-to-source", "text": text})
                lines.append("")
        (output / "quiet_conversation_casebook.md").write_text("\n".join(lines), encoding="utf-8")
        print("[5/5] Writing pair manifest and reproducibility record...", flush=True)
        pairs.to_csv(output / "case_control_pairs.csv", index=False); pd.DataFrame(shown_rows).to_csv(output / "shown_observed_replies.csv", index=False)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "quiet_threads": len(pool), "quiet_oracle_top50": int(pool.quiet_oracle_top50.sum()), "pair_count": len(pairs) // 2, "output_files": ["quiet_conversation_casebook.md", "case_control_pairs.csv", "shown_observed_replies.csv"]}); write_record(output, record)
        print(f"SUCCESS: quiet conversation casebook saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}); write_record(output, record); print(f"FAILURE: quiet conversation casebook preserved at: {output.resolve()}", flush=True); raise


if __name__ == "__main__":
    main()
