"""Read-only before/after text diagnostic for quiet future high-impact threads.

Post-cutoff text is displayed only to explain outcome differences.  It is never
used as an input, normalizer, selector, or evaluation score for an early model.
"""
from __future__ import annotations

from collections import Counter
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
TOKEN = re.compile(r"[a-z][a-z']{2,}")
STOPWORDS = frozenset("the and for that with this from you your are was have has had not but they their about into after before will just what when where who why how all can could would should our out now over more than then them its it's rt via http https co amp".split())


def write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def tokens(text: str) -> list[str]:
    return [word for word in TOKEN.findall(str(text).lower()) if word not in STOPWORDS]


def ranked_terms(group_tokens: Counter, reference_tokens: Counter, limit: int = 12) -> pd.DataFrame:
    vocabulary = set(group_tokens) | set(reference_tokens)
    group_total, reference_total = sum(group_tokens.values()), sum(reference_tokens.values())
    rows = []
    for word in vocabulary:
        if group_tokens[word] + reference_tokens[word] < 4:
            continue
        score = np.log((group_tokens[word] + 0.5) / (group_total + 0.5 * len(vocabulary))) - np.log((reference_tokens[word] + 0.5) / (reference_total + 0.5 * len(vocabulary)))
        rows.append({"term": word, "log_rate_ratio_high_quiet_vs_other_quiet": score, "high_quiet_count": group_tokens[word], "other_quiet_count": reference_tokens[word]})
    return pd.DataFrame(rows).sort_values("log_rate_ratio_high_quiet_vs_other_quiet", ascending=False).head(limit)


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_quiet_text_before_after_diagnostic")
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "purpose": "read-only explanatory content diagnostic", "scores": str(SCORES), "reply_table": str(REPLIES), "cutoff_seconds": CUTOFF_SECONDS, "oracle_definition": "event-global future preventable-impact Top-50; outcome-only diagnostic label", "research_safety": "Post-cutoff text is plotted only after labels are formed. It is not used by any early-detection model, score, feature, normalization, or selection."}
    write_record(output, record)
    try:
        print("Starting quiet text before/after diagnostic (read-only).", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/5] Loading saved eligible-event quiet labels...", flush=True)
        scores = pd.read_csv(SCORES, dtype={"thread_id": str, "event_id": str})
        labels = []
        for event_id, event in scores.groupby("event_id", sort=True):
            oracle = set(event.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable").head(TOP_K).thread_id)
            labels.extend({"thread_id": thread_id, "event_id": event_id, "quiet": bool(quiet), "quiet_oracle_top50": bool(quiet and thread_id in oracle)} for thread_id, quiet in zip(event.thread_id, event.quiet))
        labels = pd.DataFrame(labels)
        quiet_labels = labels.loc[labels.quiet].copy()
        print(f"  quiet threads={len(quiet_labels)}, quiet Oracle Top-{TOP_K}={int(quiet_labels.quiet_oracle_top50.sum())}", flush=True)
        print("[2/5] Reading text rows and separating the strict-30 boundary...", flush=True)
        replies = pd.read_csv(REPLIES, dtype={"thread_id": str, "event_id": str}, usecols=["thread_id", "event_id", "offset_sec", "text", "is_text_available"])
        replies = replies.loc[replies.is_text_available.eq(1) & replies.text.notna() & replies.offset_sec.notna()].copy()
        replies["phase"] = np.where(replies.offset_sec <= CUTOFF_SECONDS, "observed_0_30m", "future_after_30m")
        text = replies.merge(quiet_labels[["thread_id", "event_id", "quiet_oracle_top50"]], on=["thread_id", "event_id"], how="inner", validate="many_to_one")
        if text.empty:
            raise ValueError("No text rows joined to saved quiet labels")
        print("[3/5] Computing lexical contrasts; post-cutoff content remains diagnostic-only...", flush=True)
        term_frames, summary = [], []
        for phase, phase_rows in text.groupby("phase", sort=True):
            high = Counter(token for value in phase_rows.loc[phase_rows.quiet_oracle_top50, "text"] for token in tokens(value))
            other = Counter(token for value in phase_rows.loc[~phase_rows.quiet_oracle_top50, "text"] for token in tokens(value))
            terms = ranked_terms(high, other); terms["phase"] = phase; term_frames.append(terms)
            counts = phase_rows.groupby("quiet_oracle_top50").agg(text_rows=("text", "size"), threads=("thread_id", "nunique")).reset_index()
            counts["phase"] = phase; summary.append(counts)
        terms = pd.concat(term_frames, ignore_index=True); summary = pd.concat(summary, ignore_index=True)
        print("[4/5] Rendering before/after lexical contrast chart...", flush=True)
        phases = ["observed_0_30m", "future_after_30m"]
        fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
        for axis, phase in zip(axes, phases):
            view = terms.loc[terms.phase.eq(phase)].sort_values("log_rate_ratio_high_quiet_vs_other_quiet")
            axis.barh(view.term, view.log_rate_ratio_high_quiet_vs_other_quiet, color="#4472c4")
            axis.axvline(0.0, color="black", linewidth=0.8)
            axis.set_title("Observed source/replies: 0–30 min" if phase == "observed_0_30m" else "Future replies: after 30 min (diagnostic only)")
            axis.set_xlabel("Smoothed log term-rate ratio: quiet Oracle Top-50 vs other quiet")
        fig.suptitle("Terms more common in quiet threads that later enter event Oracle Top-50")
        figure = output / "quiet_oracle_text_before_after.png"
        fig.savefig(figure, dpi=180); plt.close(fig)
        print("[5/5] Writing tables and reproducibility record...", flush=True)
        terms.to_csv(output / "lexical_contrast_terms.csv", index=False); summary.to_csv(output / "text_row_summary.csv", index=False)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "quiet_threads": len(quiet_labels), "quiet_oracle_top50": int(quiet_labels.quiet_oracle_top50.sum()), "joined_text_rows": len(text), "output_files": ["quiet_oracle_text_before_after.png", "lexical_contrast_terms.csv", "text_row_summary.csv"]})
        write_record(output, record)
        print(f"SUCCESS: quiet text before/after diagnostic saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}); write_record(output, record)
        print(f"FAILURE: quiet text diagnostic preserved at: {output.resolve()}", flush=True); raise


if __name__ == "__main__":
    main()
