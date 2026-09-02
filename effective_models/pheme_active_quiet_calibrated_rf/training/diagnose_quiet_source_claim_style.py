"""Read-only strict-30 source-claim-style diagnostic for quiet threads."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCORES = HERE.parent / "experiments" / "20260902_132700_413954_quiet_tier_hybrid_policy_full" / "outer_scores.csv"
REPLIES = ROOT / "pheme_reply_level.csv"
OUT_ROOT = HERE.parent / "experiments"
TOP_K = 50
PATTERNS = {
    "uncertainty_or_question": r"\?|\b(may|might|possible|possibly|reportedly|allegedly|rumou?r|could|perhaps|unconfirmed)\b",
    "correction_or_denial": r"\b(not|no|false|fake|wrong|debunk|deny|denied|isn't|is not|clarif(?:y|ication)|discredit)\b",
    "conflict_or_accusation": r"\b(attack|shoot|killed|dead|criminal|coup|hostage|gunman|terror|police|arrest|war)\b",
    "moral_or_harm": r"\b(heartbreaking|outrage|shame|justice|victim|innocent|racis(?:m|t)|brutal|violence)\b",
    "identity_or_polarization": r"\b(muslim|islamic|black|white|police|government|putin|immigrant|gay|woman|men)\b",
    "external_evidence_or_media": r"https?://|\b(photo|video|report|source|document|data|study|via)\b",
    "urgency_or_breaking": r"\b(breaking|urgent|now|just in|alert|developing)\b",
}


def save(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_quiet_source_claim_style_diagnostic")
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "purpose": "read-only source-only claim-style diagnostic", "cutoff_seconds": 1800, "patterns": PATTERNS, "research_safety": "Source text is observed at offset zero. Future impact defines post-hoc diagnostic groups only and is not used as a model input or selector."}
    save(output, record)
    try:
        print("Starting quiet source claim-style diagnostic (read-only).", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/4] Loading quiet Oracle diagnostic labels...", flush=True)
        scores = pd.read_csv(SCORES, dtype={"thread_id": str, "event_id": str})
        labels = []
        for event_id, event in scores.groupby("event_id", sort=True):
            oracle = set(event.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable").head(TOP_K).thread_id)
            labels.extend({"thread_id": thread_id, "event_id": event_id, "quiet_oracle_top50": bool(thread_id in oracle)} for thread_id in event.loc[event.quiet, "thread_id"])
        labels = pd.DataFrame(labels)
        print("[2/4] Joining source text only...", flush=True)
        source = pd.read_csv(REPLIES, dtype={"thread_id": str, "event_id": str}, usecols=["thread_id", "event_id", "is_source", "offset_sec", "text", "is_text_available"])
        source = source.loc[source.is_source.eq(1) & source.is_text_available.eq(1) & source.text.notna() & source.offset_sec.eq(0)].merge(labels, on=["thread_id", "event_id"], how="inner", validate="one_to_one")
        for name, pattern in PATTERNS.items(): source[name] = source.text.str.contains(pattern, flags=re.I, regex=True, na=False)
        print("[3/4] Computing event-balanced source-style contrasts...", flush=True)
        features = list(PATTERNS)
        per_event = source.groupby(["event_id", "quiet_oracle_top50"])[features].mean().reset_index()
        macro = per_event.groupby("quiet_oracle_top50")[features].mean().T.reset_index(names="feature").rename(columns={False: "other_quiet_rate", True: "quiet_oracle_rate"})
        macro["difference_oracle_minus_other"] = macro.quiet_oracle_rate - macro.other_quiet_rate
        macro = macro.sort_values("difference_oracle_minus_other")
        print("[4/4] Rendering contrast chart and writing evidence...", flush=True)
        fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
        axis.barh(macro.feature, macro.difference_oracle_minus_other)
        axis.axvline(0, color="black", linewidth=.8); axis.set_xlabel("Event-balanced rate difference: quiet Oracle Top-50 − other quiet"); axis.set_title("Source claim-style signals at t=0 (exploratory)")
        figure = output / "quiet_source_claim_style_contrast.png"; fig.savefig(figure, dpi=180); plt.close(fig)
        macro.to_csv(output / "event_balanced_source_style_contrast.csv", index=False); per_event.to_csv(output / "per_event_source_style_rates.csv", index=False)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "quiet_threads": len(source), "quiet_oracle_top50": int(source.quiet_oracle_top50.sum()), "output_files": ["quiet_source_claim_style_contrast.png", "event_balanced_source_style_contrast.csv", "per_event_source_style_rates.csv"]}); save(output, record)
        print(f"SUCCESS: quiet source claim-style diagnostic saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}); save(output, record); print(f"FAILURE: source claim-style diagnostic preserved at: {output.resolve()}", flush=True); raise


if __name__ == "__main__":
    main()
