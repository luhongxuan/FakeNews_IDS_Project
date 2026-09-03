"""Convert real PHEME raw threads into social-frontend's dataset format.

Writes to social-frontend/public/datasets/<id>/thread.json, matching the
schema in social-frontend/public/datasets/README.md. This is read-only with
respect to data/raw/pheme (the protected raw dataset) -- it only reads from
there and writes to a new location under social-frontend/, per AGENTS.md
dataset-protection rules.

`is_rumour` is carried over unchanged from PHEME's own rumours/non-rumours
folder split (a thread being in the "rumours" folder means it was a
contested/circulating claim at the time -- not necessarily false; see
annotation.json's separate `true`/`misinformation` fields for eventual
veracity). This matches the frontend's own stated intent: it's a structural
marker for the operator's graph view, not a truth label shown to the end
user.

Thread selection: to keep each demo feed a readable size, threads are
capped per event, preferring the same threads already highlighted as
high-priority by the v5 text RF ranking (effective_models/pheme_v5_text_rf)
so the two frontends tell a connected story -- these are literally the
posts the early-detection dashboard already flags, now shown as they
actually looked in the live feed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PHEME_RAW = ROOT / "data" / "raw" / "pheme"
V5_OOF = (
    ROOT / "effective_models" / "pheme_v5_text_rf" / "reference_result"
    / "20260831_223827_909201_v5_best_text_oof_min_interventions" / "oof_thread_scores.csv"
)
OUT_DIR = ROOT / "social-frontend" / "public" / "datasets"
MANIFEST_PATH = OUT_DIR / "manifest.json"

TWITTER_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"
THREADS_PER_DATASET = 25

# All 9 PHEME events, matching the titles already used in
# backend/app/routers/events.py's EVENT_TITLE so both frontends agree.
# Labelled "PHEME 真實資料" rather than "保留集": these events were each
# used once as an outer-LOEO test fold in the v5 model's own nested
# cross-validation, so none of them is data the evaluation pipeline never
# touched -- "holdout" would overstate that. See conversation notes.
EVENT_TITLES = {
    "charliehebdo": "查理週刊槍擊案",
    "ebola-essien": "埃博拉疫情",
    "ferguson": "弗格森事件",
    "germanwings-crash": "德國之翼墜機事件",
    "gurlitt": "古利特失竊事件",
    "ottawashooting": "渥太華槍擊案",
    "prince-toronto": "多倫多王子事件",
    "putinmissing": "普丁失蹤事件",
    "sydneysiege": "雪梨人質事件",
}
DATASETS = [
    {"event_id": event_id, "id": event_id, "label": f"{title}（PHEME 真實資料）"}
    for event_id, title in EVENT_TITLES.items()
]


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, TWITTER_DATE_FORMAT)
    except ValueError:
        return None


def _load_tweets(thread_dir: Path) -> dict[str, dict]:
    tweets: dict[str, dict] = {}
    for folder_name in ("source-tweets", "reactions"):
        folder = thread_dir / folder_name
        if not folder.exists():
            continue
        for path in folder.glob("*.json"):
            if path.name.startswith("._"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            except (json.JSONDecodeError, OSError):
                continue
            tweet_id = data.get("id_str")
            if not tweet_id:
                continue
            tweets[tweet_id] = {
                "id": tweet_id,
                "text": data.get("text", ""),
                "user": (data.get("user") or {}).get("screen_name", "unknown"),
                "created_at": _parse_date(data.get("created_at")),
            }
    return tweets


def _thread_dir(event_id: str, thread_id: str) -> tuple[Path, bool] | None:
    event_dir = PHEME_RAW / f"{event_id}-all-rnr-threads"
    for label, is_rumour in (("rumours", True), ("non-rumours", False)):
        candidate = event_dir / label / thread_id
        if candidate.exists():
            return candidate, is_rumour
    return None


def build_thread(event_id: str, thread_id: str) -> dict | None:
    located = _thread_dir(event_id, thread_id)
    if located is None:
        return None
    thread_dir, is_rumour = located
    structure_path = thread_dir / "structure.json"
    if not structure_path.exists():
        return None
    tweets = _load_tweets(thread_dir)
    if thread_id not in tweets:
        return None
    structure = json.loads(structure_path.read_text(encoding="utf-8", errors="ignore"))

    source_tweet = tweets[thread_id]
    reactions: list[dict] = []

    def traverse(node_dict, parent_id: str | None) -> None:
        if not node_dict:
            return
        for tweet_id, children in node_dict.items():
            info = tweets.get(tweet_id)
            if info is not None and tweet_id != thread_id:
                parent_info = tweets.get(parent_id) if parent_id else None
                reactions.append({
                    "user": info["user"],
                    "content": info["text"],
                    "reply_to": parent_info["user"] if parent_info else source_tweet["user"],
                    "ts": (info["created_at"] or source_tweet["created_at"] or datetime.now(timezone.utc)).isoformat(),
                    "_sort_key": info["created_at"] or datetime.min.replace(tzinfo=timezone.utc),
                })
            traverse(children, tweet_id)

    traverse(structure, None)
    reactions.sort(key=lambda item: item["_sort_key"])
    for item in reactions:
        del item["_sort_key"]

    return {
        "thread_id": thread_id,
        "is_rumour": is_rumour,
        "source": {
            "user": source_tweet["user"],
            "content": source_tweet["text"],
            "ts": (source_tweet["created_at"] or datetime.now(timezone.utc)).isoformat(),
        },
        "reactions": reactions,
    }


def main() -> None:
    print("Starting PHEME -> social-frontend dataset conversion.", flush=True)
    print(f"Output directory: {OUT_DIR.resolve()}", flush=True)
    scores = pd.read_csv(V5_OOF, dtype={"thread_id": str, "event_id": str})

    # This script is the single source of truth for what belongs in the
    # manifest; rebuild it fresh each run rather than merging, so renaming
    # or dropping a dataset here doesn't leave stale entries behind.
    manifest = []

    for spec in DATASETS:
        event_id, dataset_id, label = spec["event_id"], spec["id"], spec["label"]
        print(f"[{dataset_id}] selecting top {THREADS_PER_DATASET} threads by v5 model score for event={event_id}...", flush=True)
        candidates = (
            scores.loc[scores.event_id == event_id]
            .sort_values(["prediction", "thread_id"], ascending=[False, True])
            .head(THREADS_PER_DATASET)
        )
        threads = []
        for index, row in enumerate(candidates.itertuples(), 1):
            thread = build_thread(event_id, row.thread_id)
            if thread is not None:
                threads.append(thread)
            print(f"  {index}/{len(candidates)}: thread_id={row.thread_id} -> {'ok' if thread else 'MISSING raw files, skipped'}", flush=True)
        if not threads:
            print(f"[{dataset_id}] FAILURE: no threads could be built, skipping this dataset.", flush=True)
            continue

        out_dir = OUT_DIR / dataset_id
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {"id": dataset_id, "label": label, "threads": threads}
        (out_dir / "thread.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{dataset_id}] wrote {len(threads)} threads to {out_dir / 'thread.json'}", flush=True)

        manifest.append({"id": dataset_id, "label": label})

    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUCCESS: manifest updated at {MANIFEST_PATH.resolve()}", flush=True)


if __name__ == "__main__":
    main()
