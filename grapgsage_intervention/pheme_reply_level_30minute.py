from __future__ import annotations

import json
import os
os.chdir(r"C:\FakeNews_IDS_Project")
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

TWITTER_TIME_FMT = "%a %b %d %H:%M:%S %z %Y"
LABEL_COL = "is_rumour"
EVENT_COL = "event_id"

# 支援的時間窗口（秒）
WINDOW_30MIN  = 1800
WINDOW_60MIN  = 3600
WINDOW_ALL    = float("inf")  # 完整傳播樹


def parse_twitter_time(time_str: Optional[str]) -> Optional[datetime]:
    if not time_str:
        return None
    try:
        return datetime.strptime(time_str, TWITTER_TIME_FMT)
    except (ValueError, TypeError):
        return None


def _load_tweets_in_folder(folder_path: Path) -> dict:
    tweets = {}
    if not folder_path.exists():
        return tweets
    for fname in os.listdir(folder_path):
        if fname.startswith("._") or not fname.endswith(".json"):
            continue
        try:
            with open(folder_path / fname, encoding="utf-8", errors="ignore") as f:
                tweet = json.load(f)
            tweets[str(tweet["id"])] = tweet
        except (json.JSONDecodeError, OSError, KeyError):
            continue
    return tweets


def parse_structure_with_depth(structure: dict) -> tuple[dict, dict]:
    parent_map: dict[str, Optional[str]] = {}
    depth_map: dict[str, int] = {}

    def dfs(node_dict: dict, parent_id: Optional[str], depth: int) -> None:
        if not isinstance(node_dict, dict):
            return
        for tweet_id, children in node_dict.items():
            parent_map[tweet_id] = parent_id
            depth_map[tweet_id] = depth
            if isinstance(children, dict):
                dfs(children, tweet_id, depth + 1)

    dfs(structure, None, 0)
    return parent_map, depth_map


def compute_coverage_scores(parent_map: dict) -> dict[str, float]:
    """完整傳播樹的 coverage_score（原始版本，不考慮時間窗口）"""
    total_nodes = len(parent_map)
    if total_nodes <= 1:
        return {tid: 0.0 for tid in parent_map}

    children_map: dict[str, list[str]] = defaultdict(list)
    root_id = None
    for tid, pid in parent_map.items():
        if pid is None:
            root_id = tid
        else:
            children_map[pid].append(tid)

    subtree_size: dict[str, int] = {}

    def count_subtree(node_id: str) -> int:
        size = 1
        for child in children_map.get(node_id, []):
            size += count_subtree(child)
        subtree_size[node_id] = size
        return size

    if root_id:
        count_subtree(root_id)

    denom = total_nodes - 1
    scores = {}
    for tid in parent_map:
        descendants = subtree_size.get(tid, 1) - 1
        scores[tid] = descendants / denom if denom > 0 else 0.0

    return scores


def compute_coverage_scores_windowed(
    parent_map: dict,
    offset_map: dict,
    window_sec: float,
) -> dict[str, float]:
    """
    時間窗口版本的 coverage_score：

    coverage_score_W(v) =
        「在 window_sec 時介入節點 v，能阻斷的後代數量」
        ÷ (總節點數 - 1)

    「能阻斷的後代」= v 的後代中 offset_sec > window_sec 的節點。
    也就是窗口截止後才出現的後代——這些是還沒發生的傳播，介入才有意義。

    語意：
      - 輸入（GNN）：只給 offset_sec <= window_sec 的節點和邊（早期資訊）
      - label：v 在窗口之後會影響多少節點（未來資訊，是合法的訓練目標）
      - 這樣模型學的是：「從早期結構預測未來的影響力」，不是作弊從結構數後代
    """
    total_nodes = len(parent_map)
    if total_nodes <= 1:
        return {tid: 0.0 for tid in parent_map}

    children_map: dict[str, list[str]] = defaultdict(list)
    for tid, pid in parent_map.items():
        if pid is not None:
            children_map[pid].append(tid)

    def count_future_descendants(node_id: str) -> int:
        """v 的後代中，offset_sec > window_sec 的數量"""
        count = 0
        for child in children_map.get(node_id, []):
            child_offset = offset_map.get(child, 0.0)
            if np.isnan(child_offset):
                child_offset = 0.0
            if child_offset > window_sec:
                count += 1
            count += count_future_descendants(child)
        return count

    denom = total_nodes - 1
    scores = {}
    for tid in parent_map:
        future_desc = count_future_descendants(tid)
        scores[tid] = future_desc / denom if denom > 0 else 0.0

    return scores


def parse_thread(
    thread_path: Path,
    is_rumour: int,
    event_id: str,
) -> tuple[list[dict], Optional[str]]:
    structure_path = thread_path / "structure.json"
    if not structure_path.exists():
        return [], "missing_structure_json"
    try:
        with open(structure_path, encoding="utf-8") as f:
            structure = json.load(f)
    except (json.JSONDecodeError, OSError):
        return [], "unreadable_structure_json"

    if not isinstance(structure, dict) or not structure:
        return [], "empty_structure"

    parent_map, depth_map = parse_structure_with_depth(structure)
    if not parent_map:
        return [], "empty_structure"

    tweets = {}
    tweets.update(_load_tweets_in_folder(thread_path / "source-tweets"))
    tweets.update(_load_tweets_in_folder(thread_path / "reactions"))

    source_id = next((tid for tid, d in depth_map.items() if d == 0), None)
    if source_id is None:
        return [], "no_root_found"

    source_tweet = tweets.get(source_id)
    source_time = parse_twitter_time(
        source_tweet.get("created_at") if source_tweet else None
    )

    # 先算每個節點的 offset_sec（後面算 windowed coverage 需要）
    offset_map: dict[str, float] = {}
    for tweet_id in parent_map:
        if tweet_id == source_id:
            offset_map[tweet_id] = 0.0
        else:
            tweet = tweets.get(tweet_id)
            if tweet and source_time:
                t = parse_twitter_time(tweet.get("created_at"))
                if t and t >= source_time:
                    offset_map[tweet_id] = (t - source_time).total_seconds()
                else:
                    offset_map[tweet_id] = np.nan
            else:
                offset_map[tweet_id] = np.nan

    # 計算三種 coverage_score
    coverage_scores        = compute_coverage_scores(parent_map)
    coverage_scores_30min  = compute_coverage_scores_windowed(parent_map, offset_map, WINDOW_30MIN)
    coverage_scores_60min  = compute_coverage_scores_windowed(parent_map, offset_map, WINDOW_60MIN)

    rows = []
    for tweet_id, depth in depth_map.items():
        tweet = tweets.get(tweet_id)
        pid = parent_map.get(tweet_id)
        parent_tweet = tweets.get(pid) if pid else None

        is_source_flag = int(tweet_id == source_id)
        is_text_available = int(tweet is not None)

        offset_sec = offset_map.get(tweet_id, 0.0 if tweet_id == source_id else np.nan)

        if pid and parent_tweet and tweet and source_time:
            t_self   = parse_twitter_time(tweet.get("created_at"))
            t_parent = parse_twitter_time(parent_tweet.get("created_at"))
            reply_latency_sec = (
                (t_self - t_parent).total_seconds()
                if t_self and t_parent and t_self >= t_parent
                else np.nan
            )
        else:
            reply_latency_sec = np.nan

        text = tweet.get("text", "") if tweet else None

        rows.append({
            "thread_id":            source_id,
            "tweet_id":             tweet_id,
            "parent_id":            pid,
            "is_source":            is_source_flag,
            "is_text_available":    is_text_available,
            "depth":                depth,
            "offset_sec":           offset_sec,           # 距 source 的秒數（原始值，建圖截斷用）
            "reply_latency_sec":    reply_latency_sec,
            "text":                 text,
            EVENT_COL:              event_id,
            LABEL_COL:              is_rumour,
            "coverage_score":       coverage_scores.get(tweet_id, 0.0),        # 完整版
            "coverage_score_30min": coverage_scores_30min.get(tweet_id, 0.0),  # 30分鐘窗口
            "coverage_score_60min": coverage_scores_60min.get(tweet_id, 0.0),  # 60分鐘窗口
        })

    return rows, None


def build_reply_level_dataframe(
    pheme_path: str,
    cache_csv: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    root = Path(pheme_path)
    if not root.exists():
        raise FileNotFoundError(f"找不到 PHEME 路徑：{root}")

    all_rows = []
    skip_counter: Counter = Counter()

    event_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    for event_dir in event_dirs:
        event_id = event_dir.name.replace("-all-rnr-threads", "")
        event_thread_count = 0
        event_node_count = 0
        event_skip: Counter = Counter()

        for label, is_rumour in [("rumours", 1), ("non-rumours", 0)]:
            label_path = event_dir / label
            if not label_path.exists():
                continue
            for thread_dir in sorted(d for d in label_path.iterdir() if d.is_dir()):
                rows, skip_reason = parse_thread(thread_dir, is_rumour, event_id)
                if rows:
                    all_rows.extend(rows)
                    event_thread_count += 1
                    event_node_count += len(rows)
                else:
                    skip_counter[skip_reason] += 1
                    event_skip[skip_reason] += 1

        if verbose:
            total_scanned = event_thread_count + sum(event_skip.values())
            line = (
                f"[{event_id}] thread={event_thread_count}/{total_scanned}"
                f"  nodes={event_node_count}"
            )
            if event_skip:
                reasons = "、".join(f"{r}={c}" for r, c in event_skip.most_common())
                line += f"  跳過：{reasons}"
            print(line)

    df = pd.DataFrame(all_rows)

    if verbose:
        print(f"\n總計：{len(df)} 列（推文節點）")
        print(f"有 JSON 的節點：{df['is_text_available'].sum()}")
        print(f"無 JSON 的節點（已刪除推文）：{(~df['is_text_available'].astype(bool)).sum()}")
        print(f"\ncoverage_score 分布（完整版）：")
        print(df["coverage_score"].describe().round(4))
        print(f"\ncoverage_score_30min 分布：")
        print(df["coverage_score_30min"].describe().round(4))

    if cache_csv:
        df.to_csv(cache_csv, index=False)
        if verbose:
            print(f"\n已快取至 {cache_csv}")

    return df


if __name__ == "__main__":
    pheme_path = os.getenv("PHEME_PATH", "data/raw/pheme")
    df = build_reply_level_dataframe(
        pheme_path, cache_csv="pheme_reply_level_v2.csv", verbose=True
    )
    print("\n前 5 列：")
    print(df[["thread_id", "tweet_id", "offset_sec",
              "coverage_score", "coverage_score_30min", "coverage_score_60min"]].head())