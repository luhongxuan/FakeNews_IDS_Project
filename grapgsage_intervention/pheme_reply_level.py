from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 0. 常數
# ------------------------------------------------------------------

TWITTER_TIME_FMT = "%a %b %d %H:%M:%S %z %Y"
LABEL_COL = "is_rumour"
EVENT_COL = "event_id"


# ------------------------------------------------------------------
# 1. 工具函式（與 pheme_feature_extraction.py 邏輯一致）
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# 2. 從 structure.json 解析完整的 parent-child 關係 + depth
# ------------------------------------------------------------------

def parse_structure_with_depth(structure: dict) -> tuple[dict, dict]:
    """
    遞迴解析 structure.json，回傳：
      - parent_map: {tweet_id: parent_id}（root 的 parent = None）
      - depth_map:  {tweet_id: depth}（root depth = 0）
    同時處理 PHEME 的 [] 和 {} 兩種葉節點格式。
    """
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
            # children 是 [] 的情況：葉節點，不需要再往下

    dfs(structure, None, 0)
    return parent_map, depth_map


# ------------------------------------------------------------------
# 3. Neighbor-Coverage 計算
# ------------------------------------------------------------------

def compute_coverage_scores(parent_map: dict) -> dict[str, float]:
    """
    對傳播樹裡的每個節點計算 neighbor-coverage score：
    
        coverage_score(v) = 移除 v 之後無法被觸達的後代節點數量
                            ÷ (總節點數 - 1)
    
    「無法被觸達」的定義：從 root 出發，經過 v 才能到達的節點。
    也就是 v 的所有後代（subtree rooted at v，不含 v 本身）。
    
    實作方式：
    1. 建立 children_map（從 parent_map 反轉）
    2. 對每個節點做 DFS 數後代數量
    3. 標準化
    
    時間複雜度：O(n) per thread（每個節點只被訪問一次）
    """
    total_nodes = len(parent_map)
    if total_nodes <= 1:
        # 只有 root，沒有後代可以被阻斷
        return {tid: 0.0 for tid in parent_map}

    # 建立 children_map
    children_map: dict[str, list[str]] = defaultdict(list)
    root_id = None
    for tid, pid in parent_map.items():
        if pid is None:
            root_id = tid
        else:
            children_map[pid].append(tid)

    # 對每個節點計算後代數量（subtree size - 1，不含自己）
    subtree_size: dict[str, int] = {}

    def count_subtree(node_id: str) -> int:
        size = 1  # 算自己
        for child in children_map.get(node_id, []):
            size += count_subtree(child)
        subtree_size[node_id] = size
        return size

    if root_id:
        count_subtree(root_id)

    # coverage_score = 後代數量 / (總節點數 - 1)
    denom = total_nodes - 1
    scores = {}
    for tid in parent_map:
        descendants = subtree_size.get(tid, 1) - 1  # 不含自己
        scores[tid] = descendants / denom if denom > 0 else 0.0

    return scores


# ------------------------------------------------------------------
# 4. 單一 thread 解析
# ------------------------------------------------------------------

def parse_thread(
    thread_path: Path,
    is_rumour: int,
    event_id: str,
) -> tuple[list[dict], Optional[str]]:
    """
    解析單一 thread，回傳 (rows, skip_reason)。
    rows 是這個 thread 裡所有節點的 reply-level 資料列。
    """
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

    # 讀取推文 JSON
    tweets = {}
    tweets.update(_load_tweets_in_folder(thread_path / "source-tweets"))
    tweets.update(_load_tweets_in_folder(thread_path / "reactions"))

    # source tweet ID = root（depth = 0 的節點）
    source_id = next((tid for tid, d in depth_map.items() if d == 0), None)
    if source_id is None:
        return [], "no_root_found"

    source_tweet = tweets.get(source_id)
    source_time = parse_twitter_time(
        source_tweet.get("created_at") if source_tweet else None
    )

    # 計算 neighbor-coverage scores
    coverage_scores = compute_coverage_scores(parent_map)

    rows = []
    for tweet_id, depth in depth_map.items():
        tweet = tweets.get(tweet_id)
        pid = parent_map.get(tweet_id)
        parent_tweet = tweets.get(pid) if pid else None

        is_source = int(tweet_id == source_id)
        is_text_available = int(tweet is not None)

        # 時間計算
        if tweet and source_time:
            t = parse_twitter_time(tweet.get("created_at"))
            offset_sec = (
                (t - source_time).total_seconds()
                if t and t >= source_time
                else np.nan
            )
        else:
            offset_sec = 0.0 if is_source else np.nan

        if pid and parent_tweet and tweet and source_time:
            t_self = parse_twitter_time(tweet.get("created_at"))
            t_parent = parse_twitter_time(parent_tweet.get("created_at"))
            reply_latency_sec = (
                (t_self - t_parent).total_seconds()
                if t_self and t_parent and t_self >= t_parent
                else np.nan
            )
        else:
            reply_latency_sec = np.nan  # source tweet 沒有 parent

        text = tweet.get("text", "") if tweet else None

        rows.append({
            "thread_id": source_id,
            "tweet_id": tweet_id,
            "parent_id": pid,
            "is_source": is_source,
            "is_text_available": is_text_available,
            "depth": depth,
            "offset_sec": offset_sec,
            "reply_latency_sec": reply_latency_sec,
            "text": text,
            EVENT_COL: event_id,
            LABEL_COL: is_rumour,
            "coverage_score": coverage_scores.get(tweet_id, 0.0),
        })

    return rows, None


# ------------------------------------------------------------------
# 5. 走訪整個 PHEME 資料夾
# ------------------------------------------------------------------

def build_reply_level_dataframe(
    pheme_path: str,
    cache_csv: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    走訪 PHEME 資料夾，建立 reply-level DataFrame。
    
    回傳欄位：
        thread_id, tweet_id, parent_id, is_source, is_text_available,
        depth, offset_sec, reply_latency_sec, text,
        event_id, is_rumour, coverage_score
    
    一則推文一列，包含：
    - 有 JSON 的推文：所有欄位都有值
    - 沒有 JSON 的推文（已刪除）：is_text_available=0，
      text/offset_sec/reply_latency_sec 為 NaN，
      但仍保留在結構裡，coverage_score 正常計算
    """
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
            for thread_dir in sorted(
                d for d in label_path.iterdir() if d.is_dir()
            ):
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
                reasons = "、".join(
                    f"{r}={c}" for r, c in event_skip.most_common()
                )
                line += f"  跳過：{reasons}"
            print(line)

    df = pd.DataFrame(all_rows)

    if verbose:
        print(f"\n總計：{len(df)} 列（推文節點）")
        print(f"有 JSON 的節點：{df['is_text_available'].sum()}")
        print(f"無 JSON 的節點（已刪除推文）：{(~df['is_text_available'].astype(bool)).sum()}")
        print(f"\ncoverage_score 分布：")
        print(df["coverage_score"].describe().round(4))

    if cache_csv:
        df.to_csv(cache_csv, index=False)
        if verbose:
            print(f"\n已快取至 {cache_csv}")

    return df


if __name__ == "__main__":
    pheme_path = os.getenv("PHEME_PATH", "data/raw/pheme")
    df = build_reply_level_dataframe(
        pheme_path, cache_csv="pheme_reply_level.csv", verbose=True
    )
    print("\n前 5 列：")
    print(df.head())