"""
PHEME 節點特徵建構（重寫版——原始 pheme_node_features.py 已遺失）
================================================
用途：把 pheme_reply_level_v3.csv（結構/時間/coverage_score/future_growth）
補上使用者特徵、VADER 情感、RoBERTa embedding，產出
pheme_graph_dataset.py / pheme_graph_dataset_30minute.py 期待的
node_features CSV（13 維數值特徵 + 768 維 embedding = 781 維）。

============================================================
重要：這是照 pheme_graph_dataset.py 裡 NUMERIC_FEAT_COLS 的欄位名稱、
配合你們過去對話裡提到的設計（RoBERTa mean-pooling、帳號年齡算事件
當下而非現在）重建的，不是原始檔案本身。幾個地方是我做的合理假設，
你拿到結果後最好抽樣核對：

1. account_age_days_log：用「這則推文自己的發文時間」減掉「帳號的
   建立時間」，不是用現在（2026）的時間去算——這點延續你們處理
   Twitter15/16 使用者特徵時同樣的原則（帳號年齡是歷史事實，用
   事件當下的時間點算才準確）。PHEME 的原始 JSON 裡兩個時間都有，
   不需要额外爬蟲，直接從 tweet 的 user 物件裡的 created_at 取。
2. offset_sec_log / reply_latency_sec_log：對 NaN（例如來源推文沒有
   latency、已刪除推文沒有 offset）先 fillna(0) 再做 log1p，等於
   把「未知/不適用」跟「間隔 0 秒」視為同一種情況處理，這是為了讓
   特徵矩陣不出現 NaN，不是說這兩者語意上真的一樣。
3. RoBERTa embedding 用 mean-pooling（跟 twitter15_node_features.py
   那支用的方式一致），沒有把 fine-tuning 這件事考慮進去——你們
   已經確認過 frozen embedding 效果比 LOEO fine-tuned 版本穩定，
   這裡沿用這個結論，不是重新引入 fine-tuning。
============================================================

用法：
    python pheme_node_features.py \
        --reply_level_csv pheme_reply_level_v3.csv \
        --pheme_path data/raw/pheme \
        --output_csv pheme_node_features_roberta.csv

需要先安裝：
    pip install vaderSentiment transformers torch
"""

import os
import re
import json
import math
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import torch
from transformers import AutoTokenizer, AutoModel


ROBERTA_MODEL_NAME = "cardiffnlp/twitter-roberta-base"
EMBED_DIM = 768
TWITTER_TIME_FMT = "%a %b %d %H:%M:%S %z %Y"

NUMERIC_FEAT_COLS = [
    "depth", "offset_sec_log", "reply_latency_sec_log", "is_source",
    "vader_compound", "vader_pos", "vader_neg", "vader_neu",
    "followers_log", "friends_log", "statuses_log",
    "verified", "account_age_days_log",
]


# ---------------------------------------------------------------------------
# Step A: 從 PHEME 原始 JSON 重新撈使用者特徵（reply_level 沒有留這塊）
# ---------------------------------------------------------------------------

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


def extract_user_features(tweet: dict) -> dict:
    """從單則推文的完整 JSON 裡取出 user 物件的特徵。"""
    user = tweet.get("user") or {}
    tweet_time = parse_twitter_time(tweet.get("created_at"))
    account_created = parse_twitter_time(user.get("created_at"))

    account_age_days = None
    if tweet_time and account_created:
        account_age_days = max((tweet_time - account_created).days, 0)

    return {
        "followers_count": user.get("followers_count"),
        "friends_count":   user.get("friends_count"),
        "statuses_count":  user.get("statuses_count"),
        "verified":        1 if user.get("verified") else 0,
        "account_age_days": account_age_days,
    }


def build_user_feature_table(pheme_path: str, verbose: bool = True) -> dict:
    """
    走訪整個 PHEME 資料夾一次，回傳 tweet_id -> user 特徵 dict。
    跟 pheme_reply_level_30minute.py 用同樣的資料夾邏輯，但這裡只要
    user 物件，不重新算 depth/offset 這些（那些 reply_level 已經有了）。
    """
    root = Path(pheme_path)
    if not root.exists():
        raise FileNotFoundError(f"找不到 PHEME 路徑：{root}")

    user_feats = {}
    event_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    for event_i, event_dir in enumerate(event_dirs, 1):
        if verbose:
            print(f"  [{event_i}/{len(event_dirs)}] 讀取使用者特徵：{event_dir.name}")
        for label in ("rumours", "non-rumours"):
            label_path = event_dir / label
            if not label_path.exists():
                continue
            for thread_dir in label_path.iterdir():
                if not thread_dir.is_dir():
                    continue
                tweets = {}
                tweets.update(_load_tweets_in_folder(thread_dir / "source-tweets"))
                tweets.update(_load_tweets_in_folder(thread_dir / "reactions"))
                for tid, tweet in tweets.items():
                    user_feats[tid] = extract_user_features(tweet)
    return user_feats


# ---------------------------------------------------------------------------
# Step B: RoBERTa embedding（跟 twitter15_node_features.py 同一套邏輯）
# ---------------------------------------------------------------------------

class RobertaEmbedder:
    def __init__(self, model_name=ROBERTA_MODEL_NAME, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"載入 {model_name}（device={self.device}）...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_batch(self, texts, batch_size=32):
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            non_empty_idx = [j for j, t in enumerate(batch) if t]
            batch_embeds = np.zeros((len(batch), EMBED_DIM), dtype=np.float32)
            if non_empty_idx:
                sub_texts = [batch[j] for j in non_empty_idx]
                enc = self.tokenizer(
                    sub_texts, padding=True, truncation=True,
                    max_length=128, return_tensors="pt",
                ).to(self.device)
                out = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                summed = (out * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                mean_pooled = (summed / counts).cpu().numpy()
                for k, j in enumerate(non_empty_idx):
                    batch_embeds[j] = mean_pooled[k]
            all_embeds.append(batch_embeds)
            if (i // batch_size) % 20 == 0:
                print(f"  embedding 進度: {min(i + batch_size, len(texts))}/{len(texts)}")
        return np.concatenate(all_embeds, axis=0)


# ---------------------------------------------------------------------------
# Step C: 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reply_level_csv", required=True)
    parser.add_argument("--pheme_path", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    print(f"讀取 {args.reply_level_csv} ...")
    reply_df = pd.read_csv(
        args.reply_level_csv,
        dtype={"thread_id": str, "tweet_id": str, "parent_id": str, "event_id": str},
        low_memory=False,
    )
    print(f"  {len(reply_df)} 列")

    print("\n重新走訪 PHEME 資料夾，取得使用者特徵 ...")
    user_feats = build_user_feature_table(args.pheme_path)
    print(f"  取得 {len(user_feats)} 個 tweet 的使用者特徵")

    print("\n計算 VADER 情感 ...")
    analyzer = SentimentIntensityAnalyzer()

    rows = []
    for _, r in reply_df.iterrows():
        text = r.get("text")
        if isinstance(text, str) and text.strip():
            vs = analyzer.polarity_scores(text)
        else:
            vs = {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 0.0}
            text = ""

        uf = user_feats.get(r["tweet_id"], {})
        followers = uf.get("followers_count")
        friends   = uf.get("friends_count")
        statuses  = uf.get("statuses_count")
        verified  = uf.get("verified", 0)
        age_days  = uf.get("account_age_days")

        offset_sec = r.get("offset_sec")
        offset_sec = 0.0 if pd.isna(offset_sec) else offset_sec
        latency_sec = r.get("reply_latency_sec")
        latency_sec = 0.0 if pd.isna(latency_sec) else latency_sec

        rows.append({
            "thread_id": r["thread_id"],
            "tweet_id": r["tweet_id"],
            "event_id": r["event_id"],
            "is_rumour": r["is_rumour"],
            "coverage_score": r.get("coverage_score", 0.0),
            "depth": r["depth"],
            "offset_sec_log": math.log1p(max(offset_sec, 0.0)),
            "reply_latency_sec_log": math.log1p(max(latency_sec, 0.0)),
            "is_source": r["is_source"],
            "vader_compound": vs["compound"],
            "vader_pos": vs["pos"],
            "vader_neg": vs["neg"],
            "vader_neu": vs["neu"],
            "followers_log": math.log1p(followers) if followers is not None else 0.0,
            "friends_log": math.log1p(friends) if friends is not None else 0.0,
            "statuses_log": math.log1p(statuses) if statuses is not None else 0.0,
            "verified": verified,
            "account_age_days_log": math.log1p(age_days) if age_days is not None else 0.0,
            "_text": text,
        })

    print(f"\n共 {len(rows)} 個節點，開始算 RoBERTa embedding ...")
    embedder = RobertaEmbedder()
    texts = [r["_text"] for r in rows]
    embeddings = embedder.embed_batch(texts)

    for r, emb in zip(rows, embeddings):
        for k in range(EMBED_DIM):
            r[f"emb_{k}"] = float(emb[k])
        del r["_text"]

    node_df = pd.DataFrame(rows)
    col_order = (["thread_id", "tweet_id", "event_id", "is_rumour", "coverage_score"]
                 + NUMERIC_FEAT_COLS
                 + [f"emb_{i}" for i in range(EMBED_DIM)])
    node_df = node_df[col_order]
    node_df.to_csv(args.output_csv, index=False)
    print(f"\n已寫入: {args.output_csv}（{len(node_df)} 列，{len(node_df.columns)} 欄）")

    missing_user = sum(1 for r in rows if user_feats.get(r["tweet_id"]) is None)
    print(f"\n找不到對應使用者特徵的節點數: {missing_user} / {len(rows)}"
          f"（這些節點的 followers/friends/statuses/verified/account_age 全部是 0，"
          f"通常是已刪除推文——is_text_available=0 的那批）")


if __name__ == "__main__":
    main()