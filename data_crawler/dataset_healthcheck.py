"""
Twitter15/16 Hydration 資料集健檢腳本
================================================
用途：對照 twitter_hydration_scraper.py 產出的 hydrated_tweets.jsonl，
一次算出：status 分布、重複內容污染比例（轉推被收合成原文的情況）、
文字品質、情感分布、高頻詞、時間分布，以及（可選）跟 label.txt /
tree_dir 對照後的標籤涵蓋率與整體 hydration 涵蓋率。

用法（在你本地端跑，不用把 30000 筆的原始檔案傳上來）：
    python dataset_healthcheck.py --tweets_jsonl hydrated_tweets.jsonl

    # 有 label.txt 的話可以加這個，會多算標籤涵蓋率：
    python dataset_healthcheck.py --tweets_jsonl hydrated_tweets.jsonl \
        --label_txt label.txt

    # 有 tree 資料夾的話可以加這個，會多算整體 hydration 涵蓋率：
    python dataset_healthcheck.py --tweets_jsonl hydrated_tweets.jsonl \
        --label_txt label.txt --tree_dir twitter15/tree

需要先安裝（沒裝過情感分析那段會自動跳過，不影響其他部分）：
    pip install vaderSentiment

跑完把終端機輸出整段複製貼給 Claude，就能判斷資料集狀況，不需要
把原始檔案傳上來。
"""

import json
import re
import os
import argparse
import collections
import statistics
from datetime import datetime
from ast import literal_eval

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_records(path):
    records = []
    bad_lines = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1
    return records, bad_lines


def analyze_status(records):
    section("1. Status 分布")
    total = len(records)
    if total == 0:
        print("  沒有資料。")
        return collections.Counter()
    counts = collections.Counter(r.get("status", "unknown") for r in records)
    for status, c in counts.most_common():
        print(f"  {status:35s} {c:6d}  ({c / total * 100:5.1f}%)")
    print(f"\n  總筆數: {total}")
    return counts


def build_signature(r):
    return (
        r.get("text"), r.get("created_at"), r.get("reply_count"),
        r.get("retweet_count"), r.get("like_count"),
    )


def analyze_duplicates(ok_records):
    section("2. 重複內容污染分析（同一內容被多個 tweet_id 重複抓到，"
            "常見於轉推被收合成原文顯示的情況）")
    sig_map = collections.defaultdict(list)
    for r in ok_records:
        if r.get("text") is None:
            continue
        sig_map[build_signature(r)].append(r["tweet_id"])

    ok_with_text = sum(1 for r in ok_records if r.get("text") is not None)
    dup_groups = {sig: ids for sig, ids in sig_map.items() if len(ids) > 1}
    total_in_dup_groups = sum(len(ids) for ids in dup_groups.values())
    unique_sig_count = len(sig_map)
    redundant = total_in_dup_groups - len(dup_groups)

    print(f"  status=ok 且有文字的筆數: {ok_with_text}")
    print(f"  不重複內容簽章數: {unique_sig_count}")
    print(f"  重複組數（同一簽章對應多個 tweet_id）: {len(dup_groups)}")
    if ok_with_text:
        print(f"  落在重複組裡的 tweet_id 總數: {total_in_dup_groups} "
              f"({total_in_dup_groups / ok_with_text * 100:.1f}% of ok-with-text)")
        print(f"  真正冗餘（扣掉每組留一個代表）的筆數: {redundant} "
              f"({redundant / ok_with_text * 100:.1f}% of ok-with-text)")
    return unique_sig_count, dup_groups


def analyze_redirect_status(records):
    redirected = [r for r in records if r.get("status") == "redirected_possible_retweet"]
    print(f"\n  status=redirected_possible_retweet 筆數: {len(redirected)}"
          f"（這是爬蟲主動偵測到轉推導向而攔下的筆數，跟上面『內容重複"
          f"但沒被主動偵測到』的筆數是分開的兩個來源，兩者加起來才是"
          f"轉推污染的完整規模）")


def analyze_text_quality(ok_records):
    section("3. 文字品質")
    ok_with_text = [r for r in ok_records if r.get("text")]
    ok_empty = [r for r in ok_records
                if r.get("text") is None and r.get("username") is None]
    print(f"  status=ok 但 text/username 全空（疑似選擇器失效或未知頁面狀態）: "
          f"{len(ok_empty)} / {len(ok_records) if ok_records else 0}")

    lengths = [len(r["text"]) for r in ok_with_text]
    if lengths:
        print(f"\n  文字長度統計（字元數，去重前）：")
        print(f"    平均: {statistics.mean(lengths):.1f}")
        print(f"    中位數: {statistics.median(lengths):.1f}")
        print(f"    最短: {min(lengths)}  最長: {max(lengths)}")
        very_short = sum(1 for l in lengths if l < 5)
        print(f"    長度 < 5 字元的筆數: {very_short} "
              f"({very_short / len(lengths) * 100:.1f}%，可能是純表情符號/連結)")
    else:
        print("  沒有帶文字的 ok 記錄。")
    return ok_with_text


def dedup_texts(ok_with_text):
    sig_seen = set()
    unique_texts = []
    for r in ok_with_text:
        sig = build_signature(r)
        if sig not in sig_seen:
            sig_seen.add(sig)
            unique_texts.append(r["text"])
    return unique_texts


def analyze_sentiment(unique_texts):
    section("4. 情感分布（VADER，去重後）")
    if not HAS_VADER:
        print("  未安裝 vaderSentiment，跳過。請先 pip install vaderSentiment")
        return
    if not unique_texts:
        print("  沒有可分析的文字。")
        return
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(t)["compound"] for t in unique_texts]
    print(f"  去重後樣本數: {len(scores)}")
    print(f"  平均: {statistics.mean(scores):.3f}")
    if len(scores) > 1:
        print(f"  標準差: {statistics.stdev(scores):.3f}")
    print(f"  中位數: {statistics.median(scores):.3f}")
    pos = sum(1 for s in scores if s > 0.05)
    neg = sum(1 for s in scores if s < -0.05)
    neu = len(scores) - pos - neg
    print(f"  正向 (>0.05): {pos} ({pos / len(scores) * 100:.1f}%)")
    print(f"  負向 (<-0.05): {neg} ({neg / len(scores) * 100:.1f}%)")
    print(f"  中性: {neu} ({neu / len(scores) * 100:.1f}%)")


def analyze_keywords(unique_texts, top_n=30):
    section("5. 高頻詞（去除 @提及、URL、常見英文停用詞）")
    if not unique_texts:
        print("  沒有可分析的文字。")
        return
    stopwords = set("""a an the is are was were be been being to of and or but in on at for with
    this that these those i you he she it we they my your his her its our their me him them us
    not no so if as it's don't just via rt amp""".split())
    url_pat = re.compile(r"http\S+|www\.\S+|\S+\.\S{2,3}/\S+")
    mention_pat = re.compile(r"@\w+")
    word_pat = re.compile(r"[a-zA-Z']+")

    counter = collections.Counter()
    for t in unique_texts:
        t_clean = url_pat.sub(" ", t)
        t_clean = mention_pat.sub(" ", t_clean)
        for w in word_pat.findall(t_clean.lower()):
            if len(w) > 2 and w not in stopwords:
                counter[w] += 1

    if not counter:
        print("  沒有抓到任何關鍵詞（可能大多是非英文文字）。")
        return
    for w, c in counter.most_common(top_n):
        print(f"  {w:20s} {c}")


def analyze_temporal(ok_records):
    section("6. 時間分布（created_at 年份，事件年代 sanity check）")
    years = collections.Counter()
    parse_fail = 0
    for r in ok_records:
        ca = r.get("created_at")
        if not ca:
            continue
        dt = None
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                dt = datetime.strptime(ca, fmt)
                break
            except ValueError:
                continue
        if dt:
            years[dt.year] += 1
        else:
            parse_fail += 1
    if years:
        for y, c in sorted(years.items()):
            print(f"  {y}: {c}")
    else:
        print("  沒有可解析的時間欄位。")
    if parse_fail:
        print(f"  時間格式解析失敗: {parse_fail} 筆")


def analyze_with_labels(records, label_path):
    section("7. 對照 label.txt（只能對到 tweet_id 剛好等於來源推文/eid 的部分）")
    labels = {}
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, eid = line.split(":")
            labels[eid] = label

    label_dist = collections.Counter(labels.values())
    print(f"  label.txt 事件總數: {len(labels)}")
    print(f"  標籤分布: {dict(label_dist)}")

    ok_records = [r for r in records if r.get("status") == "ok" and r.get("text")]
    matched = [r for r in ok_records if r["tweet_id"] in labels]
    print(f"\n  hydrated 資料中，tweet_id 剛好是來源推文的筆數: {len(matched)}")

    matched_labels = collections.Counter(labels[r["tweet_id"]] for r in matched)
    print(f"  這些筆數的標籤分布: {dict(matched_labels)}")

    print(f"\n  各標籤的來源推文涵蓋率:")
    for lbl, cnt in sorted(label_dist.items()):
        pct = matched_labels.get(lbl, 0) / cnt * 100 if cnt else 0
        print(f"    {lbl:12s} {pct:5.1f}%  ({matched_labels.get(lbl, 0)}/{cnt})")


def analyze_with_tree(records, tree_dir):
    section("8. 對照 tree_dir（整體 hydration 涵蓋率）")
    all_ids = set()
    for fname in os.listdir(tree_dir):
        path = os.path.join(tree_dir, fname)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "->" not in line:
                    continue
                for chunk in line.split("->"):
                    try:
                        _uid, tweet_id, _t = literal_eval(chunk.strip())
                    except (ValueError, SyntaxError):
                        continue
                    if tweet_id and tweet_id != "ROOT":
                        all_ids.add(tweet_id)

    hydrated_ids = {r["tweet_id"] for r in records}
    covered = all_ids & hydrated_ids
    print(f"  tree 檔案中總不重複 tweet_id 數: {len(all_ids)}")
    if all_ids:
        print(f"  已經碰過（不論成功與否）的: {len(covered)} "
              f"({len(covered) / len(all_ids) * 100:.1f}%)")
    print(f"  尚未碰過的: {len(all_ids) - len(covered)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tweets_jsonl", required=True)
    parser.add_argument("--label_txt", default=None)
    parser.add_argument("--tree_dir", default=None)
    args = parser.parse_args()

    print(f"讀取 {args.tweets_jsonl} ...")
    records, bad_lines = load_records(args.tweets_jsonl)
    print(f"成功解析 {len(records)} 筆，JSON 格式錯誤跳過 {bad_lines} 筆")

    analyze_status(records)

    ok_records = [r for r in records if r.get("status") == "ok"]
    analyze_duplicates(ok_records)
    analyze_redirect_status(records)

    ok_with_text = analyze_text_quality(ok_records)
    unique_texts = dedup_texts(ok_with_text)
    print(f"\n  （後續情感/關鍵詞分析都是用去重後的 {len(unique_texts)} 筆不重複文字）")

    analyze_sentiment(unique_texts)
    analyze_keywords(unique_texts)
    analyze_temporal(ok_records)

    if args.label_txt:
        analyze_with_labels(records, args.label_txt)
    if args.tree_dir:
        analyze_with_tree(records, args.tree_dir)

    section("完成")
    print("請把上面全部輸出複製貼給 Claude，用於判斷資料集狀況。")


if __name__ == "__main__":
    main()