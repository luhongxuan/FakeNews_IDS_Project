"""
既有 hydrated_tweets.jsonl 的重複內容清理腳本
================================================
用途：不需要重爬，直接對已經蒐集到的資料做後製清理。把「內容完全
相同（text + created_at + 三個互動數都一樣）」的多個 tweet_id 群組
起來，判定為同一則原推文被轉推收合顯示的結果，只留一個代表性內容，
其餘標記為 duplicate_of，並補上一個 is_canonical 欄位方便後續分析
篩選。

用法：
    python dataset_dedup_cleanup.py --tweets_jsonl hydrated_tweets.jsonl \
        --output cleaned_tweets.jsonl

輸出檔案每筆記錄會多兩個欄位：
    - is_canonical: 這筆是不是它所屬內容群組裡的代表（True/False）
    - duplicate_group_id: 同一群組的記錄共用同一個編號，方便之後
      查詢「這些 tweet_id 其實是同一則原推文」

下游用法建議：
    - 算文字/情感/關鍵詞特徵時，只用 is_canonical=True 的記錄，
      避免同一段內容被算很多次、灌水統計結果
    - 建圖的時候，每個 tweet_id 節點還是保留（因為 tree 結構需要
      這個節點存在），但它的文字特徵可以查 duplicate_group_id 對到
      的代表內容來填，而不是各自查自己抓到的（可能是空的或錯誤歸屬
      的）內容
"""

import json
import argparse
import collections


def build_signature(r):
    return (
        r.get("text"), r.get("created_at"), r.get("reply_count"),
        r.get("retweet_count"), r.get("like_count"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tweets_jsonl", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = []
    bad_lines = 0
    with open(args.tweets_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1

    print(f"讀入 {len(records)} 筆，JSON 格式錯誤跳過 {bad_lines} 筆")

    # 只對 status=ok 且有文字的記錄做分組；其他 status 原樣保留，
    # 不參與去重（它們本來就沒有內容可比對）。
    sig_to_group_id = {}
    sig_counts = collections.Counter()
    for r in records:
        if r.get("status") == "ok" and r.get("text"):
            sig = build_signature(r)
            sig_counts[sig] += 1

    next_group_id = 1
    output_records = []
    canonical_seen = set()

    for r in records:
        if r.get("status") == "ok" and r.get("text"):
            sig = build_signature(r)
            if sig_counts[sig] > 1:
                if sig not in sig_to_group_id:
                    sig_to_group_id[sig] = next_group_id
                    next_group_id += 1
                group_id = sig_to_group_id[sig]
                is_canonical = sig not in canonical_seen
                if is_canonical:
                    canonical_seen.add(sig)
                r = dict(r)
                r["duplicate_group_id"] = group_id
                r["is_canonical"] = is_canonical
            else:
                # 內容獨一無二，本身就是代表，不屬於任何重複群組
                r = dict(r)
                r["duplicate_group_id"] = None
                r["is_canonical"] = True
        else:
            r = dict(r)
            r["duplicate_group_id"] = None
            r["is_canonical"] = None  # 不適用（沒有文字內容可比對）
        output_records.append(r)

    with open(args.output, "w", encoding="utf-8") as out:
        for r in output_records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_ok_with_text = sum(1 for r in records if r.get("status") == "ok" and r.get("text"))
    total_groups = next_group_id - 1
    total_canonical = sum(1 for r in output_records if r.get("is_canonical") is True)
    total_duplicate_only = sum(
        1 for r in output_records
        if r.get("is_canonical") is False
    )

    print(f"\n status=ok 且有文字: {total_ok_with_text}")
    print(f" 重複群組數: {total_groups}")
    print(f" 代表性記錄（is_canonical=True）: {total_canonical}")
    print(f" 純重複記錄（is_canonical=False，建議做內容特徵時排除）: {total_duplicate_only}")
    print(f"\n 已寫入: {args.output}")


if __name__ == "__main__":
    main()