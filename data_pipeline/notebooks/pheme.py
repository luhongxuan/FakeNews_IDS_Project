import os
import json
import pandas as pd

# ── 載入所有推文 ──────────────────────────────────────
def load_pheme(base_path):
    records = []
    
    for event in os.listdir(base_path):
        event_path = os.path.join(base_path, event)
        if not os.path.isdir(event_path):
            continue
            
        for label in ["rumours", "non-rumours"]:
            label_path = os.path.join(event_path, label)
            if not os.path.exists(label_path):
                continue
                
            for thread_id in os.listdir(label_path):
                thread_path = os.path.join(label_path, thread_id)
                
                # 讀原始推文
                source_path = os.path.join(thread_path, "source-tweets")
                if os.path.exists(source_path):
                    for fname in os.listdir(source_path):
                        if fname.startswith("._"):
                            continue
                        fpath = os.path.join(source_path, fname)
                        if os.path.getsize(fpath) == 0:
                            continue
                        try:
                            with open(os.path.join(source_path, fname), encoding="utf-8", errors="ignore") as f:
                                tweet = json.load(f)
                                records.append({
                                    "event"     : event,
                                    "thread_id" : thread_id,
                                    "tweet_id"  : tweet.get("id_str"),
                                    "text"      : tweet.get("text"),
                                    "user"      : tweet.get("user", {}).get("screen_name"),
                                    "created_at": tweet.get("created_at"),
                                    "retweet_count": tweet.get("retweet_count"),
                                    "is_rumour" : label == "rumours",
                                    "type"      : "source"
                                })
                        except json.JSONDecodeError:
                            print(f"無法解析 JSON：{fpath}")
                            continue
    return pd.DataFrame(records)

# ── 載入資料 ──────────────────────────────────────────
print(os.getcwd())
df = load_pheme("data/raw/pheme")
print(f"總推文數：{len(df)}")
print(f"欄位：{df.columns.tolist()}")
df.head()