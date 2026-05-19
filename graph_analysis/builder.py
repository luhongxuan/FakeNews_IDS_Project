import os, json
import networkx as nx
from dotenv import load_dotenv

def build_graph_from_pheme(base_path: str, event: str) -> nx.DiGraph:
    """
    從單一 PHEME 事件建構傳播圖
    回傳 DiGraph，節點是 user，邊是傳播方向
    """
    G = nx.DiGraph()
    event_path = os.path.join(base_path, event + "-all-rnr-threads")
    print(f"正在建構圖：{event_path}")
    for label in ["rumours", "non-rumours"]:
        label_path = os.path.join(event_path, label)
        if not os.path.exists(label_path):
            continue

        for thread_id in os.listdir(label_path):
            thread_path = os.path.join(label_path, thread_id)
            is_rumour = (label == "rumours")
            source_user = _read_source_user(thread_path)
            if not source_user:
                continue

            reactions_path = os.path.join(thread_path, "reactions")
            if not os.path.exists(reactions_path):
                continue

            for fname in os.listdir(reactions_path):
                fpath = os.path.join(reactions_path, fname)
                try:
                    with open(fpath, encoding="utf-8", errors="ignore") as f:
                        tweet = json.load(f)
                        reply_user = tweet.get("user", {}).get("screen_name")
                        if reply_user and source_user:
                            # 邊方向：source → reply（訊息往外擴散）
                            G.add_edge(source_user, reply_user,
                                       thread_id=thread_id,
                                       is_rumour=is_rumour)
                except:
                    continue

    return G

def _read_source_user(thread_path: str):
    source_path = os.path.join(thread_path, "source-tweets")
    if not os.path.exists(source_path):
        return None
    for fname in os.listdir(source_path):
        if fname.startswith("._"):
            continue
        fpath = os.path.join(source_path, fname)
        try:
            with open(fpath, encoding="utf-8", errors="ignore") as f:
                tweet = json.load(f)
                return tweet.get("user", {}).get("screen_name")
        except:
            continue
    return None
