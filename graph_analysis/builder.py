import os, json
import networkx as nx
from datetime import datetime

def build_graph_from_pheme(base_path: str, event: str):
    G = nx.DiGraph()
    date = None
    rumours_num = 0
    event_path = os.path.join(base_path, event + "-all-rnr-threads")

    for label in ["rumours", "non-rumours"]:
        label_path = os.path.join(event_path, label)
        if not os.path.exists(label_path):
            continue

        for thread_id in os.listdir(label_path):
            thread_path = os.path.join(label_path, thread_id)
            is_rumour = (label == "rumours")
            if is_rumour:
                rumours_num += 1

            # 1. 讀取所有推文，建立 tweet_id → screen_name 對照表
            tweet_to_user = {}
            source_path = os.path.join(thread_path, "source-tweets")
            reactions_path = os.path.join(thread_path, "reactions")

            for folder in [source_path, reactions_path]:
                if not os.path.exists(folder):
                    continue
                for fname in os.listdir(folder):
                    if fname.startswith("._"):
                        continue
                    fpath = os.path.join(folder, fname)
                    try:
                        with open(fpath, encoding="utf-8", errors="ignore") as f:
                            tweet = json.load(f)
                            tid = tweet.get("id_str")
                            user = tweet.get("user", {}).get("screen_name")
                            created_at = tweet.get("created_at")
                            if tid and user:
                                tweet_to_user[tid] = user
                            # 更新最早日期
                            if created_at:
                                parsed = _parse_data(created_at)
                                if parsed and (not date or parsed < date):
                                    date = parsed
                    except:
                        continue

            # 2. 讀取 structure.json，遞迴建邊
            structure_path = os.path.join(thread_path, "structure.json")
            if not os.path.exists(structure_path):
                continue
            try:
                with open(structure_path, encoding="utf-8") as f:
                    structure = json.load(f)
            except:
                continue
            def traverse(node_dict, parent_id=None):
                if node_dict == []:
                    return
                for tweet_id, children in node_dict.items():
                    if parent_id:
                        src = tweet_to_user.get(parent_id)
                        dst = tweet_to_user.get(tweet_id)
                        # 有對應 user 且不是自環才加邊
                        if src and dst and src != dst:
                            G.add_edge(src, dst,
                                       thread_id=thread_id,
                                       is_rumour=is_rumour)
                    traverse(children, tweet_id)

            traverse(structure)

    return G, date.strftime("%Y-%m-%d") if date else None, rumours_num


def _parse_data(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
    except:
        return None
    
if __name__ == "__main__":
    G, date, rumours_num = build_graph_from_pheme("/home/luhongxuan/FakeNews_IDS_Project/data/raw/pheme", "charliehebdo")
    print(f"圖的節點數: {G.number_of_nodes()}")
    print(f"圖的邊數: {G.number_of_edges()}")
    print(f"最早日期: {date}")
    print(f"謠言數量: {rumours_num}")