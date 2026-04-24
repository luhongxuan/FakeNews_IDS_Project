import os, json
import networkx as nx
from dotenv import load_dotenv

load_dotenv()

PHEME_PATH = os.getenv("PHEME_PATH")

def build_graph_from_pheme(base_path: str, event: str) -> nx.DiGraph:
    """
    從單一 PHEME 事件建構傳播圖
    回傳 DiGraph，節點是 user，邊是傳播方向
    """

    event_path = os.path.join(base_path, event)
    print(os.listdir(event_path))
    

print(os.listdir(PHEME_PATH))
build_graph_from_pheme(PHEME_PATH, "charliehebdo-all-rnr-threads")
