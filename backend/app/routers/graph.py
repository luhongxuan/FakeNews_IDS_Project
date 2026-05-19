# 檔案位置：backend/app/routers/graph.py
from fastapi import APIRouter
import os
import sys
import networkx as nx
import networkx as nx  # <--- 🌟 確保這裡有引入 networkx

sys.path.append("/app")
from graph_analysis.builder import build_graph_from_pheme

router = APIRouter()

@router.get("/api/graph")
async def get_graph():
    PHEME_PATH = os.getenv("PHEME_PATH", "/app/data/raw/pheme")
    event_name = "sydneysiege"  
    
    G = build_graph_from_pheme(PHEME_PATH, event_name)
    elements = []
    
    if G.number_of_nodes() > 0:
        # 🌟 演算法更換：雪球擴展法 (BFS)
        
        # 1. 找出全網度數 (連線數) 最高的「核心源頭節點」
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:100]
        subG = G.subgraph(top_nodes)
        
        # 1. 轉換節點 (Nodes) 資料格式
        for node in subG.nodes():
            elements.append({
                "data": {
                    "id": str(node),
                    "label": str(node),
                    # 視覺化大小調整：依照在「子圖」中的重要性來放大縮小
                    "pagerank": min(sub_degrees.get(node, 1) * 3 + 15, 60), 
                    "is_rumour": False 
                }
            })
            
        # 轉換 Edges
        for u, v, data in subG.edges(data=True):
            elements.append({
                "data": {
                    "source": str(u),
                    "target": str(v),
                    "is_rumour": data.get("is_rumour", False)
                }
            })

    return {"elements": elements}
