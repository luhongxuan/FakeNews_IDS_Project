# 檔案位置：backend/app/routers/graph.py
from fastapi import APIRouter
import os
import sys
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
        root_node = max(degrees, key=degrees.get) 
        
        # 2. 使用 Queue 來向外一層一層擴散，直到收集滿 50 個相連的節點
        sampled_nodes = set([root_node])
        queue = [root_node]
        target_node_count = 10  # 你可以在這裡控制要顯示的節點數量
        
        while queue and len(sampled_nodes) < target_node_count:
            current = queue.pop(0)
            
            # 找出與 current 節點有相連的所有人 (無論是轉推或被轉推)
            for neighbor in nx.all_neighbors(G, current):
                if neighbor not in sampled_nodes:
                    sampled_nodes.add(neighbor)
                    queue.append(neighbor)
                    
                    # 只要人數一滿，立刻煞車停止擴散
                    if len(sampled_nodes) >= target_node_count:
                        break

        # 3. 根據這群彼此相連的節點，裁切出子圖
        subG = G.subgraph(sampled_nodes)
        
        # 重新計算這個子圖內的 degree，用來決定前端圓圈的大小
        sub_degrees = dict(subG.degree())
        
        # --- 轉換為前端 Cytoscape 格式 ---
        # 轉換 Nodes
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