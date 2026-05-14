# 檔案位置：backend/app/routers/graph.py
from fastapi import APIRouter
import os
import sys

# 確保程式可以載入位於專案根目錄的 graph_analysis 模組
sys.path.append("/app")
from graph_analysis.builder import build_graph_from_pheme

router = APIRouter()

@router.get("/api/graph")
async def get_graph():
    # 這裡的路徑對應 Docker 容器內掛載的資料夾路徑
    PHEME_PATH = os.getenv("PHEME_PATH", "/app/data/raw/pheme")
    event_name = "sydneysiege"  # 這裡先使用 charliehebdo 事件作為測試
    
    G = build_graph_from_pheme(PHEME_PATH, event_name)
    elements = []
    
    if G.number_of_nodes() > 0:
        # ⚠️ 為了避免前端卡死，我們計算每個節點的連線數 (Degree)，並只取前 100 個最活躍的節點
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:100]
        subG = G.subgraph(top_nodes)
        
        # 1. 轉換節點 (Nodes) 資料格式
        for node in subG.nodes():
            elements.append({
                "data": {
                    "id": str(node),
                    "label": str(node),
                    # 利用連線數來決定節點在前端顯示的大小 (設定最小值 15，最大值 60)
                    "pagerank": min(degrees[node] * 2 + 15, 60), 
                    "is_rumour": False # 目前先預設為 False，未來可根據實際資料標記
                }
            })
            
        # 2. 轉換邊線 (Edges) 資料格式
        for u, v, data in subG.edges(data=True):
            elements.append({
                "data": {
                    "source": str(u),
                    "target": str(v),
                    "is_rumour": data.get("is_rumour", False)
                }
            })

    return {"elements": elements}