from fastapi import APIRouter, Query
import os
import sys
import networkx as nx
sys.path.append("/app")

from graph_analysis.builder import build_graph_from_pheme

router = APIRouter()
["charliehebdo", "sydneysiege", "ottawashooting", "putinmissing", "ukraine-ghost", "taiwan-quake", "paris-olympics"]
eventTitle = {"charliehebdo": '查理週刊槍擊案',
              "sydneysiege": '雪梨人質事件',
              "ottawashooting": '渥太華槍擊案',
              "putinmissing": '普丁失蹤謠言',
              "ukraine-ghost": '烏克蘭幽靈坦克',
              "taiwan-quake": '台灣地震預測',
              "paris-olympics": '巴黎奧運取消謠言',
              }
@router.get("/api/events/{event_id}/summary")
def get_event_summary(event_id: str):
    PHEME_PATH = os.getenv("PHEME_PATH", "../../../data/raw/pheme")

    G, date, rumours_num = build_graph_from_pheme(PHEME_PATH, event_id)

    return {"id": event_id, 
            "title": eventTitle.get(event_id, '未知事件'), 
            "date": date, 
            "nodes": len(G.nodes()), 
            "rumours": rumours_num, 
            "status": 'monitoring', 
            "severity": 'high', 
            "description": '法國巴黎《查理週刊》總部遭恐怖攻擊，Twitter 上迅速爆發大規模聲援與陰謀論。' }

# print(get_event_summary("charliehebdo"))