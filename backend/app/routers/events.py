from fastapi import APIRouter, Query, Depends
import os
import sys
import random
import networkx as nx
from sqlalchemy.orm import Session
sys.path.append("/app")

from app.database import get_db
from app.services.graph_store import save_graph_to_db, load_event_summary
from graph_analysis.builder import build_graph_from_pheme

router = APIRouter()
EVENT_IDS = ["charliehebdo", "ebola-essien", "ferguson", "germanwings-crash", "gurlitt", "ottawashooting", "prince-toronto", "putinmissing", "sydneysiege"]
EVENT_TITLE = {"charliehebdo": '查理週刊槍擊案',
              "ebola-essien": '埃博拉疫情',
              "ferguson": '弗格森事件',
              "germanwings-crash": '德國之翼墜機事件',
              "gurlitt": '古利特失竊事件',
              "ottawashooting": '渥太華槍擊案',
              "prince-toronto": '多倫多王子事件',
              "putinmissing": '普丁失蹤事件',
              "sydneysiege": '雪梨人質事件',
}

EVENT_META = {
    "charliehebdo":  {"status": "monitoring", "severity": "high",
                      "desc": "法國巴黎《查理週刊》總部遭恐怖攻擊，Twitter 上迅速爆發大規模聲援與陰謀論。"},
    "ebola-essien":  {"status": "monitoring", "severity": "low",    "desc": "網傳知名足球員 Michael Essien 感染伊波拉病毒，為惡作劇。"},
    "ferguson":     {"status": "monitoring", "severity": "medium", "desc": "2014 年美國密蘇里州弗格森市爆發警察暴力事件，Twitter 上充斥著各種謠言和陰謀論。"},
    "germanwings-crash": {"status": "monitoring", "severity": "medium", "desc": "2015 年德國之翼航空公司 9525 航班墜機事件，網路上流傳著各種關於墜機原因的謠言和陰謀論。"},
    "gurlitt":     {"status": "monitoring", "severity": "low",    "desc": "2013 年德國古利特失竊事件，網路上流傳著各種關於失竊藝術品的下落和真相的謠言和陰謀論。"},
    "ottawashooting": {"status": "monitoring", "severity": "high",   "desc": "2014 年加拿大渥太華槍擊案，網路上充斥著各種關於槍手動機和背景的謠言和陰謀論。"},
    "prince-toronto": {"status": "monitoring", "severity": "low",    "desc": "2016 年多倫多王子事件，網路上流傳著各種關於王子身份和動機的謠言和陰謀論。"},
    "putinmissing": {"status": "monitoring", "severity": "medium", "desc": "2015 年普丁失蹤事件，網路上流傳著各種關於普丁行蹤和健康狀況的謠言和陰謀論。"},
    "sydneysiege": {"status": "monitoring", " severity": "high",   "desc": "2014 年雪梨人質事件，網路上充斥著各種關於事件經過和涉案人員的謠言和陰謀論。"},
}
@router.get("/api/events/{event_id}/summary")
def get_event_summary(event_id: str, db: Session = Depends(get_db)):
    cached = load_event_summary(db, event_id)

    if cached:
        print(f"[DB HIT] {event_id}")
        return cached

    print(f"[DB MISS] 正在建構圖：{event_id}")
    PHEME_PATH = os.getenv("PHEME_PATH", "../../../data/raw/pheme")

    G, date, rumours_num = build_graph_from_pheme(PHEME_PATH, event_id)

    event = save_graph_to_db(
        db             = db,
        G              = G,
        pheme_event_id = event_id,
        title          = EVENT_TITLE.get(event_id, event_id),
        date           = date,
        node_count     = G.number_of_nodes(),
        rumour_count   = rumours_num,
        status         = random.sample(["monitoring", "pending", "archived"], 1)[0],
        severity       = random.sample(["low", "medium", "high"], 1)[0],
        description    = EVENT_META.get(event_id, {}).get("desc", ""),
        summary_jsonb  = None,
    )
 
    return {
        "id":          event.pheme_event_id,
        "title":       event.title,
        "date":        str(event.first_seen_at) if event.first_seen_at else None,
        "node_count":  event.node_count,
        "rumour_count":event.rumour_count,
        "status":      event.status,
        "severity":    event.severity,
        "description": event.description,
    }

# print(get_event_summary("charliehebdo"))