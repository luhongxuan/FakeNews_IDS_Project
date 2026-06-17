import networkx as nx
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.models.graph_schema import Event, GraphNode, GraphEdge

def save_graph_to_db(db: Session, G: nx.Graph, pheme_event_id: str, title: str, date: datetime, node_count: int, rumour_count: int, status: str, severity: str, description: str, summary_jsonb: dict | None) -> Event:
    existing_event = db.query(Event).filter_by(pheme_event_id=pheme_event_id).first()
    if existing_event:
        existing_event.title        = title
        existing_event.status       = status
        existing_event.severity     = severity
        existing_event.description  = description
        existing_event.node_count   = G.number_of_nodes()
        existing_event.rumour_count = rumour_count
        existing_event.first_seen_at = date
        existing_event.built_at     = datetime.now(timezone.utc)
        db.commit()
        return existing_event
    
    event = Event(
        pheme_event_id=pheme_event_id,
        title=title,
        query_text=title,
        normalized_claim=title.lower(),
        status=status,
        severity=severity,
        first_seen_at=date,
        node_count=node_count,
        rumour_count=rumour_count,
        description=description,
        summary_jsonb=None,
        built_at=datetime.now(timezone.utc)
    )
    db.add(event)
    db.flush()

    rumour_nodes = set()
    for u, v, data in G.edges(data=True):
        if data.get("is_rumour"):
            rumour_nodes.add(u)
            rumour_nodes.add(v)
 
    screen_name_to_node_id = {}
 
    node_objects = []
    for screen_name in G.nodes():
        node_obj = GraphNode(
            event_id    = event.event_id,
            screen_name = screen_name,
            is_rumour   = screen_name in rumour_nodes,
        )
        node_objects.append(node_obj)
 
    db.bulk_save_objects(node_objects, return_defaults=True)
    db.flush()
 
    for node_obj in node_objects:
        screen_name_to_node_id[node_obj.screen_name] = node_obj.node_id
 
    edge_objects = []
    for u, v, data in G.edges(data=True):
        edge_obj = GraphEdge(
            event_id        = event.event_id,
            src_screen_name = u,
            dst_screen_name = v,
            src_node_id     = screen_name_to_node_id.get(u),
            dst_node_id     = screen_name_to_node_id.get(v),
            is_rumour       = data.get("is_rumour", False),
            thread_id       = data.get("thread_id"),
        )
        edge_objects.append(edge_obj)
 
    db.bulk_save_objects(edge_objects)
    db.commit()
 
    return event

def load_event_summary(db: Session, pheme_event_id: str) -> dict | None:
    event = db.query(Event).filter_by(pheme_event_id=pheme_event_id).first()
    if not event:
        return None
    return {"id": event.pheme_event_id, 
            "title": event.title, 
            "date": event.first_seen_at, 
            "node_count": event.node_count, 
            "rumour_count": event.rumour_count, 
            "status": event.status, 
            "severity": event.severity, 
            "description": event.description
            }