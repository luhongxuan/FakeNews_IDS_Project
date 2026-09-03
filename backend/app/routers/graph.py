from fastapi import APIRouter, Depends, Query
from functools import lru_cache
import os
import sys
import networkx as nx
from sqlalchemy.orm import Session

sys.path.append("/app")
from graph_analysis.builder import build_graph_from_pheme

from app.database import get_db
from app.models.schema import PropagationEvent

router = APIRouter()


@lru_cache(maxsize=16)
def _build_graph_cached(pheme_path: str, event: str):
    """build_graph_from_pheme re-reads every raw PHEME file for the event on
    every call -- fine for small events, but charliehebdo alone is 84,841
    individual JSON files across 2,079 threads with zero caching, which took
    several minutes per call (confirmed by direct timing) and, because the
    endpoint ran synchronously on the one async event loop, froze every
    other request on the server for that whole time. PHEME's raw data never
    changes, so this only needs to happen once per event per process
    lifetime.
    """
    return build_graph_from_pheme(pheme_path, event)


def _graph_to_elements(G: nx.DiGraph) -> list[dict]:
    """Cytoscape `elements` array from a directed user graph.

    Shared by /api/graph (static PHEME event replay) and /api/graph/live
    (accumulated from social-frontend's WebSocket events), so both render
    with the same top-500-by-degree cap and largest-connected-component
    trim -- a client can't tell which source produced a given response.
    """
    elements: list[dict] = []
    if G.number_of_nodes() == 0:
        return elements

    degrees = dict(G.degree())
    top_nodes = set(sorted(degrees, key=degrees.get, reverse=True)[:500])

    edges = [(u, v) for u, v in G.edges()
             if u in top_nodes and v in top_nodes and u != v]

    connected_nodes = set()
    for u, v in edges:
        connected_nodes.add(u)
        connected_nodes.add(v)

    subG = nx.DiGraph()
    subG.add_nodes_from(connected_nodes)
    subG.add_edges_from(edges)

    UG = subG.to_undirected()
    if UG.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(UG), key=len)
        subG = subG.subgraph(largest_cc)

    sub_degrees = dict(subG.degree())

    for node in subG.nodes():
        elements.append({
            "data": {
                "id": str(node),
                "label": str(node),
                "pagerank": min(sub_degrees.get(node, 1) * 3 + 15, 60),
                "is_rumour": False
            }
        })

    for u, v, data in subG.edges(data=True):
        elements.append({
            "data": {
                "source": str(u),
                "target": str(v),
                "is_rumour": data.get("is_rumour", False)
            }
        })

    return elements


@router.get("/api/graph")
def get_graph(event: str = Query(..., description="PHEME 事件名稱")):
    # Plain `def`, not `async def`: FastAPI runs sync path functions in a
    # worker thread automatically, so a slow (uncached) first call for a
    # large event no longer blocks every other request on the server while
    # it runs -- only matters for the first call per event per process,
    # since _build_graph_cached makes every call after that instant.
    PHEME_PATH = os.getenv("PHEME_PATH", "/app/data/raw/pheme")
    # build_graph_from_pheme returns (graph, first_seen_date, rumour_count) --
    # see graph_analysis/builder.py. This previously assigned the whole tuple
    # to G, which meant this endpoint had never actually worked; it only
    # went unnoticed because the frontend was always on mock data until now.
    G, _first_seen_date, _rumour_count = _build_graph_cached(PHEME_PATH, event)
    return {"elements": _graph_to_elements(G), "event": event}


@router.get("/api/graph/live")
def get_live_graph(db: Session = Depends(get_db)):
    """Current full graph state built from every event social-frontend has
    sent to /ws/events so far. Intended for a relationship-graph client to
    call once on first connect, to catch up on everything that happened
    before it joined -- the WebSocket broadcast only covers events sent
    *after* a client is connected.
    """
    rows = (
        db.query(PropagationEvent)
        .filter(PropagationEvent.from_user.isnot(None), PropagationEvent.to_user.isnot(None))
        .all()
    )
    G = nx.DiGraph()
    for row in rows:
        if not row.from_user or not row.to_user or row.from_user == row.to_user:
            continue
        G.add_edge(row.from_user, row.to_user, thread_id=row.thread_id, is_rumour=bool(row.is_rumour))

    return {
        "elements": _graph_to_elements(G),
        "event_count": len(rows),
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
    }