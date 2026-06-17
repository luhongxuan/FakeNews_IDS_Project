from fastapi import APIRouter, Query
import os
import sys
import networkx as nx

sys.path.append("/app")
from graph_analysis.builder import build_graph_from_pheme

router = APIRouter()

@router.get("/api/graph")
async def get_graph(event: str = Query(..., description="PHEME 事件名稱")):
    PHEME_PATH = os.getenv("PHEME_PATH", "/app/data/raw/pheme")
    
    G = build_graph_from_pheme(PHEME_PATH, event)
    elements = []

    if G.number_of_nodes() > 0:
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

    return {"elements": elements, "event": event}