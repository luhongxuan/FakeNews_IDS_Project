# backend/app/routers/graph.py
from fastapi import APIRouter, Query
from graph_analysis.builder import build_graph_from_pheme
from graph_analysis.centrality import compute_centrality, get_top_nodes
from graph_analysis.intervention import run_intervention_comparison
import networkx as nx

router = APIRouter()

PHEME_PATH = "/app/data/raw/pheme"

@router.get("/graph")
async def get_graph(
    event: str = Query(..., description="PHEME 事件名稱"),
    k: int = Query(5, description="干預節點數量"),
    beta: float = Query(0.1),
    gamma: float = Query(0.05),
):
    G = build_graph_from_pheme(PHEME_PATH, event)

    centrality = compute_centrality(G)
    top_nodes  = get_top_nodes(centrality, "pagerank", top_k=10)

    comparison = run_intervention_comparison(G, k=k, beta=beta, gamma=gamma)

    return {
        "event": event,
        "graph_stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        },
        "top_nodes": top_nodes,
        "intervention_comparison": comparison,
    }