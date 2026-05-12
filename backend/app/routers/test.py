import sys
sys.path.append("/home/luhongxuan/FakeNews_IDS_Project")
from graph_analysis.builder import build_graph_from_pheme
from graph_analysis.centrality import compute_centrality, get_top_nodes
from graph_analysis.intervention import run_intervention_comparison
import networkx as nx

def get_graph(event="charliehebdo", k=5, beta=0.1, gamma=0.05):
    G = build_graph_from_pheme("/home/luhongxuan/FakeNews_IDS_Project/data/raw/pheme", event)

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

print(get_graph())