import networkx as nx
import os
from builder import build_graph_from_pheme
from dotenv import load_dotenv


load_dotenv()
PHEME_PATH = os.getenv("PHEME_PATH")

G = build_graph_from_pheme(PHEME_PATH, "charliehebdo")

def compute_centrality(G: nx.DiGraph) -> dict:
    """
    計算三種中心性，回傳 dict
    """
    UG = G.to_undirected()

    degree    = nx.degree_centrality(UG)
    between   = nx.betweenness_centrality(UG, normalized=True, k=500)
    pagerank  = nx.pagerank(G, alpha=0.85)

    nodes_info = {}
    for node in G.nodes():
        nodes_info[node] = {
            "degree_centrality":      round(degree.get(node, 0), 4),
            "betweenness_centrality": round(between.get(node, 0), 4),
            "pagerank":               round(pagerank.get(node, 0), 4),
            "degree":                 G.degree(node),
        }

    return nodes_info


def get_top_nodes(centrality_data: dict, metric: str, top_k: int = 10) -> list:
    """
    取特定指標排名前 k 的節點
    metric: 'degree_centrality' | 'betweenness_centrality' | 'pagerank'
    """
    print(centrality_data)
    sorted_nodes = sorted(
        centrality_data.items(),
        key=lambda x: x[1][metric],
        reverse=True
    )
    return [{"node": n, **data} for n, data in sorted_nodes[:top_k]]

centrality_data = compute_centrality(G)
# print(centrality_data)
top_degree = get_top_nodes(centrality_data, "degree_centrality")
top_betweenness = get_top_nodes(centrality_data, "betweenness_centrality")
top_pagerank = get_top_nodes(centrality_data, "pagerank")