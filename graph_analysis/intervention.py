import random
import os
import sys
import time
from collections import deque
from typing import Optional
import networkx as nx
from graph_analysis.simulation import run_sir_simulation

STRATEGIES = ["greedy"]

if os.path.exists("/home/luhongxuan/FakeNews_IDS_Project"):
    sys.path.append("/home/luhongxuan/FakeNews_IDS_Project")

def select_intervention_nodes(
    G: nx.Graph,
    strategy: str,
    k: int = 5,
    pheme_trees: Optional[dict] = None,
) -> list:
    if strategy not in STRATEGIES:
        raise ValueError(f"未知策略：{strategy}，可用策略：{STRATEGIES}")

    UG = G.to_undirected() if G.is_directed() else G

    dispatch = {
        "random":      lambda: _strategy_random(UG, k),
        "degree":      lambda: _strategy_degree(UG, k),
        "betweenness": lambda: _strategy_betweenness(UG, k),
        "greedy":      lambda: _strategy_greedy(UG, k),
        "min_cut":     lambda: _strategy_min_cut(UG, k),
    }
    return dispatch[strategy]()

def run_intervention_comparison(
    G: nx.Graph,
    k: int = 5,
    beta: float = 0.1,
    gamma: float = 0.05,
    iterations: int = 50,
    pheme_trees: Optional[dict] = None,
) -> dict:
    results = {}

    baseline_sim  = run_sir_simulation(G, beta, gamma, iterations=iterations)
    baseline_peak = baseline_sim["peak_infected"]

    results["baseline"] = {
        "peak_infected":  baseline_peak,
        "final_infected": _final_infected(baseline_sim),
        "removed_nodes":  [],
        "reduction_rate": 0.0,
        "trends":         baseline_sim["trends"],
    }

    active = STRATEGIES if pheme_trees else [s for s in STRATEGIES if s != "tree_dp"]

    for strategy in active:
        print(strategy)
        start_time = time.time()
        nodes_to_remove = select_intervention_nodes(G, strategy, k, pheme_trees)
        elapsed = time.time() - start_time

        G_i = G.copy()
        G_i.remove_nodes_from(nodes_to_remove)

        sim = run_sir_simulation(G_i, beta, gamma, iterations=iterations)

        results[strategy] = {
            "peak_infected":  sim["peak_infected"],
            "final_infected": _final_infected(sim),
            "removed_nodes":  nodes_to_remove,
            "reduction_rate": round(
                1 - sim["peak_infected"] / max(baseline_peak, 1), 4
            ),
            "trends": sim["trends"],
            "execution_time": elapsed
        }

    return results


def _strategy_random(G: nx.Graph, k: int) -> list:
    return random.sample(list(G.nodes()), min(k, G.number_of_nodes()))


def _strategy_degree(G: nx.Graph, k: int) -> list:
    sorted_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in sorted_nodes[:k]]


def _strategy_betweenness(G: nx.Graph, k: int) -> list:
    between = nx.betweenness_centrality(G, normalized=True)
    sorted_nodes = sorted(between.items(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in sorted_nodes[:k]]


def _strategy_greedy(G: nx.Graph, k: int) -> list:
    selected = []
    covered  = set()

    for _ in range(k):
        best_node, best_gain = None, -1

        for node in G.nodes():
            if node in selected:
                continue
            gain = len(set(G.neighbors(node)) - covered - {node})
            if gain > best_gain:
                best_gain = gain
                best_node = node

        if best_node is None:
            break

        selected.append(best_node)
        covered.update(G.neighbors(best_node))
        covered.add(best_node)

    return selected


def _strategy_min_cut(G: nx.Graph, k: int) -> list:
    nodes    = list(G.nodes())
    n        = len(nodes)
    if n == 0:
        return []

    node_idx = {node: i for i, node in enumerate(nodes)}
    INF      = 10 ** 9

    S     = 2 * n
    T     = 2 * n + 1
    total = 2 * n + 2

    dinic = _DinicMaxFlow(total)

    for i in range(n):
        dinic.add_edge(2 * i, 2 * i + 1, 1)

    for u, v in G.edges():
        ui, vi = node_idx[u], node_idx[v]
        dinic.add_edge(2 * ui + 1, 2 * vi,     INF)
        dinic.add_edge(2 * vi + 1, 2 * ui,     INF)

    source_node = max(G.degree(), key=lambda x: x[1])[0]
    dinic.add_edge(S, 2 * node_idx[source_node], INF)

    leaves = [node_idx[nd] for nd in nodes if G.degree(nd) == 1]
    if not leaves:
        leaves = sorted(range(n), key=lambda i: G.degree(nodes[i]))[: max(1, n // 4)]
    for li in leaves:
        dinic.add_edge(2 * li + 1, T, INF)

    dinic.max_flow(S, T)

    reachable  = dinic.reachable_from(S)
    cut_nodes  = []

    for i, node in enumerate(nodes):
        if 2 * i in reachable and 2 * i + 1 not in reachable:
            cut_nodes.append(node)

    if len(cut_nodes) < k:
        degree_fallback = _strategy_degree(G, k)
        seen = set(cut_nodes)
        for node in degree_fallback:
            if node not in seen:
                cut_nodes.append(node)
                seen.add(node)

    return cut_nodes[:k]


class _DinicMaxFlow:

    def __init__(self, n: int):
        self.n     = n
        self.graph = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: int):
        self.graph[u].append([v, cap, len(self.graph[v])])
        self.graph[v].append([u, 0,   len(self.graph[u]) - 1])

    def _bfs(self, s: int, t: int) -> bool:
        self.level    = [-1] * self.n
        self.level[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v, cap, _ in self.graph[u]:
                if cap > 0 and self.level[v] < 0:
                    self.level[v] = self.level[u] + 1
                    q.append(v)
        return self.level[t] >= 0

    def _dfs(self, u: int, t: int, pushed: int, it: list) -> int:
        if u == t:
            return pushed
        while it[u] < len(self.graph[u]):
            e       = self.graph[u][it[u]]
            v, cap, rev = e
            if cap > 0 and self.level[v] == self.level[u] + 1:
                d = self._dfs(v, t, min(pushed, cap), it)
                if d > 0:
                    e[1] -= d
                    self.graph[v][rev][1] += d
                    return d
            it[u] += 1
        return 0

    def max_flow(self, s: int, t: int) -> int:
        flow = 0
        while self._bfs(s, t):
            it = [0] * self.n
            while True:
                f = self._dfs(s, t, 10 ** 9, it)
                if f == 0:
                    break
                flow += f
        return flow

    def reachable_from(self, s: int) -> set:
        visited = {s}
        q       = deque([s])
        while q:
            u = q.popleft()
            for v, cap, _ in self.graph[u]:
                if cap > 0 and v not in visited:
                    visited.add(v)
                    q.append(v)
        return visited


def _final_infected(sim_result: dict) -> int:
    trends = sim_result["trends"]
    return trends["infected"][-1] + trends["removed"][-1]

